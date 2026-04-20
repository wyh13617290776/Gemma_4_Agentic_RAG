"""
记忆管理器：统一调度短期/长期/压缩三层记忆的对外接口。

设计原则：
- 不依赖 Streamlit session_state，通过构造函数注入任意 dict-like 状态容器
- 与原 chat_memory.py 接口完全兼容，无需修改调用方
- 新增 try_compress() 异步接口，供 FastAPI 或后台任务调用
"""
import re
import asyncio
import logging
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .compressor import MemoryCompressor
from core.config import CFG

logger = logging.getLogger("AgenticRAG")


def strip_thinking(text: str) -> str:
    """剥离思维链标签（全局正则升级版，兼容所有变体格式）"""
    if not text:
        return ""
    
    text = re.sub(r"<(?:\|)?think(?:\|)?>.*?</(?:\|)?think(?:\|)?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 清理杂散的特殊 token
    for tag in ["<turn|>", "<|turn|>", "<eos>"]:
        text = text.replace(tag, "")
    return text.strip()


# 保留原函数名，兼容 web_ui.py 中的调用
strip_thinking_from_history = strip_thinking


class MemoryManager:
    """
    三层记忆统一调度器。

    Args:
        state_container: 任意支持 dict 协议的状态容器
                         - Streamlit：直接传入 st.session_state
                         - FastAPI / 测试：传入普通 dict
        max_window: 短期记忆滑动窗口轮数，None 时从 config.yaml 读取
    """

    def __init__(self, state_container, max_window: int = None):
        # 兼容 Streamlit SessionState（支持属性访问）和普通 dict（支持 key 访问）
        self._state = state_container
        self.summary_threshold = CFG["memory"].get("summary_threshold", 5)

        # 初始化状态容器中的必要字段
        self._ensure_key("messages", [])
        self._ensure_key("current_summary", "")

        # 短期记忆：_messages 直接引用 state["messages"]，保证单一数据源
        self.short_term = ShortTermMemory(max_window)
        self.short_term._messages = self._get("messages")

        # 长期记忆：从 state 恢复已有摘要
        self.long_term = LongTermMemory()
        existing_summary = self._get("current_summary")
        if existing_summary:
            self.long_term.store_summary(existing_summary)

        self.compressor = MemoryCompressor()

    # ——— 内部辅助：兼容 Streamlit SessionState 和 dict ———

    def _ensure_key(self, key: str, default):
        """确保状态容器中存在指定 key（兼容 st.session_state 属性访问）"""
        try:
            # Streamlit SessionState：用 in 运算符检测
            if key not in self._state:
                setattr(self._state, key, default) if hasattr(self._state, '__setattr__') else None
                self._state[key] = default
        except (TypeError, AttributeError):
            pass

    def _get(self, key: str, default=None):
        """兼容属性访问（st.session_state.messages）和 key 访问（dict["messages"]）"""
        try:
            return self._state[key]
        except (TypeError, KeyError):
            return getattr(self._state, key, default)

    def _set(self, key: str, value):
        """兼容属性写入和 key 写入"""
        try:
            self._state[key] = value
        except TypeError:
            setattr(self._state, key, value)

    # ——— 对外接口（与原 chat_memory.py 完全兼容） ———

    def get_ui_messages(self) -> list:
        """返回全部消息列表（用于 UI 渲染）"""
        return self._get("messages", [])

    def add_user_message(self, content):
        msgs = self._get("messages", [])
        msgs.append({"role": "user", "content": content, "thought": ""})
        self.short_term._messages = msgs

    def add_assistant_message(self, thought: str, content: str):
        msgs = self._get("messages", [])
        msgs.append({"role": "assistant", "thought": thought, "content": content})
        self.short_term._messages = msgs

    def update_last_message(self, thought: str, content: str):
        """更新最后一条 assistant 消息（流式输出结束后调用）"""
        self.short_term.update_last(thought, content)

    def get_llm_payload(self, current_user_content, strip_multimodal: bool = False) -> list:
        """
        组装 LLM 请求 payload：
        [长期摘要 system] + [短期窗口历史] + [当前用户输入]
        """
        payload = []

        # 1. 注入长期摘要（如果存在）
        if self.long_term.has_summary():
            payload.append({
                "role": "system",
                "content": f"[历史对话摘要]\n{self.long_term.get_summary()}"
            })

        # 2. 注入滑动窗口（排除最后一条，避免重复）
        window = self.short_term.get_window()
        history_window = window[:-1] if window else []

        for msg in history_window:
            role = msg["role"]
            content = msg["content"]
            # 历史多模态一律只保留文字，节省显存
            if role == "user" and isinstance(content, list):
                content = next((i["text"] for i in content if i["type"] == "text"), "")
            elif role == "assistant":
                content = strip_thinking(content)
            if content:
                payload.append({"role": role, "content": content})

        # 3. 当前轮用户输入 (新增：多模态防爆显存剥离逻辑)
        if strip_multimodal and isinstance(current_user_content, list):
            # 强制过滤掉所有非文本块 (物理级砍掉 image_url/audio/video)
            clean_content = [item for item in current_user_content if item["type"] == "text"]
            payload.append({"role": "user", "content": clean_content})
        else:
            payload.append({"role": "user", "content": current_user_content})
            
        return payload

    # ——— 兼容原 chat_memory.py 的摘要相关接口 ———

    def need_summarize(self) -> bool:
        """判断是否需要触发摘要压缩（总轮数 >= 阈值时触发）"""
        messages = self._get("messages", [])
        total_turns = len(messages) // 2
        return total_turns >= self.summary_threshold

    def get_summary_prompt(self) -> str:
        """获取当前摘要，拼接到 System Prompt 中"""
        summary = self._get("current_summary", "")
        if summary:
            return f"\n\n【前情提要（历史对话摘要）】：\n{summary}"
        return ""

    def update_summary(self, new_summary: str):
        """更新持久化摘要（后台线程回调调用）"""
        self.long_term.store_summary(new_summary)
        self._set("current_summary", self.long_term.get_summary())
        
        # ==========================================
        # 新增：硬核控制台探针，验证记忆是否入库
        # ==========================================
        # print("\n" + "="*50)
        # print("🧠 [记忆引擎] 触发长期记忆静默压缩并入库成功！")
        # print(f"📄 当前累积的长期摘要内容如下：\n{self.long_term.get_summary()}")
        # print("="*50 + "\n")
        
        logger.info("✅ [记忆] 长期记忆静默更新完成")

    # ——— 新增：异步压缩接口（供 FastAPI 或 async 上下文使用） ———

    async def try_compress(self):
        """当消息超过阈值时，后台异步触发记忆压缩（无需手动管理线程）"""
        messages = self._get("messages", [])
        if len(messages) >= self.summary_threshold * 2:
            old_messages = messages[: -self.summary_threshold * 2]
            summary = await self.compressor.compress(old_messages)
            if summary:
                self.update_summary(summary)
