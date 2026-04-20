import asyncio
import logging
from core.llm_gateway import LLMGateway
from core.config import PROMPTS

logger = logging.getLogger("AgenticRAG")


class MemoryCompressor:
    """
    记忆压缩器：当对话轮数超过阈值时，调用 LLM 将旧历史压缩为摘要。

    从原 chat_memory.py 中的 trigger_background_summary 逻辑独立出来。
    不依赖任何 UI 框架，可在 Streamlit 或 FastAPI 中通用。
    """

    def __init__(self):
        self.gateway = LLMGateway(task_name="memory_summary")

    async def compress(self, messages: list) -> str:
        """
        异步压缩：将消息列表摘要为一段简洁的上下文描述。

        Args:
            messages: 待压缩的消息列表（role/content 格式）

        Returns:
            str: 压缩后的摘要文本，失败时返回空字符串
        """
        if not messages:
            return ""

        history_text = "\n".join(
            f"[{m['role'].upper()}]: {m['content'] if isinstance(m['content'], str) else '[多媒体内容]'}"
            for m in messages
        )
        prompt = PROMPTS.get(
            "memory_summary_prompt",
            "请将以下对话历史压缩为简洁摘要，保留关键信息和用户意图：\n"
        ) + history_text

        try:
            import sys
            import threading
            
            ctx = None
            if "streamlit" in sys.modules:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                ctx = get_script_run_ctx()

            def _compress_task():
                if ctx:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                    add_script_run_ctx(threading.current_thread(), ctx)
                return self.gateway.invoke(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )

            result = await asyncio.to_thread(_compress_task)
            logger.info(f"✅ 记忆压缩完成，摘要长度: {len(result)} 字符")
            return result
        except Exception as e:
            logger.warning(f"⚠️ 记忆压缩失败: {e}")
            return ""
