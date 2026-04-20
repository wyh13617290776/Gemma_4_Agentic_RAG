import os
import sys
import httpx
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# 核心依赖：从中央配置中心导入解析好的全局变量和路由函数
from core.config import ROUTER, TASKS, get_model_generation_params

load_dotenv()

class LLMGateway:
    """
    企业级智能大模型网关 (Enterprise LLM Gateway)
    
    该类作为系统与所有大模型（本地或云端）交互的唯一入口。
    支持基于任务的参数自动注入、多厂商适配器剥离、以及防代理劫持的物理直连逻辑。
    """
    
    def __init__(self, task_name="llm_chat", **kwargs):
        """
        任务驱动型网关初始化
        
        Args:
            task_name (str): 任务标识符。决定参数的初始来源。
                - "llm_chat": 正常对话任务，执行三级配置继承。
                - "intent_routing": 意图识别，强制低采样。
                - "memory_summary": 记忆压缩，强制高忠实度。
            **kwargs: 瞬时覆盖参数。通常来自 UI 侧边栏的手动微调，优先级最高。
        """
        self.task_name = task_name

        # 1. 参数路由：根据任务类型执行不同的加载策略
        if task_name == "llm_chat":
            # 策略：对话任务。调用配置中心的三级继承解析函数 (模型微调 > 系列模板 > 全局默认)
            active_model = st.session_state.get("STREAMLIT_ACTIVE_MODEL", "gemma-4-local")
            self.params = get_model_generation_params(active_model)
        else:
            # 策略：系统内部任务。直接从任务模板库中抓取专业参数
            self.params = TASKS.get(task_name, {}).copy()
            # 针对云端 API 的 top_k 字段兼容性处理
            if "extra_body" in self.params:
                self.params["top_k"] = self.params["extra_body"].get("top_k", 64)

        # 2. 最终合并：手动参数 (kwargs) 具有该实例内的最高否决权
        self.params.update(kwargs)
        
        # 3. 确定当前上下文中的模型名称
        self.model_name = st.session_state.get("STREAMLIT_ACTIVE_MODEL", "gemma-4-local")

    def _get_client_and_model(self):
        """
        根据当前状态动态生成 OpenAI 兼容客户端
        
        该函数会实时读取 ROUTER 配置，确保模型切换时 API 地址与密钥同步更新。
        
        Returns:
            tuple: (OpenAI 客户端对象, 物理模型 ID 字符串)
        
        Raises:
            ValueError: 当模型未在 model_router.yaml 中注册时抛出。
        """
        active_model = st.session_state.get("STREAMLIT_ACTIVE_MODEL", "gemma-4-local")
        model_info = ROUTER.get("models", {}).get(active_model, {})
        
        if not model_info:
            raise ValueError(f"模型 {active_model} 未在 model_router.yaml 中注册！")
        
        # 实例化底层客户端：强制开启直连模式以绕过系统代理干扰
        client = OpenAI(
            api_key=os.getenv(model_info.get("key_env", "NONE"), "sk-dummy"),
            base_url=model_info.get("base_url"),
            # 物理直连保险：防代理劫持导致的网络不可达
            http_client=httpx.Client(proxy=None, trust_env=False)
        )
        return client, active_model

    @property
    def is_local_active(self) -> bool:
        """判定当前是否处于本地推理模式"""
        return self.model_name == ROUTER.get("default_model", "gemma-4-local")

    def invoke(self, messages: list, **kwargs):
        """
        同步调用入口：适用于后台 Agent 决策、JSON 生成等非流式场景
        
        Args:
            messages (list): 符合 OpenAI 格式的对话历史列表。
            **kwargs: 额外的透传参数，如 request_thinking 开关或 tools 定义。
            
        Returns:
            Union[str, object]: 
                - 如果触发了工具调用，返回 OpenAI Message 对象供上层处理。
                - 如果是正常回复，返回提取后的字符串内容 (Dehydrated Content)。
        """
        # 1. 初始化运行时客户端
        client, model_id = self._get_client_and_model()
        model_info = ROUTER.get("models", {}).get(model_id, {})
        adapter_type = model_info.get("adapter", "standard")
        
        # 2. 消费 UI 特有控制意图
        request_thinking = kwargs.pop("request_thinking", False)
        
        # 3. 合并运行时参数
        call_params = self.params.copy()
        call_params.update(kwargs)
        
        # 4. 动态注入特权参数 (如 DeepSeek V3.2 的思考模式或本地模型的 Stop 词)
        if model_info.get("dynamic_thinking") and request_thinking:
            extra = call_params.get("extra_body", {})
            extra["enable_thinking"] = True
            call_params["extra_body"] = extra
            
        if adapter_type == "local_think":
            call_params["stop"] = ["<turn|>", "<|turn|>", "<eos>", "<end_of_turn>"]
        
        # =========================================================
        # 修复：云端 API 严格校验防爆盾 (Parameter Sanitization)
        # =========================================================
        if "extra_body" in call_params and "enable_thinking" in call_params["extra_body"]:
            # 修复：放行 DeepSeek V3/V3.2 的动态思维链开关，仅对原生具备推理标签的模型剥离此参数
            if "r1" in model_id.lower() or "reasoner" in model_id.lower() or "qvq" in model_id.lower():
                call_params["extra_body"].pop("enable_thinking", None)
            # 如果清理后 extra_body 为空，直接干掉这个字典，防止连空字典都被拦截
            if not call_params["extra_body"]:
                call_params.pop("extra_body", None)

        # 5. 发起请求
        response = client.chat.completions.create(
            model=model_id, 
            messages=messages, 
            temperature=call_params.get("temperature", 0.7),
            top_p=call_params.get("top_p", 0.95),
            max_tokens=call_params.get("max_tokens", 4096),
            # 兼容性封装：将 top_k 统一压入 extra_body
            extra_body={"top_k": call_params.get("top_k", 64), **call_params.get("extra_body", {})},
            **{k: v for k, v in call_params.items() if k not in ["temperature", "top_p", "max_tokens", "top_k", "extra_body"]}
        )
        message = response.choices[0].message
        
        # 6. 后处理逻辑：工具调用直传
        if getattr(message, "tool_calls", None):
            return message
            
        # 7. 内容提取：剥离 local_think 产生的内联思考标签
        content = message.content or ""
        if adapter_type == "local_think" and "</think>" in content:
            content = content.split("</think>")[-1].strip()
            
        return content

    def stream_invoke(self, messages: list, **kwargs):
        """
        流式调用入口：专为 Web UI 对话设计，支持思维链的实时剥离与并行渲染
        
        Args:
            messages (list): 对话历史。
            **kwargs: 包含 request_thinking 等 UI 指令。
            
        Yields:
            dict: 包含以下字段的标准化字典：
                - "reasoning" (str): 实时生成的思考过程片段。
                - "content" (str): 实时生成的正文回复片段。
        """
        client, model_id = self._get_client_and_model()
        model_info = ROUTER.get("models", {}).get(model_id, {})
        adapter_type = model_info.get("adapter", "standard")
        
        request_thinking = kwargs.pop("request_thinking", False)
        
        call_params = self.params.copy()
        call_params.update(kwargs)
        
        # 云端推理注入
        if model_info.get("dynamic_thinking") and request_thinking:
            extra = call_params.get("extra_body", {})
            extra["enable_thinking"] = True
            call_params["extra_body"] = extra

        # 本地模型 Stop 词防护：
        # 注意：不能把 <end_of_turn> 加入 stop 列表！
        # Gemma 4 在思考模式下，</think> 标签之前就可能触发 <end_of_turn>，
        # 导致整个思考块被截断，正文永远无法输出。
        if adapter_type == "local_think":
            call_params["stop"] = ["<turn|>", "<|turn|>", "<eos>"]

        # =========================================================
        # 修复：云端 API 严格校验防爆盾 (Parameter Sanitization)
        # =========================================================
        if "extra_body" in call_params and "enable_thinking" in call_params["extra_body"]:
            # 修复：放行 DeepSeek V3/V3.2 的动态思维链开关，仅对原生具备推理标签的模型剥离此参数
            if "r1" in model_id.lower() or "reasoner" in model_id.lower() or "qvq" in model_id.lower():
                call_params["extra_body"].pop("enable_thinking", None)
            # 如果清理后 extra_body 为空，直接干掉这个字典，防止某些厂商连空字典都拦截
            if not call_params["extra_body"]:
                call_params.pop("extra_body", None)

        # 发起流式请求
        stream = client.chat.completions.create(
            model=model_id, 
            messages=messages, 
            temperature=call_params.get("temperature", 0.7),
            top_p=call_params.get("top_p", 0.95),
            max_tokens=call_params.get("max_tokens", 4096),
            stream=True,
            extra_body={"top_k": call_params.get("top_k", 64), **call_params.get("extra_body", {})},
            **{k: v for k, v in call_params.items() if k not in ["temperature", "top_p", "max_tokens", "top_k", "extra_body", "stream"]}
        )
        
        # 🚀 适配器分流处理 (三轨策略)
        
        # A 轨：DeepSeek 专属 (保持您已修复的 V3.2 兼容逻辑)
        if adapter_type == "deepseek":
            for chunk in stream:
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if not reasoning and hasattr(delta, "model_extra") and delta.model_extra:
                    reasoning = delta.model_extra.get("reasoning_content", "")
                content = delta.content or ""
                yield {"reasoning": reasoning or "", "content": content}
        
        # 👑 B 轨：本地模型专属 (网关级内联标签剥离 - 增强防泄露版)
        elif adapter_type == "local_think":
            is_thinking = False
            buffer = ""
            # 定义所有可能的标签及其前缀
            all_tags = ["<think>", "<|think|>", "</think>", "</|think|>"]
            # 自动生成所有可能的标签前缀（如 "<", "<t", "<th" ...）用于拦截
            tag_prefixes = {tag[:i] for tag in all_tags for i in range(1, len(tag))}

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                buffer += token
                
                # 1. 递归处理缓冲区中的完整标签
                found_tag = True
                while found_tag:
                    found_tag = False
                    for tag in all_tags:
                        if tag in buffer:
                            parts = buffer.split(tag, 1)
                            # 如果是结束标签，先把之前的 buffer 作为 reasoning 吐出
                            if tag.startswith("</"):
                                if parts[0]: yield {"reasoning": parts[0], "content": ""}
                                is_thinking = False
                            # 如果是开始标签，先把之前的 buffer 作为 content 吐出
                            else:
                                if parts[0]: yield {"reasoning": "", "content": parts[0]}
                                is_thinking = True
                            
                            buffer = parts[1] # 剩余部分留存
                            found_tag = True
                            break
                
                # 2. 👑 核心修复：安全探测。检查 buffer 结尾是否匹配任何标签的前缀
                # 如果 buffer 结尾是 "<" 或 "<t" 等，我们必须 hold 住，不能 yield
                is_incomplete = any(buffer.endswith(p) for p in tag_prefixes)
                
                if not is_incomplete and buffer:
                    if is_thinking:
                        yield {"reasoning": buffer, "content": ""}
                    else:
                        yield {"reasoning": "", "content": buffer}
                    buffer = ""
            
            # 3. 扫尾逻辑
            if buffer:
                if is_thinking: yield {"reasoning": buffer, "content": ""}
                else: yield {"reasoning": "", "content": buffer}

        # C 轨：其他标准模型
        else:
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                yield {"reasoning": "", "content": content}