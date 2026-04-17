import json
import httpx
from openai import OpenAI
# 引入：提示词字典、工具字典
from core.config import PROMPTS, TOOLS
from core.llm_gateway import LLMGateway

class IntentRouter:
    def __init__(self):
        # 明确指定任务为意图路由，网关会自动加载 0.1 的极低温度
        self.gateway = LLMGateway(task_name="intent_routing")

        # 硅基流动适配：如果用户在 UI 选了 R1 模型，后台强行降级为 V3，因为 R1 不支持工具调用
        if "deepseek-r1" in self.gateway.model_name.lower():
            self.gateway.model_name = "deepseek-ai/DeepSeek-V3" # 强制降级到同平台的 V3

    def analyze_intent(self, user_query: str, has_media: bool, enable_web_search: bool = True) -> dict:
        """
        使用 Function Calling 分析用户意图，支持并行调用提取多重意图。
        返回格式: {"intents": ["search", "chat"], "parameters": {"search_query": "...", "topic": "..."}}
        """
        if has_media and not user_query.strip():
            return {"intents": ["analyze_image"], "parameters": {}}
        
        # 动态过滤工具列表。如果用户关闭了联网，直接从物理层面没收 web_search 工具
        active_tools = []
        for tool in TOOLS:
            if not enable_web_search and tool["function"]["name"] == "web_search":
                continue
            active_tools.append(tool)
            
        # 如果所有工具都被禁用了，直接走无工具模式
        tool_choice = "auto" if active_tools else "none"

        try:
            messages = [
                {"role": "system", "content": PROMPTS["router_system_prompt"]},
                {"role": "user", "content": PROMPTS["router_user_prompt"].format(has_media=str(has_media), user_query=user_query)}
            ]
            
            # 所有的特殊参数直接塞进 kwargs
            message = self.gateway.invoke(
                messages=messages,
                tools=active_tools if active_tools else None,
                tool_choice=tool_choice,
                stream=False
            )
            
            intents = []
            parameters = {}
            
            # 正常解析 tool_calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    intents.append(tool_call.function.name)
                    try:
                        args = json.loads(tool_call.function.arguments)
                        parameters.update(args)
                    except Exception as e:
                        print(f"参数解析跳过: {e}")
                return {"intents": intents, "parameters": parameters}
            else:
                return {"intents": ["chat"], "parameters": {}}
            
        except Exception as e:
            print(f"⚠️ Function Calling 路由解析失败，降级为默认闲聊。错误: {e}")
            return {"intents": ["chat"], "parameters": {}}