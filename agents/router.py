import json
import httpx
from openai import OpenAI
# 👑 引入底层的三个巨头：基础配置、提示词字典、工具字典
from core.config import CFG, PROMPTS, TOOLS

class IntentRouter:
    def __init__(self):
        # 路由器自带一个轻量级客户端实例，用于极速意图判定
        self.client = OpenAI(
            base_url=f"http://{CFG['llm_server']['host']}:{CFG['llm_server']['port']}/v1", 
            api_key="sk-local",
            http_client=httpx.Client(proxy=None, trust_env=False)
        )

    def analyze_intent(self, user_query: str, has_media: bool, enable_web_search: bool = True) -> dict:
        """
        使用 Function Calling 分析用户意图，支持并行调用提取多重意图。
        返回格式: {"intents": ["search", "chat"], "parameters": {"search_query": "...", "topic": "..."}}
        """
        if has_media and not user_query.strip():
            return {"intents": ["analyze_image"], "parameters": {}}
        
        # 👑 新增：动态过滤工具列表。如果用户关闭了联网，直接从物理层面没收 web_search 工具
        active_tools = []
        for tool in TOOLS:
            if not enable_web_search and tool["function"]["name"] == "web_search":
                continue
            active_tools.append(tool)
            
        # 如果所有工具都被禁用了，直接走无工具模式
        tool_choice = "auto" if active_tools else "none"

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {
                        "role": "system", 
                        "content": PROMPTS["router_system_prompt"]
                    },
                    {
                        "role": "user", 
                        "content": PROMPTS["router_user_prompt"].format(
                            has_media=str(has_media), 
                            user_query=user_query
                        )
                    }
                ],
                tools=active_tools if active_tools else None,  # 👑 传入动态过滤后的可用工具
                tool_choice=tool_choice,
                temperature=0.2,
                stream=False
            )
            
            message = response.choices[0].message
            
            intents = []
            parameters = {}
            
            # 👑 核心升级：遍历大模型返回的所有并行 Tool Calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    intent_name = tool_call.function.name
                    intents.append(intent_name)
                    
                    # 合并所有的参数到一个字典中
                    try:
                        args = json.loads(tool_call.function.arguments)
                        parameters.update(args)
                    except Exception as e:
                        print(f"参数解析跳过: {e}")
                        
                return {"intents": intents, "parameters": parameters}
            else:
                # 如果没用工具，兜底走聊天
                return {"intents": ["chat"], "parameters": {}}
            
        except Exception as e:
            print(f"⚠️ Function Calling 路由解析失败，降级为默认闲聊。错误: {e}")
            return {"intents": ["chat"], "parameters": {}}