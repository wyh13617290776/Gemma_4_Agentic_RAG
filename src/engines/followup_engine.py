import json
import logging
from core.llm_gateway import LLMGateway
from core.config import ROUTER, PROMPTS

logger = logging.getLogger("AgenticRAG")

class FollowupEngine:
    """动态推荐问题引擎：负责多模型轮询与问题生成"""
    
    def __init__(self):
        # 提取资源池并过滤黑名单
        excluded_models = {"qwen3.6-flash-2026-04-16", "qwen3-vl-flash-2026-01-22"}
        self.flash_pool = [
            m_id for m_id, m_info in ROUTER.get("models", {}).items() 
            if "flash" in m_id.lower() 
            and m_info.get("type") == "qwen"
            and m_id not in excluded_models
        ]

    async def generate(self, user_query: str, assistant_reply: str, current_idx: int) -> tuple[list, int]:
        """
        生成推荐问题，并返回 (问题列表, 下一个轮询指针)
        """
        if not self.flash_pool:
            logger.warning("⚠️ 未在配置中发现 Qwen Flash 模型，跳过推荐生成。")
            return [], current_idx

        try:
            # 游标推进算法
            safe_idx = current_idx % len(self.flash_pool)
            chosen_model = self.flash_pool[safe_idx]
            next_idx = (safe_idx + 1) % len(self.flash_pool)

            logger.info(f"🔄 推荐引擎调度：[{chosen_model}] 接管本轮推演")

            gateway = LLMGateway(task_name="generate_followup")
            gateway.model_name = chosen_model
            
            messages = [
                {"role": "system", "content": PROMPTS["followup_questions_system"]},
                {"role": "user", "content": PROMPTS["followup_questions_user"].format(
                    user_query=user_query,
                    assistant_reply=assistant_reply[:800] 
                )}
            ]
            
            import asyncio
            import sys
            import threading

            # 👑 修复：动态捕获主线程的 Streamlit 上下文 (如果处于 Web UI 环境)
            ctx = None
            if "streamlit" in sys.modules:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                ctx = get_script_run_ctx()

            def _invoke_task():
                # 将捕获的上下文手动挂载到 asyncio 创建的子线程中
                if ctx:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                    add_script_run_ctx(threading.current_thread(), ctx)
                return gateway.invoke(messages=messages, response_format={"type": "json_object"}, stream=False)

            # 让子线程执行包装好的任务
            json_str = await asyncio.to_thread(_invoke_task)
            
            res_data = json.loads(json_str)
            
            return res_data.get("questions", [])[:4], next_idx

        except Exception as e:
            logger.error(f"⚠️ 推荐生成失败: {e}")
            return [], current_idx