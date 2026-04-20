import json
import logging
from core.config import PROMPTS
from core.llm_gateway import LLMGateway

logger = logging.getLogger("AgenticRAG")


class QueryTransformer:
    def __init__(self):
        # 指定任务为查询转换，网关会自动加载 0.6 的温度和 40 的 Top-K
        self.gateway = LLMGateway(task_name="query_transformation")

    async def transform(self, raw_query: str):
        """
        独立出来的核心中间件：后台智能拆解并发重写
        返回: sub_queries (list), hyde_passage (str)
        """
        try:
            messages = [
                {"role": "system", "content": PROMPTS["query_transform_system"]},
                {"role": "user", "content": raw_query}
            ]
            json_str = self.gateway.invoke(
                messages=messages,
                response_format={"type": "json_object"},
                stream=False
            )
            res_dict = json.loads(json_str)
            return res_dict.get("sub_queries", []), res_dict.get("hyde_passage", "")

        except Exception as e:
            logger.warning(f"⚠️ Query Transform 失败: {e}")
            return [raw_query], ""
