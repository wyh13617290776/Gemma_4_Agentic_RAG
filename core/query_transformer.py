# core/query_transformer.py
import json
import httpx
from openai import OpenAI
from core.config import CFG, PROMPTS

class QueryTransformer:
    def __init__(self):
        # 建议使用与 Router 相同的极速本地/API模型配置
        self.client = OpenAI(
            base_url=f"http://{CFG['llm_server']['host']}:{CFG['llm_server']['port']}/v1",
            api_key="sk-local",
            http_client=httpx.Client(proxy=None, trust_env=False)
        )

    async def transform(self, raw_query: str): 
        """
        独立出来的核心中间件：后台智能拆解并发重写
        返回: sub_queries (list), hyde_passage (str)
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # 替换为您的模型名
                messages=[
                    {"role": "system", "content": PROMPTS["query_transform_system"]},
                    {"role": "user", "content": raw_query}
                ],
                response_format={ "type": "json_object" },
                temperature=0.2 # 提纯需要高确定性
            )
            res_dict = json.loads(response.choices[0].message.content)
            return res_dict.get("sub_queries", []), res_dict.get("hyde_passage", "")
        except Exception as e:
            print(f"⚠️ Query Transform 失败: {e}")
            # 兜底：如果拆解失败，直接原句返回
            return [raw_query], ""