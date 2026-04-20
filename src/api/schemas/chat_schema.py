from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户输入的问题或指令")
    session_id: str = Field(default="default", description="会话 ID，用于隔离多用户上下文")
    enable_web_search: bool = Field(default=True, description="是否开启全域联网检索")
    model: str = Field(default="gemma-4-local", description="指定使用的模型 ID（需在 model_router.yaml 中注册）")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="模型的最终回答")
    reasoning: Optional[str] = Field(default=None, description="思维链内容（仅思考型模型返回）")
    sources: list[str] = Field(default_factory=list, description="引用的文档来源列表")
    session_id: str = Field(..., description="本次响应对应的会话 ID")
    intents: list[str] = Field(default_factory=list, description="系统识别出的用户意图列表")
