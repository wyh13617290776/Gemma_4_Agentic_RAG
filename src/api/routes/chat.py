"""
对话接口路由。

待集成：GraphOrchestrator + MemoryManager，复用 web_ui.py 的完整业务链路。
实现时，session_id 对应一个独立的 MemoryManager(state_container=sessions[session_id]) 实例。
"""
from fastapi import APIRouter
from ..schemas.chat_schema import ChatRequest, ChatResponse

router = APIRouter(tags=["Chat"])

# 简单的内存会话存储（生产环境可替换为 Redis）
_sessions: dict = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    核心对话接口。

    复用链路：
        ChatRequest → MemoryManager → GraphOrchestrator.run()
                    → (RAGPipeline | UltimateWebRetriever) → LLMGateway → ChatResponse
    """
    # TODO: 集成 GraphOrchestrator
    # session = _sessions.setdefault(request.session_id, {})
    # memory = MemoryManager(session)
    # orchestrator = GraphOrchestrator(IntentRouter(), RAGPipeline(), UltimateWebRetriever())
    # state = await orchestrator.run(request.query, has_media=False, enable_web_search=request.enable_web_search)
    # ...
    raise NotImplementedError("Chat API 待集成 GraphOrchestrator，当前使用 Streamlit UI 入口")
