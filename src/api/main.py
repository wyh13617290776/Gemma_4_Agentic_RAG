"""
FastAPI 接口层入口。

与 web_ui.py 共享同一套 core/engines/agents 业务逻辑，互不干扰。

启动方式：
    cd src
    uvicorn api.main:app --reload --port 8080
"""
from fastapi import FastAPI
from .routes.chat import router as chat_router
from .routes.documents import router as documents_router

app = FastAPI(
    title="Gemma 4 Agentic RAG API",
    description="企业级多模态 RAG 系统 REST 接口，与 Streamlit UI 共享同一套业务引擎。",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """服务健康探针"""
    return {"status": "ok", "service": "Gemma 4 Agentic RAG API"}
