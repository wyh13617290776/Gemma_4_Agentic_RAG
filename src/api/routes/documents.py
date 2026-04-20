"""
文档上传与向量化接口路由。

待集成：tools/doc_parser.py 中的 process_and_embed_documents()。
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter(tags=["Documents"])


@router.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    文档上传与向量化接口。

    复用链路：
        UploadFile → doc_parser.process_and_embed_documents()
                   → MinerU/LlamaIndex 解析 → EmbeddingService → Milvus 存储
    """
    if not files:
        raise HTTPException(status_code=400, detail="未提供任何文件")

    # TODO: 集成 process_and_embed_documents
    # from tools.doc_parser import process_and_embed_documents
    # result = await process_and_embed_documents(files)
    # return {"status": "success", "processed": len(files)}
    raise NotImplementedError("Documents API 待集成 doc_parser，当前使用 Streamlit UI 入口")
