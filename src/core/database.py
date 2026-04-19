import time
from core.config import CFG
from core.hardware import HardwareManager
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
from llama_index.vector_stores.milvus import MilvusVectorStore

logger = logging.getLogger("AgenticRAG")

# ==========================================
# 1. 向量模型服务
# ==========================================
class EmbeddingService:
    _is_loaded = False
    _last_query_time = 0  

    @staticmethod
    def load(device="cuda"):
        # 记录/刷新活动时间
        EmbeddingService._last_query_time = time.time()

        if not EmbeddingService._is_loaded or Settings.embed_model is None:
            logger.info("📡 正在装载 BGE-M3 向量模型到显存...")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=CFG["embedding"]["model_path"],
                device=device
            )
            EmbeddingService._is_loaded = True
            logger.info("✅ BGE-M3 模型已就绪，进入超时保活状态")

    @staticmethod
    def unload(reason="timeout"):
        """
        卸载向量模型
        :param reason: "timeout" (默认，看门狗超时触发) 或 "immediate" (解析文档时手动强制腾显存)
        """
        if EmbeddingService._is_loaded:
            Settings.embed_model = None
            EmbeddingService._is_loaded = False
            EmbeddingService._last_query_time = 0
            HardwareManager.free_vram()
            
            if reason == "immediate":
                logger.info("♻️ BGE 向量模型显存已释放")
            else:
                logger.info("♻️ 长时间未使用，BGE 向量模型显存已安全释放")
    
    @staticmethod
    def get_idle_time():
        """获取当前已空闲的时长（秒）"""
        if not EmbeddingService._is_loaded:
            return 0
        return time.time() - EmbeddingService._last_query_time

# ==========================================
# 2. 向量数据库服务 (负责连接 Milvus 存储和检索数据)
# ==========================================
class DatabaseService:
    @staticmethod
    def get_vector_store():
        return MilvusVectorStore(
            uri=CFG["milvus"]["uri"],  
            dim=CFG["milvus"]["dim"],
            collection_name=CFG["rag"]["collection_name"],
            overwrite=False
        )