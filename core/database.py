from core.config import CFG
from core.hardware import HardwareManager
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

class EmbeddingService:
    _is_loaded = False

    @staticmethod
    def load(device="cuda"):
        if not EmbeddingService._is_loaded or Settings.embed_model is None:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=CFG["embedding"]["model_path"],
                device=device
            )
            EmbeddingService._is_loaded = True

    @staticmethod
    def unload():
        Settings.embed_model = None
        EmbeddingService._is_loaded = False
        HardwareManager.free_vram()

class DatabaseService:
    @staticmethod
    def get_vector_store():
        return MilvusVectorStore(
            uri=CFG["milvus"]["uri"],  
            dim=CFG["milvus"]["dim"],
            collection_name=CFG["rag"]["collection_name"],
            overwrite=False
        )