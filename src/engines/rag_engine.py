import os
import pickle
import math
import concurrent.futures
import logging
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from core.database import DatabaseService, EmbeddingService
from core.config import CFG, PROMPTS

from llama_index.core import Settings
from llama_index.core.llms import MockLLM

logger = logging.getLogger("AgenticRAG")

# ==========================================
# Sigmoid 映射与悬崖截断算法
# ==========================================
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def smart_filter_nodes(reranked_nodes, base_threshold=0.15, max_dropoff=0.2):
    if not reranked_nodes:
        return []

    valid_nodes = []
    for node in reranked_nodes:
        node.score = sigmoid(node.score)  # 将 logit 转为 0~1 的置信度
        if node.score >= base_threshold:
            valid_nodes.append(node)

    valid_nodes.sort(key=lambda x: x.score, reverse=True)
    if len(valid_nodes) <= 1:
        return valid_nodes

    cutoff_idx = len(valid_nodes)
    for i in range(len(valid_nodes) - 1):
        drop = valid_nodes[i].score - valid_nodes[i + 1].score
        # 寻找相对分数断层
        if drop >= max_dropoff:
            cutoff_idx = i + 1
            logger.info(f"🔪 触发悬崖截断！在第 {cutoff_idx} 名处发现落差 {drop:.2f}")
            break

    return valid_nodes[:cutoff_idx]


class RAGPipeline:
    def __init__(self):
        Settings.llm = MockLLM()
        self.nodes_file = CFG["rag"]["retrieval"]["bm25_nodes_path"]
        self.fusion_top_k = CFG["rag"]["retrieval"]["fusion_top_k"]
        self.rerank_top_k = CFG["rag"]["retrieval"]["rerank_top_k"]
        self.reranker_model_path = CFG["rag"]["reranker"]["model_path"]
        self.reranker_use_fp16 = CFG["rag"]["reranker"]["use_fp16"]

    # ==========================================
    # RAG 引擎执行器：接收外部的纯净弹药
    # ==========================================
    def retrieve_and_format(self, search_query: str, sub_queries: list = None, hyde_passage: str = ""):
        # 防止外部传 None
        if sub_queries is None:
            sub_queries = []

        EmbeddingService.load(device="cuda")
        index = VectorStoreIndex.from_vector_store(DatabaseService.get_vector_store())

        all_nodes = []
        if os.path.exists(self.nodes_file):
            with open(self.nodes_file, "rb") as f:
                all_nodes = pickle.load(f)

        vector_retriever = index.as_retriever(similarity_top_k=self.fusion_top_k)

        if all_nodes:
            bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=self.fusion_top_k)
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=self.fusion_top_k,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=False
            )
        else:
            retriever = vector_retriever

        # 1. 组合所有的检索弹药 (原始词 + 拆解短句 + HyDE 伪造答案)
        all_search_strings = [search_query]
        all_search_strings.extend(sub_queries)
        if hyde_passage:
            all_search_strings.append(hyde_passage)

        # 去重并去除空字符串
        all_search_strings = list(dict.fromkeys([q for q in all_search_strings if q.strip()]))

        logger.info(f"🔬 启动并发检索策略池: {all_search_strings}")

        all_retrieved_nodes = []
        # 使用多线程极速撒网
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(retriever.retrieve, q) for q in all_search_strings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    nodes = future.result()
                    all_retrieved_nodes.extend(nodes)
                except Exception as e:
                    logger.warning(f"⚠️ 某路检索崩溃跳过: {e}")

        # 2. 根据 Node ID 去重
        unique_nodes_map = {n.node.node_id: n for n in all_retrieved_nodes}
        unique_nodes = list(unique_nodes_map.values())

        # ==========================================================
        # 3. 联邦名额精排 + 局部独立悬崖截断
        # ==========================================================
        reranker = FlagEmbeddingReranker(
            top_n=self.rerank_top_k,
            model=self.reranker_model_path,
            use_fp16=self.reranker_use_fp16
        )

        quota_per_query = max(4, self.rerank_top_k // len(all_search_strings))
        all_surviving_nodes = []

        for query_str in all_search_strings:
            nodes_for_q = reranker.postprocess_nodes(
                unique_nodes, query_bundle=QueryBundle(query_str)
            )
            filtered_for_q = smart_filter_nodes(nodes_for_q, base_threshold=0.15, max_dropoff=0.35)
            all_surviving_nodes.extend(filtered_for_q[:quota_per_query])

        # ==========================================================
        # 4. 去重与合并 (最高分保留法则)
        # ==========================================================
        final_unique_map = {}
        for n in all_surviving_nodes:
            if n.node.node_id not in final_unique_map:
                final_unique_map[n.node.node_id] = n
            else:
                if n.score > final_unique_map[n.node.node_id].score:
                    final_unique_map[n.node.node_id] = n

        mixed_final_nodes = list(final_unique_map.values())
        mixed_final_nodes.sort(key=lambda x: x.score, reverse=True)
        final_nodes = mixed_final_nodes

        # 5. Small-to-Big 上下文解包
        metadata_replacement = MetadataReplacementPostProcessor(target_metadata_key="window")
        final_nodes = metadata_replacement.postprocess_nodes(final_nodes)

        # 6. 拼装文本
        context_blocks = []
        for i, n in enumerate(final_nodes):
            meta = n.node.metadata
            file_name = meta.get("file_name", "未知文档")
            file_type = meta.get("file_type", "txt").lower()
            upload_time = meta.get("upload_time", "未知时间")

            if file_type == "pdf":
                page_info = meta.get("page_label", "?")
                source_tag = f"[{file_name}, 第{page_info}页, 上传于 {upload_time}]"
                block_text = f"--- 核心片段 ---\n[强制引用标签]: {source_tag}\n[片段内容]:\n{n.get_content()}"
            else:
                source_tag = f"[{file_name}, 上传于 {upload_time}]"
                block_text = f"--- 核心片段 ---\n[强制引用标签]: {source_tag}\n[片段内容]:\n{n.get_content()}"

            context_blocks.append(block_text)

        context_str = "\n\n".join(context_blocks)
        logger.info(f"📚 本地资料检索完成，共 {len(final_nodes)} 个片段")
        return final_nodes, context_str
