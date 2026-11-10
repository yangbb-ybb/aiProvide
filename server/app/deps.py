"""FastAPI 依赖。

这里放的都是**进程级单例**：向量化模型和重排模型加载一次要好几秒，
每个请求重新加载会把首字延迟从 1 秒拖到 30 秒以上。
"""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings
from .llm.ollama import OllamaClient
from .rag.embedder import Embedder
from .rag.reranker import Reranker
from .rag.retriever import Retriever
from .store.bm25_index import BM25Index
from .store.vector_store import VectorStore


def get_ollama() -> OllamaClient:
    """构造 Ollama 客户端。

    全项目只有 llm/ollama.py 与推理后端耦合；要换成 llama.cpp server，
    只需要改这一个文件（两者都是 OpenAI 兼容协议）。
    """
    s = get_settings()
    return OllamaClient(
        host=s.ollama_host,
        model=s.ollama_chat_model,
        keep_alive=s.ollama_keep_alive,
    )


@lru_cache
def get_embedder() -> Embedder:
    """向量化器单例。模型本身是懒加载的，这里只是造壳子。"""
    s = get_settings()
    return Embedder(
        model_name=s.embed_model,
        dim=s.vector_dim,
        query_instruction=s.embed_query_instruction,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    """向量仓库单例。"""
    s = get_settings()
    return VectorStore(s.lancedb_path, dim=s.vector_dim)


@lru_cache
def get_bm25_index() -> BM25Index:
    """BM25 索引单例。首次调用时从磁盘加载。"""
    s = get_settings()
    index = BM25Index(s.bm25_path)
    index.load()
    return index


@lru_cache
def get_reranker() -> Reranker:
    """重排器单例。RERANK_ENABLED=false 时它自己会跳过。"""
    s = get_settings()
    return Reranker(model_name=s.rerank_model, enabled=s.rerank_enabled)


@lru_cache
def get_retriever() -> Retriever:
    """召回器单例。"""
    s = get_settings()
    return Retriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25_index(),
        reranker=get_reranker(),
        top_n=s.retrieve_top_n,
        rrf_k=s.rrf_k,
        rerank_top_k=s.rerank_top_k,
    )
