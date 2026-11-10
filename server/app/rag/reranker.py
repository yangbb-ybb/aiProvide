"""重排。

fastembed TextCrossEncoder + BAAI/bge-reranker-v2-m3（ONNX，约 1.1GB）。

RAG 检索质量的最大单一杠杆：召回阶段用向量 + BM25 粗筛，
再用 cross-encoder 精排，效果提升通常远大于换生成模型。

RERANK_ENABLED=false 时整体跳过 —— 既是磁盘 / 内存紧张时的降级开关，
也是 A/B 评测的对照组。
"""

from __future__ import annotations

from .retriever import ScoredChunk


class Reranker:
    """bge-reranker-v2-m3 重排器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        """重排并取 top_k。禁用时直接返回原列表的前 top_k 个。"""
        raise NotImplementedError("M1 阶段实现：重排")
