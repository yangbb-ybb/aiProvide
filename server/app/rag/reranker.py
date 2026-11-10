"""重排。

fastembed TextCrossEncoder + BAAI/bge-reranker-v2-m3-int8（ONNX，约 0.57GB）。

RAG 检索质量的最大单一杠杆：召回阶段用向量 + BM25 粗筛，
再用 cross-encoder 精排，效果提升通常远大于换生成模型。

选 int8 版本的原因：fp16 的 bge-reranker-v2-m3 要 2.27GB，磁盘只剩 22GB，
装不下也不划算；int8 是 0.57GB，质量损失在 RAG 场景里几乎看不出来。

RERANK_ENABLED=false 时整体跳过 —— 既是磁盘 / 内存紧张时的降级开关，
也是 A/B 评测的对照组。
"""

from __future__ import annotations

import logging
from dataclasses import replace

from .retriever import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker:
    """bge-reranker-v2-m3 重排器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3-int8", enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def _ensure_model(self):
        """懒加载重排模型。"""
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("正在加载重排模型：%s（首次运行需要下载）", self.model_name)
            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        """重排并取 top_k。禁用时直接返回原列表的前 top_k 个。

        **不改原对象**：返回的是新对象，原来的 score / scores 保持不变。
        调试面板要同时看 RRF 分和重排分，就不能被重排结果覆盖掉。
        """
        if not chunks:
            return []
        if not self.enabled:
            return chunks[:top_k]

        model = self._ensure_model()
        passages = [c.content for c in chunks]
        raw = list(model.rerank(query, passages, batch_size=32))

        # fastembed 0.9 的 rerank 返回的是「与输入一一对应的分数」，
        # 既不给下标也不排序，所以排序和 top_k 都得自己做。
        # 老版本返回的是 [{"index":..,"score":..}]，这里一并兼容。
        if raw and isinstance(raw[0], dict):
            pairs = [
                (int(item["index"]), float(item["score"]))
                for item in raw
                if 0 <= int(item["index"]) < len(chunks)
            ]
        else:
            pairs = [(i, float(score)) for i, score in enumerate(raw)]

        pairs.sort(key=lambda pair: -pair[1])

        ranked: list[ScoredChunk] = []
        for index, score in pairs[:top_k]:
            source = chunks[index]
            scores = dict(source.scores)
            scores["rerank"] = score
            ranked.append(replace(source, score=score, scores=scores))
        return ranked
