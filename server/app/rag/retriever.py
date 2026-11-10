"""混合召回。

融合算法自己写（规划文档 §4.4）：

    dense = vector_store.search(query_vec, limit=RETRIEVE_TOP_N)   # LanceDB
    bm25  = bm25_index.search(jieba.lcut(query), limit=RETRIEVE_TOP_N)

    RRF:  score(d) = 1/(K + rank_dense(d)) + 1/(K + rank_bm25(d))
          只出现在一路里的，另一路贡献 0
    → top-20
    → 邻居扩展：命中块同 doc_id 的 chunk_idx ±1
    → 交给 reranker 精排

**不要用 LanceDB 内置的 hybrid search**：它的 FTS 是英文分词器，中文会崩。
LanceDB 在本项目里只当「向量仓库 + 元数据过滤」。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    """带分数的块。scores 里保留各路原始分，方便调试面板展示。"""

    content: str
    chunk_id: str = ""
    source_path: str = ""
    heading_path: list[str] = field(default_factory=list)
    page: int | None = None
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    # ---- 以下用于引用展示与调试面板 ----
    doc_id: str = ""
    title: str = ""
    source_type: str = ""
    chunk_idx: int = 0

    @staticmethod
    def from_row(row: dict) -> "ScoredChunk":
        """从向量库返回的一行构造。"""
        heading_path = row.get("heading_path") or []
        if isinstance(heading_path, str):
            try:
                heading_path = json.loads(heading_path)
            except json.JSONDecodeError:
                heading_path = []
        return ScoredChunk(
            content=row.get("content", ""),
            chunk_id=row.get("chunk_id", ""),
            source_path=row.get("source_path", ""),
            heading_path=list(heading_path),
            page=row.get("page"),
            doc_id=row.get("doc_id", ""),
            title=row.get("title", ""),
            source_type=row.get("source_type", ""),
            chunk_idx=int(row.get("chunk_idx", 0)),
        )


class Retriever:
    """混合召回器。

    构造参数刻意不依赖 config，全靠外部注入，方便单测里塞假组件。
    """

    def __init__(
        self,
        embedder,
        vector_store,
        bm25_index,
        reranker=None,
        top_n: int = 50,
        rrf_k: int = 60,
        fusion_top_n: int = 20,
        rerank_top_k: int = 5,
        neighbor_expand: int = 1,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.top_n = top_n
        self.rrf_k = rrf_k
        self.fusion_top_n = fusion_top_n
        self.rerank_top_k = rerank_top_k
        self.neighbor_expand = neighbor_expand

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """返回精排后的 top_k 个块。"""
        candidates = self.retrieve_raw(query, top_n=self.top_n)
        if not candidates:
            return []
        if self.reranker is None:
            return candidates[:top_k]
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def retrieve_raw(self, query: str, top_n: int = 50) -> list[ScoredChunk]:
        """只做召回和 RRF 融合，不重排。给调试面板和评测用。"""
        limit = min(top_n, self.top_n) if top_n else self.top_n

        dense_rows = self.vector_store.search(self.embedder.embed_query(query), limit=limit)
        bm25_hits = self.bm25_index.search(query, limit=limit)

        # ---- RRF 融合 ----
        # score(d) = 1/(K + rank_dense) + 1/(K + rank_bm25)
        # 只出现在一路里的，另一路贡献 0
        fused: dict[str, dict] = {}

        for rank, row in enumerate(dense_rows, start=1):
            chunk_id = row.get("chunk_id", "")
            if not chunk_id:
                continue
            entry = fused.setdefault(chunk_id, {"row": row, "dense_rank": 0, "bm25_rank": 0})
            entry["dense_rank"] = rank
            entry["row"] = row

        bm25_only = [cid for cid, _ in bm25_hits if cid not in fused]
        if bm25_only:
            # 只在关键词路出现的块，向量库里也要把元数据捞出来
            for row in self.vector_store.get_chunks(bm25_only):
                entry = fused.setdefault(
                    row.get("chunk_id", ""), {"row": row, "dense_rank": 0, "bm25_rank": 0}
                )
                entry["row"] = row

        for rank, (chunk_id, score) in enumerate(bm25_hits, start=1):
            entry = fused.get(chunk_id)
            if entry is None:
                continue
            entry["bm25_rank"] = rank
            entry["bm25_score"] = score

        scored: list[ScoredChunk] = []
        for chunk_id, entry in fused.items():
            rrf = 0.0
            if entry["dense_rank"]:
                rrf += 1.0 / (self.rrf_k + entry["dense_rank"])
            if entry["bm25_rank"]:
                rrf += 1.0 / (self.rrf_k + entry["bm25_rank"])
            chunk = ScoredChunk.from_row(entry["row"])
            chunk.score = rrf
            chunk.scores = {"rrf": rrf}
            if entry["dense_rank"]:
                chunk.scores["dense"] = float(entry["row"].get("similarity", 0.0))
            if entry["bm25_rank"]:
                chunk.scores["bm25"] = float(entry.get("bm25_score", 0.0))
            scored.append(chunk)

        # 分数相同时用 chunk_id 兜底排序，保证结果稳定可复现
        scored.sort(key=lambda c: (-c.score, c.chunk_id))
        top = scored[: self.fusion_top_n]

        if self.neighbor_expand > 0:
            top = self._expand_neighbors(top)

        logger.info(
            "召回完成：dense=%d bm25=%d 融合=%d 候选=%d",
            len(dense_rows),
            len(bm25_hits),
            len(scored),
            len(top),
        )
        return top

    # ------------------------------------------------------------------
    # 邻居扩展
    # ------------------------------------------------------------------

    def _expand_neighbors(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """把命中块同 doc_id 的 chunk_idx±1 也加进候选。

        解决「答案正好被切断在两块之间」的情况：命中的是后半块，
        上下文其实在前半块。扩展出来的邻居继承父块的 RRF 分数，
        分数相同的情况下由重排模型来决定谁更相关。
        """
        doc_ids = {c.doc_id for c in chunks if c.doc_id}
        if not doc_ids:
            return chunks

        # 把这些文档的全部块取回来，建 (doc_id, chunk_idx) → chunk 的索引
        siblings = self.vector_store.get_by_doc_ids(sorted(doc_ids))
        index: dict[tuple[str, int], dict] = {}
        for row in siblings:
            index[(row.get("doc_id", ""), int(row.get("chunk_idx", 0)))] = row

        merged = {c.chunk_id: c for c in chunks}
        for chunk in chunks:
            for offset in range(-self.neighbor_expand, self.neighbor_expand + 1):
                if offset == 0:
                    continue
                key = (chunk.doc_id, chunk.chunk_idx + offset)
                row = index.get(key)
                if row is None:
                    continue
                neighbor_id = row.get("chunk_id", "")
                if not neighbor_id or neighbor_id in merged:
                    continue
                neighbor = ScoredChunk.from_row(row)
                neighbor.score = chunk.score
                neighbor.scores = dict(chunk.scores)
                neighbor.scores["neighbor"] = 1.0
                merged[neighbor_id] = neighbor

        result = list(merged.values())
        # 分数相同时按 chunk_id 排序，保证同一查询每次结果一致
        result.sort(key=lambda c: (-c.score, c.chunk_id))
        return result
