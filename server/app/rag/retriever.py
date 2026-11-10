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

from dataclasses import dataclass, field


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


class Retriever:
    """混合召回器。"""

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """返回精排后的 top_k 个块。"""
        raise NotImplementedError("M1 阶段实现：混合召回")

    def retrieve_raw(self, query: str, top_n: int = 50) -> list[ScoredChunk]:
        """只做召回和 RRF 融合，不重排。给调试面板和评测用。"""
        raise NotImplementedError("M1 阶段实现：原始召回")
