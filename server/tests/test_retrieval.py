"""召回层的单元测试（用假组件，不碰真模型）。

RRF 融合是检索链路的核心，值得单独锁住它的数学：

    score(d) = 1/(K + rank_dense) + 1/(K + rank_bm25)

    uv run pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.retriever import Retriever, ScoredChunk


def _row(chunk_id: str, doc_id: str = "doc1", chunk_idx: int = 0, similarity: float = 0.9) -> dict:
    """伪造一条向量库返回的行。"""
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "content": f"块 {chunk_id} 的正文",
        "source_path": f"/tmp/{doc_id}.md",
        "source_type": "md",
        "title": doc_id,
        "heading_path": ["标题"],
        "chunk_idx": chunk_idx,
        "char_len": 10,
        "page": None,
        "similarity": similarity,
    }


class FakeEmbedder:
    def embed_query(self, text: str):
        return [0.0, 1.0]


class FakeVectorStore:
    def __init__(self, dense_rows: list[dict], all_rows: list[dict] | None = None):
        self.dense_rows = dense_rows
        self.all_rows = all_rows if all_rows is not None else dense_rows

    def search(self, vector, limit: int = 50, doc_id: str | None = None):
        return self.dense_rows[:limit]

    def get_chunks(self, chunk_ids: list[str]):
        wanted = set(chunk_ids)
        return [r for r in self.all_rows if r["chunk_id"] in wanted]

    def get_by_doc_ids(self, doc_ids: list[str]):
        wanted = set(doc_ids)
        return [r for r in self.all_rows if r["doc_id"] in wanted]


class FakeBM25:
    def __init__(self, hits: list[tuple[str, float]]):
        self.hits = hits

    def search(self, query: str, limit: int = 50):
        return self.hits[:limit]


def _retriever(dense, bm25_hits, all_rows=None, **kwargs):
    return Retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(dense, all_rows),
        bm25_index=FakeBM25(bm25_hits),
        reranker=None,
        **kwargs,
    )


def test_rrf_formula():
    """RRF 分数必须严格等于 1/(K+rank_dense) + 1/(K+rank_bm25)。"""
    dense = [_row("a"), _row("b")]
    bm25 = [("b", 5.0), ("c", 3.0)]
    r = _retriever(dense, bm25, all_rows=[*dense, _row("c")], rrf_k=60, fusion_top_n=10)

    got = {c.chunk_id: c.score for c in r.retrieve_raw("查询", 10)}

    assert abs(got["a"] - 1 / 61) < 1e-9
    assert abs(got["b"] - (1 / 62 + 1 / 61)) < 1e-9
    assert abs(got["c"] - 1 / 62) < 1e-9


def test_single_path_contributes_zero():
    """只在一路里出现的块，另一路贡献 0。"""
    dense = [_row("a")]
    bm25 = [("z", 1.0)]
    r = _retriever(dense, bm25, all_rows=[*dense, _row("z")], rrf_k=60, fusion_top_n=10)
    got = {c.chunk_id: c.score for c in r.retrieve_raw("查询", 10)}
    assert abs(got["z"] - 1 / 61) < 1e-9
    assert abs(got["a"] - 1 / 61) < 1e-9


def test_agreement_ranks_higher():
    """两路都命中的块应该排在只命中一路的前面。"""
    dense = [_row("only_dense"), _row("both")]
    bm25 = [("only_bm25", 1.0), ("both", 2.0)]
    r = _retriever(
        dense,
        bm25,
        all_rows=[*dense, _row("only_bm25")],
        rrf_k=60,
        fusion_top_n=10,
    )
    order = [c.chunk_id for c in r.retrieve_raw("查询", 10)]
    assert order[0] == "both"
    assert set(order) == {"both", "only_dense", "only_bm25"}


def test_bm25_only_chunk_metadata_is_filled():
    """只在关键词路出现的块，元数据也要从向量库捞出来。"""
    dense = [_row("a")]
    bm25 = [("b", 1.0)]
    r = _retriever(dense, bm25, all_rows=[*dense, _row("b")], fusion_top_n=10)
    chunks = {c.chunk_id: c for c in r.retrieve_raw("查询", 10)}
    assert "b" in chunks
    assert chunks["b"].content == "块 b 的正文"
    assert chunks["b"].source_path == "/tmp/doc1.md"


def test_neighbor_expansion():
    """命中块的 ±1 邻居要进候选，且带上 neighbor 标记。"""
    dense = [_row("doc1:1", chunk_idx=1)]
    r = _retriever(
        dense,
        [],
        all_rows=[
            _row("doc1:0", chunk_idx=0),
            _row("doc1:1", chunk_idx=1),
            _row("doc1:2", chunk_idx=2),
        ],
        fusion_top_n=10,
        neighbor_expand=1,
    )
    chunks = {c.chunk_id: c for c in r.retrieve_raw("查询", 10)}
    assert set(chunks) == {"doc1:0", "doc1:1", "doc1:2"}
    assert chunks["doc1:0"].scores.get("neighbor") == 1.0
    assert chunks["doc1:1"].scores.get("neighbor") is None


def test_neighbor_expansion_disabled():
    """neighbor_expand=0 时不做扩展。"""
    dense = [_row("doc1:1", chunk_idx=1)]
    r = _retriever(
        dense,
        [],
        all_rows=[_row("doc1:0", chunk_idx=0), _row("doc1:1", chunk_idx=1)],
        fusion_top_n=10,
        neighbor_expand=0,
    )
    chunks = {c.chunk_id: c for c in r.retrieve_raw("查询", 10)}
    assert set(chunks) == {"doc1:1"}


def test_top_n_limits_results():
    """融合后只保留 fusion_top_n 条。"""
    dense = [_row(f"c{i}") for i in range(10)]
    r = _retriever(dense, [], all_rows=dense, fusion_top_n=3, neighbor_expand=0)
    assert len(r.retrieve_raw("查询", 10)) == 3


def test_scores_are_exposed_per_route():
    """各路分数要分别保留下来给调试面板用。"""
    dense = [_row("a", similarity=0.83)]
    bm25 = [("a", 12.5)]
    r = _retriever(dense, bm25, all_rows=dense, fusion_top_n=10, neighbor_expand=0)
    chunk = r.retrieve_raw("查询", 10)[0]
    assert abs(chunk.scores["dense"] - 0.83) < 1e-6
    assert abs(chunk.scores["bm25"] - 12.5) < 1e-6
    assert "rrf" in chunk.scores


def test_empty_knowledge_base():
    """空库不能炸，返回空列表。"""
    r = _retriever([], [], all_rows=[])
    assert r.retrieve_raw("查询", 10) == []
    assert r.retrieve("查询", 5) == []


def test_heading_path_json_parsing():
    """向量库里 heading_path 存的是 JSON 字符串，取出来要还原成 list。"""
    row = _row("a")
    row["heading_path"] = '["第3章", "3.2 环境变量"]'
    chunk = ScoredChunk.from_row(row)
    assert chunk.heading_path == ["第3章", "3.2 环境变量"]
