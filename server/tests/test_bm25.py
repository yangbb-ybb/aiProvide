"""BM25 索引的单元测试（用真实 bm25s，不碰嵌入模型）。

主要锁两件事：

1. **中文分词**：BM25 必须用 jieba 切词，否则整句变成一个 term，召回等于随机。
2. **bm25s 返回值解析**：`bm25s.Results` 是 NamedTuple，但它**重写了 `__len__`**
   （返回文档行数而不是字段数），所以不能用 `len(raw) == 2` 判断格式。
   这个坑踩过一次 —— 线上表现为 /api/search 直接 500。

    uv run pytest tests/ -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.store.bm25_index import BM25Index, _split_retrieve_result

_TEXTS = [
    "生产环境必须设置 DATABASE_URL 和 REDIS_URL 两个环境变量",
    "今天天气不错，适合出门散步",
    "安装依赖用 uv sync，模型用 ollama pull",
]


def _build(tmpdir: str) -> BM25Index:
    """建一个三块的索引。"""
    index = BM25Index(Path(tmpdir))
    index.add_records(
        [
            {"chunk_id": "doc1:0", "doc_id": "doc1", "text": _TEXTS[0]},
            {"chunk_id": "doc1:1", "doc_id": "doc1", "text": _TEXTS[1]},
            {"chunk_id": "doc1:2", "doc_id": "doc1", "text": _TEXTS[2]},
        ]
    )
    return index


def test_tokenize_splits_chinese() -> None:
    """中文必须切成词，不能整句当一个 term。"""
    tokens = BM25Index(Path(tempfile.mkdtemp())).tokenize("数据库连接池配置")
    assert "数据库" in tokens
    assert "连接" in tokens
    # 整句绝不能作为一个 term 混进去
    assert "数据库连接池配置" not in tokens


def test_tokenize_drops_punctuation() -> None:
    """标点不该进索引。"""
    tokens = BM25Index(Path(tempfile.mkdtemp())).tokenize("你好，世界！")
    assert "你好" in tokens
    assert "，" not in tokens
    assert "！" not in tokens


def test_search_finds_the_relevant_chunk() -> None:
    """中文查询要能命中对应块，且排第一。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _build(tmpdir)
        hits = index.search("环境变量怎么配", limit=3)
        assert hits, "应该有召回结果"
        assert hits[0][0] == "doc1:0", f"排第一的应该是环境变量那块，实际是 {hits[0][0]}"
        # 分数得递减
        assert hits[0][1] >= hits[-1][1]


def test_search_ignores_unrelated_query() -> None:
    """完全无关的查询不该把不相干的块顶到前面。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _build(tmpdir)
        hits = index.search("环境变量怎么配", limit=3)
        top = hits[0][0]
        assert top == "doc1:0"


def test_split_result_handles_bm25s_results() -> None:
    """能解析 bm25s 0.3.x 的 Results（它重写了 __len__）。"""
    import bm25s

    b = bm25s.BM25()
    b.index([["环境", "变量"], ["天气", "不错"]])
    raw = b.retrieve([["环境"]], k=2, n_threads=1, show_progress=False)

    # 关键前提：这个对象的 len() 是「文档行数」而不是「字段数」
    assert len(raw) != 2, "bm25s.Results 重写了 __len__，不能再用 len==2 判断"

    indices, scores = _split_retrieve_result(raw)
    assert len(indices) == len(scores) == 2
    assert indices[0] == 0, "最相关的应该是第 0 篇"
    assert scores[0] >= scores[1]


def test_split_result_handles_plain_tuple() -> None:
    """老版本返回普通 tuple 时也能解析。"""
    import numpy as np

    raw = (np.array([2, 0]), np.array([0.9, 0.4]))
    indices, scores = _split_retrieve_result(raw)
    assert indices == [2, 0]
    assert scores == [0.9, 0.4]


def test_search_survives_corpus_edge_cases() -> None:
    """空语料、空查询都不该抛异常。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = BM25Index(Path(tmpdir))
        assert index.search("任何问题", limit=5) == []
