"""BM25 关键词索引（bm25s + jieba）。

**中文必须分词**：BM25 原本按空格切词，中文不分词的话整句会被当成一个 term，
召回基本等于随机。这是本项目最容易踩的坑。

索引持久化到 data/bm25/，入库时增量更新。
"""

from __future__ import annotations

import jieba


class BM25Index:
    """BM25 关键词索引。"""

    def __init__(self, path) -> None:
        self.path = path
        self._index = None
        self._chunk_ids: list[str] = []

    def tokenize(self, text: str) -> list[str]:
        """jieba 精确模式切词。"""
        return list(jieba.cut_for_search(text))

    def add(self, chunk_ids: list[str], texts: list[str]) -> None:
        """增量加入若干块。"""
        raise NotImplementedError("M1 阶段实现：BM25 索引写入")

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序。"""
        raise NotImplementedError

    def delete_doc(self, doc_id: str) -> int:
        """删掉某个文档的所有条目，返回删除条数。"""
        raise NotImplementedError

    def save(self) -> None:
        """落盘。"""
        raise NotImplementedError

    def load(self) -> None:
        """从磁盘加载。"""
        raise NotImplementedError
