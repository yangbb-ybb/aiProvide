"""LanceDB 向量仓库。

本项目只拿它当**向量仓库 + 元数据过滤**，不用它的 hybrid search
（内置 FTS 是英文分词器，中文会崩，融合算法我们自己写 —— 见 rag/retriever.py）。

表名 chunks，数据落 data/lancedb/chunks.lance，schema 见规划文档 §3.3。
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    """LanceDB 封装。"""

    def __init__(self, path, dim: int = 1024) -> None:
        self.path = path
        self.dim = dim
        self._db = None
        self._table = None

    def _ensure_table(self):
        """懒连接 + 必要时建表建索引。"""
        raise NotImplementedError("M1 阶段实现：LanceDB 连接与建表")

    def upsert(self, rows: list[dict]) -> int:
        """写入/覆盖若干行，返回写入条数。按 chunk_id 去重。"""
        raise NotImplementedError

    def search(self, vector: np.ndarray, limit: int = 50, doc_id: str | None = None) -> list[dict]:
        """向量检索。doc_id 不为空时按文档过滤。"""
        raise NotImplementedError

    def delete_doc(self, doc_id: str) -> int:
        """删掉某个文档的所有块，返回删除条数。"""
        raise NotImplementedError

    def list_docs(self) -> list[dict]:
        """按 doc_id 聚合，返回文档列表。"""
        raise NotImplementedError
