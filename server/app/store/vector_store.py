"""LanceDB 向量仓库。

本项目只拿它当**向量仓库 + 元数据过滤**，不用它的 hybrid search
（内置 FTS 是英文分词器，中文会崩，融合算法我们自己写 —— 见 rag/retriever.py）。

表名 chunks，数据落 data/lancedb/chunks.lance，schema 见规划文档 §3.3。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

TABLE_NAME = "chunks"


def _escape(value: str) -> str:
    """转义 SQL 字符串字面量里的单引号。"""
    return str(value).replace("'", "''")

# §3.3 的字段表。注意 heading_path 存的是 JSON 字符串而不是 list：
# LanceDB 的 schema 里嵌 list 会带来额外的类型负担，JSON 字符串更省事也更好查。
SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 1024)),
        pa.field("content", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("source_type", pa.string()),
        pa.field("title", pa.string()),
        pa.field("heading_path", pa.string()),
        pa.field("chunk_idx", pa.int32()),
        pa.field("char_len", pa.int32()),
        pa.field("content_hash", pa.string()),
        pa.field("mtime", pa.float64()),
        pa.field("url", pa.string()),
        pa.field("page", pa.int32()),
    ]
)


class VectorStore:
    """LanceDB 封装。"""

    def __init__(self, path, dim: int = 1024) -> None:
        self.path = Path(path)
        self.dim = dim
        self._db = None
        self._table = None

    # ------------------------------------------------------------------
    # 连接与建表
    # ------------------------------------------------------------------

    def _ensure_table(self):
        """懒连接 + 必要时建表建索引。"""
        if self._table is not None:
            return self._table

        import lancedb

        self.path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.path))

        if TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(TABLE_NAME)
        else:
            self._table = self._db.create_table(TABLE_NAME, schema=self._schema())
            self._create_index(self._table)
        return self._table

    def _schema(self):
        """按当前 dim 生成 schema（dim 来自配置，模型换了要跟着变）。"""
        if self.dim == SCHEMA.field("vector").type.list_size:
            return SCHEMA
        fields = [
            pa.field("vector", pa.list_(pa.float32(), self.dim)) if f.name == "vector" else f
            for f in SCHEMA
        ]
        return pa.schema(fields)

    @staticmethod
    def _create_index(table) -> None:
        """建 HNSW 索引。

        lancedb 新版本推荐用 IndexConfig，旧版本用 index_type 字符串；
        两套 API 都留着，哪个能用走哪个。
        """
        try:
            from lancedb.index import IndexConfig, IndexType

            table.create_index(
                "vector",
                config=IndexConfig(index_type=IndexType.HNSW, metric="cosine"),
            )
            logger.info("已建 HNSW 向量索引（cosine）")
            return
        except Exception as exc:  # noqa: BLE001 - 老版本没有 IndexConfig
            logger.debug("IndexConfig 不可用（%s），退回旧版 index_type 写法", exc)

        try:
            table.create_index("vector", index_type="hnsw", metric="cosine")
            logger.info("已建 HNSW 向量索引（cosine）")
        except Exception as exc:  # noqa: BLE001 - 建索引失败不影响正确性，只是变慢
            logger.warning("建向量索引失败（%s），检索会退化成全表扫描", exc)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def upsert(self, rows: list[dict]) -> int:
        """写入/覆盖若干行，返回写入条数。按 chunk_id 去重。"""
        if not rows:
            return 0
        table = self._ensure_table()
        payload = [self._to_row(r) for r in rows]

        try:
            table.merge_insert("chunk_id").when_matched_update_all().when_not_matched_insert_all().execute(
                payload
            )
        except Exception as exc:  # noqa: BLE001 - 老版本没有 merge_insert，退回先删后插
            logger.debug("merge_insert 不可用（%s），退回先删后插", exc)
            ids = [r["chunk_id"] for r in payload]
            quoted = ",".join("'" + i.replace("'", "''") + "'" for i in ids)
            table.delete(f"chunk_id IN ({quoted})")
            table.add(payload)
        return len(payload)

    @staticmethod
    def _to_row(row: dict) -> dict:
        """把内部 dict 转成符合 schema 的一行。"""
        vector = row.get("vector")
        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32).tolist()
        heading_path = row.get("heading_path") or []
        if not isinstance(heading_path, str):
            heading_path = json.dumps(heading_path, ensure_ascii=False)
        return {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "vector": vector,
            "content": row.get("content", ""),
            "source_path": row.get("source_path", ""),
            "source_type": row.get("source_type", ""),
            "title": row.get("title", ""),
            "heading_path": heading_path,
            "chunk_idx": int(row.get("chunk_idx", 0)),
            "char_len": int(row.get("char_len", 0)),
            "content_hash": row.get("content_hash", ""),
            "mtime": float(row.get("mtime", 0.0)),
            "url": row.get("url") or "",
            "page": int(row["page"]) if row.get("page") is not None else None,
        }

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, vector: np.ndarray, limit: int = 50, doc_id: str | None = None) -> list[dict]:
        """向量检索。doc_id 不为空时按文档过滤。

        返回的每行带一个 similarity 字段（余弦相似度）。

        相似度直接用点积算，不靠 1 - _distance 反推：
        向量入库前已经 L2 归一化，两边都是单位向量，点积就是余弦相似度。
        这样即使索引退化成默认的 L2 度量，排序依然正确，分数也不会算错。
        """
        table = self._ensure_table()
        if table.count_rows() == 0:
            return []

        query = np.asarray(vector, dtype=np.float32)
        builder = table.search(query.tolist())
        if doc_id:
            builder = builder.where(f"doc_id = '{_escape(doc_id)}'")
        rows = builder.limit(limit).to_list()

        for row in rows:
            stored = row.pop("vector", None)
            distance = row.pop("_distance", None)
            if stored is not None:
                similarity = float(np.dot(query, np.asarray(stored, dtype=np.float32)))
            elif distance is not None:
                # 拿不到向量时的兜底：按 cosine 假设反推
                similarity = 1.0 - float(distance)
            else:
                similarity = 0.0
            row["similarity"] = similarity
        return rows

    # ------------------------------------------------------------------
    # 按 id / doc 取块（给召回的邻居扩展用）
    # ------------------------------------------------------------------

    def get_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """按 chunk_id 取块（不含向量）。"""
        return self._filtered(
            "chunk_id",
            chunk_ids,
            [f.name for f in SCHEMA if f.name != "vector"],
        )

    def get_by_doc_ids(self, doc_ids: list[str]) -> list[dict]:
        """按 doc_id 取该文档的所有块（不含向量）。"""
        return self._filtered(
            "doc_id",
            doc_ids,
            [f.name for f in SCHEMA if f.name != "vector"],
        )

    def _filtered(self, column: str, values: list[str], columns: list[str]) -> list[dict]:
        """按某列的 IN 过滤取行。

        优先走 LanceDB 的 SQL 过滤（不下推全表），
        版本不支持时退回全表扫 + Python 侧过滤。
        """
        table = self._ensure_table()
        if not values:
            return []
        quoted = ",".join("'" + str(v).replace("'", "''") + "'" for v in values)
        try:
            rows = (
                table.search()
                .where(f"{column} IN ({quoted})")
                .limit(len(values) * 64)
                .select(columns)
                .to_list()
            )
            return rows
        except Exception as exc:  # noqa: BLE001 - 老版本没有无向量的 search()
            logger.debug("SQL 过滤不可用（%s），退回全表扫描", exc)
            targets = set(values)
            return [r for r in self._iter_rows(columns) if r.get(column) in targets]

    # ------------------------------------------------------------------
    # 删除与列表
    # ------------------------------------------------------------------

    def delete_doc(self, doc_id: str) -> int:
        """删掉某个文档的所有块，返回删除条数。"""
        return self._delete_where(f"doc_id = '{_escape(doc_id)}'")

    def delete_by_source_path(self, source_path: str) -> int:
        """删掉某个来源路径下的所有块（同一文件重新导入时清旧版本）。"""
        return self._delete_where(f"source_path = '{_escape(source_path)}'")

    def _delete_where(self, predicate: str) -> int:
        table = self._ensure_table()
        before = table.count_rows()
        if before == 0:
            return 0
        table.delete(predicate)
        return before - table.count_rows()

    def count(self) -> int:
        """表里一共多少个块。"""
        return self._ensure_table().count_rows()

    def list_docs(self) -> list[dict]:
        """按 doc_id 聚合，返回文档列表。"""
        table = self._ensure_table()
        if table.count_rows() == 0:
            return []

        docs: dict[str, dict] = {}
        for row in self._iter_rows():
            doc_id = row["doc_id"]
            entry = docs.get(doc_id)
            if entry is None:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "source_path": row["source_path"],
                    "source_type": row["source_type"],
                    "title": row["title"],
                    "chunks": 1,
                    "mtime": float(row["mtime"]),
                }
            else:
                entry["chunks"] += 1
                entry["mtime"] = max(entry["mtime"], float(row["mtime"]))
        return list(docs.values())

    def _iter_rows(self, columns: list[str] | None = None):
        """遍历表里的所有行。默认不带向量，省内存。"""
        table = self._ensure_table()
        if columns is None:
            columns = ["doc_id", "source_path", "source_type", "title", "mtime"]
        try:
            arrow_table = table.to_arrow(columns=columns)
        except TypeError:
            # 老版本 to_arrow 不接受 columns 参数
            arrow_table = table.to_arrow().select(columns)
        return arrow_table.to_pylist()
