"""BM25 关键词索引（bm25s + jieba）。

**中文必须分词**：BM25 原本按空格切词，中文不分词的话整句会被当成一个 term，
召回基本等于随机。这是本项目最容易踩的坑。

索引持久化到 data/bm25/，入库时增量更新。

实现说明：bm25s 本身不支持「增量加文档」和「按文档删除」，
所以这里把语料（chunk_id / doc_id / 正文）单独存成 corpus.jsonl，
bm25s 的索引文件作为**可重建的派生数据**保存：
语料变了就重建索引，重建耗时在几千个块的量级是 1~3 秒，可以接受。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import jieba

logger = logging.getLogger(__name__)

# jieba 第一次用会打印词典构建日志，太吵，压掉
jieba.setLogLevel(logging.WARNING)

_CORPUS_FILE = "corpus.jsonl"
_INDEX_DIR = "bm25s"

# 空文档的分词占位符：bm25s 对空 token 列表处理不好，用一个不可能出现在查询里的哨兵顶上
_EMPTY_TOKEN = "\x00empty"

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)


class BM25Index:
    """BM25 关键词索引。"""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._index = None
        self._chunk_ids: list[str] = []
        self._docs: list[dict] = []
        """每条是 {"chunk_id": str, "doc_id": str, "text": str}。"""
        self._dirty = True

    # ------------------------------------------------------------------
    # 分词
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """jieba 精确模式切词。

        用 cut_for_search 而不是 cut：搜索引擎模式下同时给出长词和它的分词
        （比如「数据库连接」和「数据库」「连接」），召回率更高。
        英文保留小写，数字保留。
        """
        tokens: list[str] = []
        for word in jieba.cut_for_search(text):
            word = word.strip().lower()
            if not word or _PUNCT_RE.match(word):
                continue
            tokens.append(word)
        return tokens

    # ------------------------------------------------------------------
    # 语料维护
    # ------------------------------------------------------------------

    def add(self, chunk_ids: list[str], texts: list[str], doc_ids: list[str] | None = None) -> None:
        """增量加入若干块。"""
        if not chunk_ids:
            return
        if doc_ids is None:
            doc_ids = [""] * len(chunk_ids)
        self.add_records(
            [
                {"chunk_id": cid, "doc_id": did, "text": text}
                for cid, text, did in zip(chunk_ids, texts, doc_ids, strict=True)
            ]
        )

    def add_records(self, records: list[dict]) -> None:
        """批量加入若干块。

        每条记录需要 chunk_id / doc_id / text，source_path 可选
        （source_path 用来在同一个文件被重新导入时清掉旧版本）。
        """
        if not records:
            return
        for record in records:
            self._docs.append(
                {
                    "chunk_id": record["chunk_id"],
                    "doc_id": record.get("doc_id", ""),
                    "source_path": record.get("source_path", ""),
                    "text": record.get("text", ""),
                }
            )
        self._chunk_ids = [d["chunk_id"] for d in self._docs]
        self._dirty = True

    def delete_doc(self, doc_id: str) -> int:
        """删掉某个文档的所有条目，返回删除条数。"""
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["doc_id"] != doc_id]
        removed = before - len(self._docs)
        if removed:
            self._chunk_ids = [d["chunk_id"] for d in self._docs]
            self._dirty = True
        return removed

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """删掉指定的若干块（内容哈希没变、块被重新切分时用）。"""
        targets = set(chunk_ids)
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["chunk_id"] not in targets]
        removed = before - len(self._docs)
        if removed:
            self._chunk_ids = [d["chunk_id"] for d in self._docs]
            self._dirty = True
        return removed

    def delete_by_source_path(self, source_path: str) -> int:
        """删掉某个来源路径下的所有条目（同一个文件重新导入时清旧版本）。"""
        before = len(self._docs)
        self._docs = [d for d in self._docs if d.get("source_path") != source_path]
        removed = before - len(self._docs)
        if removed:
            self._chunk_ids = [d["chunk_id"] for d in self._docs]
            self._dirty = True
        return removed

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[tuple[str, float]]:
        """返回 [(chunk_id, score)]，按分数降序。"""
        if not self._docs:
            return []
        self._ensure_index()

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        k = min(limit, len(self._docs))
        try:
            # bm25s 的 query_tokens 必须是「每个查询一个 token 列表」，
            # 也就是外面还要再包一层；k 超过语料条数它会直接抛错，所以上面先夹住。
            raw = self._index.retrieve(
                [query_tokens], k=k, n_threads=1, show_progress=False
            )
            indices, scores = _split_retrieve_result(raw)
        except Exception as exc:  # noqa: BLE001 - bm25s 版本差异兜底
            # 关键词路挂掉不该让整个请求 500 —— 退化成只有向量路，总比没有结果强
            logger.warning("BM25 检索失败（%s），本轮关键词路返回空", exc)
            return []

        results: list[tuple[str, float]] = []
        for idx, score in zip(indices, scores, strict=True):
            if idx < 0 or idx >= len(self._docs):
                continue
            results.append((self._docs[idx]["chunk_id"], float(score)))
        return results

    def _ensure_index(self) -> None:
        """确保 bm25s 索引是最新的（语料变了就重建）。"""
        if self._index is not None and not self._dirty:
            return
        import bm25s

        corpus_tokens: list[list[str]] = []
        for doc in self._docs:
            tokens = self.tokenize(doc["text"])
            corpus_tokens.append(tokens if tokens else [_EMPTY_TOKEN])

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        self._index = retriever
        self._dirty = False

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------

    def save(self) -> None:
        """在同一文件系统构建临时目录，完成后原子替换，避免半写状态。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mkdtemp(prefix=f".{self.path.name}-", dir=self.path.parent))
        backup_path = self.path.with_name(f".{self.path.name}.old")
        try:
            with (temp_path / _CORPUS_FILE).open("w", encoding="utf-8") as fh:
                for doc in self._docs:
                    fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                fh.flush()

            if self._docs:
                self._ensure_index()
                self._index.save(str(temp_path / _INDEX_DIR), show_progress=False)

            if backup_path.exists():
                shutil.rmtree(backup_path)
            if self.path.exists():
                self.path.replace(backup_path)
            temp_path.replace(self.path)
            if backup_path.exists():
                shutil.rmtree(backup_path)
        except Exception:
            if not self.path.exists() and backup_path.exists():
                backup_path.replace(self.path)
            raise
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path)
        logger.info("BM25 索引已保存：%d 个块 → %s", len(self._docs), self.path)

    def load(self) -> None:
        """从磁盘加载。"""
        corpus_path = self.path / _CORPUS_FILE
        if not corpus_path.exists():
            self._docs = []
            self._chunk_ids = []
            self._dirty = True
            return

        docs: list[dict] = []
        with corpus_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("BM25 语料里有坏行，已跳过：%s", line[:80])
        self._docs = docs
        self._chunk_ids = [d["chunk_id"] for d in docs]

        index_dir = self.path / _INDEX_DIR
        if docs and index_dir.exists():
            import bm25s

            try:
                self._index = bm25s.BM25.load(str(index_dir), show_progress=False)
                self._dirty = False
            except Exception as exc:  # noqa: BLE001 - 索引坏了就重建
                logger.warning("BM25 索引加载失败（%s），将重建", exc)
                self._index = None
                self._dirty = True
        else:
            self._index = None
            self._dirty = True
        logger.info("BM25 索引已加载：%d 个块", len(self._docs))


def _split_retrieve_result(raw) -> tuple[list[int], list[float]]:
    """把 bm25s.retrieve 的返回值拆成 (索引列表, 分数列表)。

    bm25s 0.3.x 返回的是 `Results(documents, scores)` 这个 NamedTuple，
    但它**重写了 `__len__`**（返回文档行数，而不是字段数），
    所以不能用 `len(raw) == 2` 来判断格式 —— 这里必须按字段名取。

    documents / scores 的形状是 (查询数, k)；本项目只查一个查询，取第 0 行。

    老版本 bm25s 返回普通 tuple（(indices, scores)），走兜底分支；
    那个分支再用 dtype 判别，因为个别版本的顺序是反的。
    """
    import numpy as np

    documents = getattr(raw, "documents", None)
    scores = getattr(raw, "scores", None)

    if documents is None or scores is None:
        if not isinstance(raw, (tuple, list)) or len(raw) < 2:
            raise ValueError(f"bm25s.retrieve 返回值格式异常：{type(raw)}")
        first = np.asarray(raw[0]).ravel()
        second = np.asarray(raw[1]).ravel()
        if np.issubdtype(first.dtype, np.integer):
            documents, scores = first, second
        else:
            documents, scores = second, first

    indices = np.asarray(documents)
    values = np.asarray(scores)
    if indices.ndim == 2:
        indices = indices[0]
    if values.ndim == 2:
        values = values[0]
    return indices.ravel().tolist(), values.ravel().tolist()
