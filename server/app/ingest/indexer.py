"""入库编排：解析 → 分块 → 向量化 → 写入向量库 + 关键词索引。"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from ..config import get_settings
from .chunker import Chunk, chunk_markdown
from .loader import ParsedDoc, load

logger = logging.getLogger(__name__)


def make_doc_id(source_path: str, mtime: float) -> str:
    """doc_id = sha1(来源路径 + mtime)[:16]。

    带上 mtime 是为了让「同一个文件改了内容」变成一个新文档：
    mtime 变了 → doc_id 变了 → 旧版本的块可以被清掉，不会出现两份。
    """
    return hashlib.sha1(f"{source_path}:{mtime}".encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    """内容哈希，用于跨文档去重和幂等更新判断。"""
    return hashlib.sha1(text.encode()).hexdigest()


def index_document(
    source: str,
    source_type: str,
    *,
    embedder=None,
    vector_store=None,
    bm25_index=None,
) -> dict:
    """把一个文档入库，返回 {doc_id, chunks, needs_ocr}。

    步骤：
    1. loader.load 解析成 Markdown
    2. chunker.chunk_markdown 切块
    3. embedder.embed_documents 向量化
    4. store.vector_store 写 LanceDB
    5. store.bm25_index 写关键词索引（jieba 分词）
    """
    # 依赖默认从 deps 取单例（模型加载一次要好几秒，不能每次请求都重来）
    if embedder is None or vector_store is None or bm25_index is None:
        from ..deps import get_bm25_index, get_embedder, get_vector_store

        embedder = embedder or get_embedder()
        vector_store = vector_store or get_vector_store()
        bm25_index = bm25_index or get_bm25_index()

    settings = get_settings()

    # 本地文件用绝对路径，保证换个工作目录启动，doc_id 也不变
    if source_type == "url":
        source_path = source
        mtime = 0.0
    else:
        source_path = str(Path(source).expanduser().resolve())
        mtime = os.path.getmtime(source_path)

    parsed: ParsedDoc = load(source_path, source_type)
    chunks: list[Chunk] = chunk_markdown(
        parsed.markdown,
        {},
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    if not chunks:
        raise ValueError(f"这份文档切不出任何内容块（正文为空？）：{source_path}")

    doc_id = make_doc_id(source_path, mtime)

    # 先完成向量化；失败时保留旧索引，避免一次异常导致数据丢失。
    vectors = embedder.embed_documents([c.content for c in chunks])

    # 同一份文档重新导入时，旧版本仅在新数据准备好后清掉。
    vector_store.delete_by_source_path(source_path)
    bm25_index.delete_by_source_path(source_path)

    rows: list[dict] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk_id = f"{doc_id}:{chunk.chunk_idx}"
        chunk.chunk_id = chunk_id
        chunk.doc_id = doc_id
        chunk.source_path = source_path
        chunk.source_type = source_type
        chunk.title = parsed.title
        chunk.content_hash = content_hash(chunk.content)
        chunk.mtime = mtime
        chunk.url = parsed.url
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "vector": vector,
                "content": chunk.content,
                "source_path": source_path,
                "source_type": source_type,
                "title": parsed.title,
                "heading_path": chunk.heading_path,
                "chunk_idx": chunk.chunk_idx,
                "char_len": chunk.char_len,
                "content_hash": chunk.content_hash,
                "mtime": mtime,
                "url": parsed.url,
                "page": chunk.page,
            }
        )

    vector_store.upsert(rows)
    bm25_index.add_records(
        [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "source_path": r["source_path"],
                "text": r["content"],
            }
            for r in rows
        ]
    )
    bm25_index.save()

    logger.info(
        "入库完成：%s → doc_id=%s，%d 个块%s",
        source_path,
        doc_id,
        len(rows),
        "（PDF 部分页面可能需要 OCR）" if parsed.needs_ocr else "",
    )
    return {
        "doc_id": doc_id,
        "chunks": len(rows),
        "needs_ocr": parsed.needs_ocr,
        "title": parsed.title,
        "source_path": source_path,
        "source_type": source_type,
        "meta": parsed.meta,
    }
