"""入库编排：解析 → 分块 → 向量化 → 写入向量库 + 关键词索引。"""

from __future__ import annotations

from ..ingest.chunker import Chunk


def index_document(source: str, source_type: str) -> dict:
    """把一个文档入库，返回 {doc_id, chunks, needs_ocr}。

    步骤：
    1. loader.load 解析成 Markdown
    2. chunker.chunk_markdown 切块
    3. embedder.embed_documents 向量化
    4. store.vector_store 写 LanceDB
    5. store.bm25_index 写关键词索引（jieba 分词）
    """
    raise NotImplementedError("M1 阶段实现：文档入库")
