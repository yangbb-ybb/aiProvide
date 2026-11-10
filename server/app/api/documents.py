"""文档管理接口（上传 / URL 入库 / 列表 / 删除）。

M1 实现。这里先留好路由契约，前端可以照着开发。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import UrlIngestRequest

router = APIRouter(tags=["documents"])


@router.post("/documents")
async def upload_document() -> None:
    """M1 实现：接收上传文件 → 解析 → 分块 → 入库。"""
    raise HTTPException(status_code=501, detail="M1 阶段实现：文档上传与入库")


@router.post("/documents/url")
async def ingest_url(req: UrlIngestRequest) -> None:
    """M1 实现：抓取网页正文 → 分块 → 入库。"""
    raise HTTPException(status_code=501, detail="M1 阶段实现：URL 入库")


@router.get("/documents")
async def list_documents() -> None:
    """M1 实现：返回文档列表（含 chunk 数、mtime）。"""
    raise HTTPException(status_code=501, detail="M1 阶段实现：文档列表")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> None:
    """M1 实现：删除文档及其所有块 + BM25 条目。"""
    raise HTTPException(status_code=501, detail="M1 阶段实现：删除文档")
