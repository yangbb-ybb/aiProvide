"""文档管理接口（上传 / URL 入库 / 列表 / 删除）。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..config import get_settings
from ..deps import get_bm25_index, get_vector_store
from ..ingest.indexer import index_document
from ..schemas import DocumentSummary, IngestResponse, UrlIngestRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

# 扩展名 → source_type
_EXT_TYPES = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
}


@router.post("/documents", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    vector_store=Depends(get_vector_store),
    bm25_index=Depends(get_bm25_index),
) -> IngestResponse:
    """接收上传文件 → 解析 → 分块 → 入库。"""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    source_type = _EXT_TYPES.get(suffix)
    if source_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix or filename!r}，"
            f"目前支持：{'、'.join(sorted(_EXT_TYPES))}",
        )

    settings = get_settings()
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    # 用 doc_id 之外的可读名保存：时间戳 + 原名，避免同名覆盖
    safe_name = Path(filename).name
    target = settings.uploads_path / f"{_timestamp()}_{safe_name}"
    try:
        total = 0
        with target.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.upload_max_bytes:
                    raise HTTPException(status_code=413, detail="上传文件超过大小限制")
                fh.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        result = await asyncio.to_thread(index_document, str(target), source_type)
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档入库失败：%s", filename)
        raise HTTPException(status_code=500, detail="文档入库失败，请查看服务日志") from exc

    return IngestResponse(
        doc_id=result["doc_id"],
        source_path=result["source_path"],
        source_type=result["source_type"],
        chunks=result["chunks"],
        needs_ocr=result["needs_ocr"],
    )


@router.post("/documents/url", response_model=IngestResponse)
async def ingest_url(
    req: UrlIngestRequest,
    vector_store=Depends(get_vector_store),
    bm25_index=Depends(get_bm25_index),
) -> IngestResponse:
    """抓取网页正文 → 分块 → 入库。"""
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL 必须以 http:// 或 https:// 开头")

    try:
        result = await asyncio.to_thread(index_document, url, "url")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("URL 入库失败：%s", url)
        raise HTTPException(status_code=500, detail="URL 入库失败，请查看服务日志") from exc

    return IngestResponse(
        doc_id=result["doc_id"],
        source_path=result["source_path"],
        source_type=result["source_type"],
        chunks=result["chunks"],
        needs_ocr=result["needs_ocr"],
    )


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(vector_store=Depends(get_vector_store)) -> list[DocumentSummary]:
    """返回文档列表（含 chunk 数、mtime）。"""
    docs = await asyncio.to_thread(vector_store.list_docs)
    docs.sort(key=lambda d: d["mtime"], reverse=True)
    return [DocumentSummary(**d) for d in docs]


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    vector_store=Depends(get_vector_store),
    bm25_index=Depends(get_bm25_index),
) -> dict:
    """删除文档及其所有块 + BM25 条目。"""
    deleted = await asyncio.to_thread(vector_store.delete_doc, doc_id)
    await asyncio.to_thread(bm25_index.delete_doc, doc_id)
    await asyncio.to_thread(bm25_index.save)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"没有这个文档：{doc_id}")
    return {"doc_id": doc_id, "deleted": deleted}


def _timestamp() -> str:
    """文件名用的时间戳。"""
    import time

    return time.strftime("%Y%m%d_%H%M%S")
