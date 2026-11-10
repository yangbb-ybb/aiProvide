"""POST /api/search —— 原始召回结果，给检索调试面板用，不经过 LLM。

M1 实现。这里先留好路由契约。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..schemas import SearchRequest

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(req: SearchRequest) -> None:
    """M1 实现：返回召回块 + 各路分数（dense / bm25 / rrf / rerank），用于调参。"""
    raise HTTPException(status_code=501, detail="M1 阶段实现：检索调试")
