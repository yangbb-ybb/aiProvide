"""POST /api/search —— 原始召回结果，给检索调试面板用，不经过 LLM。"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends

from ..config import get_settings
from ..deps import get_retriever
from ..schemas import SearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(req: SearchRequest, retriever=Depends(get_retriever)) -> dict:
    """返回召回块 + 各路分数（dense / bm25 / rrf / rerank），用于调参。

    列表按 RRF 融合分排序（能看到召回本身的排序），
    重排分挂在每条结果的 scores 里（能看到精排之后谁会上位），
    两者的差异正是调参时要看的东西。
    """
    settings = get_settings()
    # 召回阶段至少要取 RETRIEVE_TOP_N 条进融合（不然融合的候选池太小），
    # 但返回给前端只看用户要的几条 —— top_k=3 就该回 3 条，不是 50 条。
    limit = max(req.top_k, settings.retrieve_top_n)
    top_n = req.top_k

    chunks = await asyncio.to_thread(retriever.retrieve_raw, req.query, limit)

    # 重排只用来补一个分数，不改列表顺序 —— 顺序留给 RRF 看
    if retriever.reranker is not None and chunks:
        reranked = await asyncio.to_thread(
            retriever.reranker.rerank, req.query, chunks, top_k=len(chunks)
        )
        rerank_scores = {c.chunk_id: c.score for c in reranked}
        for chunk in chunks:
            if chunk.chunk_id in rerank_scores:
                chunk.scores["rerank"] = rerank_scores[chunk.chunk_id]

    return {
        "query": req.query,
        "top_n": top_n,
        "count": len(chunks),
        "results": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "source_path": c.source_path,
                "source_type": c.source_type,
                "title": c.title,
                "heading_path": c.heading_path,
                "page": c.page,
                "chunk_idx": c.chunk_idx,
                "score": round(float(c.score), 6),
                "scores": {k: round(float(v), 6) for k, v in c.scores.items()},
            }
            for c in chunks[:top_n]
        ],
    }
