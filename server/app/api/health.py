"""GET /api/health —— 探活，顺带报告本地模型状态。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from ..config import get_settings
from ..deps import get_ollama
from ..llm.ollama import OllamaClient, OllamaError

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(response: Response, client: OllamaClient = Depends(get_ollama)) -> dict:
    """返回各依赖的状态。ollama 字段为 "ok" 或一段可操作的中文提示。"""
    s = get_settings()
    try:
        models = await client.list_models()
        ollama_status = "ok"
    except OllamaError as exc:
        models = []
        ollama_status = str(exc)

    if ollama_status != "ok":
        response.status_code = 503
    return {
        "status": "ok" if ollama_status == "ok" else "degraded",
        "ollama": ollama_status,
        "chat_model": s.ollama_chat_model,
        "models": models,
        "embed_model": s.embed_model,
        "vector_dim": s.vector_dim,
        "rerank_model": s.rerank_model,
        "rerank_enabled": s.rerank_enabled,
    }
