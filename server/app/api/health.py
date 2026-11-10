"""GET /api/health —— 探活，顺带报告本地模型状态。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..config import get_settings
from ..deps import get_ollama
from ..llm.ollama import OllamaClient, OllamaError

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(client: OllamaClient = Depends(get_ollama)) -> dict:
    """返回各依赖的状态。ollama 字段为 "ok" 或一段可操作的中文提示。"""
    s = get_settings()
    try:
        models = await client.list_models()
        ollama_status = "ok"
    except OllamaError as exc:
        models = []
        ollama_status = str(exc)

    return {
        "status": "ok",
        "ollama": ollama_status,
        "chat_model": s.ollama_chat_model,
        "models": models,
        "embed_model": s.embed_model,
        "vector_dim": s.vector_dim,
        "rerank_model": s.rerank_model,
        "rerank_enabled": s.rerank_enabled,
    }
