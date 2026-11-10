"""FastAPI 应用装配。

启动：uv run uvicorn app.main:app --reload --port 8000
接口文档：http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, documents, health, search
from .config import get_settings

settings = get_settings()

app = FastAPI(
    title="本地 RAG 知识库",
    description="完全本地运行的文档问答服务（Ollama + bge-m3），不调用任何云端 AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")


@app.get("/")
async def root() -> dict:
    """根路径只做提示，真正的入口在 /docs。"""
    return {
        "name": "本地 RAG 知识库",
        "docs": "/docs",
        "health": "/api/health",
    }
