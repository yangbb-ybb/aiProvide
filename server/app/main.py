"""FastAPI 应用装配。

启动：uv run uvicorn app.main:app --reload --port 8000
接口文档：http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, documents, health, search, sessions
from .config import get_settings
from .deps import get_conversation_store
from .logging_config import configure_logging
from .db.migrate import migrate

settings = get_settings()
configure_logging(settings)
migrate(settings.conversation_db_path)
logger = logging.getLogger("app.request")

app = FastAPI(
    title="本地 RAG 知识库",
    description="完全本地运行的文档问答服务（Ollama + Qwen3-Embedding），不调用任何云端 AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """记录每个接口的状态码、耗时、来源和会话 ID。"""
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    session_id = request.headers.get("X-Session-ID")
    started = time.perf_counter()
    status_code = 500
    error = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        client_ip = request.client.host if request.client else None
        extra = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip,
            "session_id": session_id,
        }
        logger.info("接口请求完成", extra=extra)
        store = get_conversation_store()
        await asyncio.to_thread(
            store.log_api_request,
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query=request.url.query,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            session_id=session_id,
            error=error,
        )


app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")


@app.get("/")
async def root() -> dict:
    """根路径只做提示，真正的入口在 /docs。"""
    return {
        "name": "本地 RAG 知识库",
        "docs": "/docs",
        "health": "/api/health",
    }
