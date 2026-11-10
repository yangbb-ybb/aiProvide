"""POST /api/chat —— SSE 流式问答。"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..deps import get_ollama
from ..llm.ollama import OllamaClient, OllamaError
from ..schemas import ChatRequest

router = APIRouter(tags=["chat"])

# M0 阶段的直通提示词；M1 起换成 rag/prompt.py 里的正式提示词 + 检索上下文。
_PLACEHOLDER_SYSTEM = "你是一个乐于助人的中文 AI 助手。回答简洁、直接。"


def _sse(payload: dict) -> str:
    """把一个事件序列化成 SSE 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, client: OllamaClient = Depends(get_ollama)) -> StreamingResponse:
    """流式问答。

    M0：问题直通 Ollama，用于打通「前端 → 后端 → 本地模型 → 流式返回」链路。
    M1 起：先检索知识库 → 组装带引用编号的上下文 → 生成 → 追加 citation 事件。
    """
    system = _PLACEHOLDER_SYSTEM
    user = req.question

    async def gen():
        try:
            async for piece in client.chat_stream(system=system, user=user):
                yield _sse({"type": "delta", "text": piece})
            yield _sse({"type": "done"})
        except OllamaError as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
