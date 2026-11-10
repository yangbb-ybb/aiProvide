"""POST /api/chat —— SSE 流式问答。"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..config import get_settings
from ..deps import get_ollama, get_retriever, get_vector_store
from ..llm.ollama import OllamaClient, OllamaError
from ..rag.chain import rag_answer_stream
from ..schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# 知识库是空的时候用这条提示词退化成普通对话。
# 注意：知识库非空但没召回到相关内容时**不能**用这条 —— 那时要老实说「无法回答」。
_PLAIN_SYSTEM = "你是一个乐于助人的中文 AI 助手。回答简洁、直接。"


def _sse(payload: dict) -> str:
    """把一个事件序列化成 SSE 帧。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    req: ChatRequest,
    client: OllamaClient = Depends(get_ollama),
    retriever=Depends(get_retriever),
    vector_store=Depends(get_vector_store),
) -> StreamingResponse:
    """流式问答。

    知识库里有文档时：先检索 → 组装带引用编号的上下文 → 生成 → 推送 citation。
    知识库还是空的：退化成普通对话，让 M0 阶段也能正常聊天。
    """
    settings = get_settings()
    top_k = req.top_k or settings.rerank_top_k

    async def gen():
        try:
            kb_empty = await asyncio.to_thread(vector_store.count) == 0
            if kb_empty:
                logger.info("知识库为空，走直通对话")
                async for piece in client.chat_stream(system=_PLAIN_SYSTEM, user=req.question):
                    yield _sse({"type": "delta", "text": piece})
                yield _sse({"type": "done"})
                return

            async for event_type, payload in rag_answer_stream(
                question=req.question,
                client=client,
                retriever=retriever,
                top_k=top_k,
                context_chunk_chars=settings.context_chunk_chars,
            ):
                if event_type == "delta":
                    yield _sse({"type": "delta", "text": payload})
                elif event_type == "citation":
                    yield _sse({"type": "citation", "citations": payload})
                elif event_type == "done":
                    yield _sse({"type": "done"})
        except OllamaError as exc:
            yield _sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - 兜底，别让连接挂着
            logger.exception("问答链路异常")
            yield _sse({"type": "error", "message": f"处理失败：{exc}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
