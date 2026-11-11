"""POST /api/chat —— SSE 流式问答。"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..config import get_settings
from ..deps import get_conversation_store, get_ollama, get_retriever, get_vector_store
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
    conversation_store=Depends(get_conversation_store),
) -> StreamingResponse:
    """流式问答。

    知识库里有文档时：先检索 → 组装带引用编号的上下文 → 生成 → 推送 citation。
    知识库还是空的：退化成普通对话，让 M0 阶段也能正常聊天。
    """
    settings = get_settings()
    top_k = req.top_k or settings.rerank_top_k
    session_id = await asyncio.to_thread(
        conversation_store.ensure_session,
        req.session_id,
        req.question,
    )
    turn_id = uuid4().hex
    await asyncio.to_thread(
        conversation_store.add_message,
        session_id,
        turn_id,
        "user",
        req.question,
    )

    async def gen():
        answer_parts: list[str] = []
        citations: list[dict] = []
        error_message: str | None = None
        try:
            kb_empty = await asyncio.to_thread(vector_store.count) == 0
            if kb_empty:
                logger.info("知识库为空，走直通对话")
                async for piece in client.chat_stream(system=_PLAIN_SYSTEM, user=req.question):
                    answer_parts.append(piece)
                    yield _sse({"type": "delta", "text": piece})
                yield _sse({"type": "done", "session_id": session_id})
                return

            async for event_type, payload in rag_answer_stream(
                question=req.question,
                client=client,
                retriever=retriever,
                top_k=top_k,
                context_chunk_chars=settings.context_chunk_chars,
                history=[item.model_dump() for item in req.history],
            ):
                if event_type == "delta":
                    answer_parts.append(str(payload))
                    yield _sse({"type": "delta", "text": payload})
                elif event_type == "citation":
                    citations = list(payload)
                    yield _sse({"type": "citation", "citations": payload})
                elif event_type == "done":
                    yield _sse({"type": "done", "session_id": session_id})
        except OllamaError as exc:
            error_message = str(exc)
            yield _sse({"type": "error", "message": error_message, "session_id": session_id})
        except Exception as exc:  # noqa: BLE001 - 兜底，别让连接挂着
            logger.exception("问答链路异常")
            error_message = "处理失败，请查看服务日志"
            yield _sse({"type": "error", "message": error_message, "session_id": session_id})
        finally:
            await asyncio.to_thread(
                conversation_store.add_message,
                session_id,
                turn_id,
                "assistant",
                "".join(answer_parts),
                citations=citations,
                status="error" if error_message else "complete",
                error=error_message,
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
