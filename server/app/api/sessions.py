"""会话查询接口。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from ..deps import get_conversation_store

router = APIRouter(tags=["sessions"])


@router.get("/sessions/{session_id}/messages")
async def list_session_messages(
    session_id: str,
    store=Depends(get_conversation_store),
) -> dict:
    messages = await asyncio.to_thread(store.list_messages, session_id)
    return {"session_id": session_id, "messages": messages}
