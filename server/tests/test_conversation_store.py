"""SQLite 会话和接口审计存储测试。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.store.conversation_store import ConversationStore


def test_session_messages_are_persisted(tmp_path):
    store = ConversationStore(tmp_path / "app.db")
    session_id = store.ensure_session(None, "第一个问题")
    store.add_message(session_id, "turn-1", "user", "第一个问题")
    store.add_message(
        session_id,
        "turn-1",
        "assistant",
        "第一个回答",
        citations=[{"index": 1, "source_path": "README.md"}],
    )

    messages = store.list_messages(session_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "第一个回答"
    assert messages[1]["citations"][0]["source_path"] == "README.md"


def test_api_log_is_persisted(tmp_path):
    store = ConversationStore(tmp_path / "app.db")
    store.log_api_request(
        request_id="req-1",
        method="POST",
        path="/api/chat",
        query="",
        status_code=200,
        duration_ms=12.5,
        client_ip="127.0.0.1",
        session_id="session-1",
    )

    with store._connect() as conn:
        row = conn.execute("SELECT * FROM api_logs WHERE request_id='req-1'").fetchone()
    assert row["path"] == "/api/chat"
    assert row["status_code"] == 200
    assert row["session_id"] == "session-1"
