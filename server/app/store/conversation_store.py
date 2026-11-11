"""SQLite 会话、消息与接口审计存储。

SQLite 属于 Python 标准库，不增加外部服务；WAL 模式适合本项目的单机并发读写。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4


class ConversationStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'complete',
                    error TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_created
                    ON messages(session_id, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_messages_turn
                    ON messages(turn_id);

                CREATE TABLE IF NOT EXISTS api_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    query TEXT NOT NULL DEFAULT '',
                    status_code INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    client_ip TEXT,
                    session_id TEXT,
                    error TEXT,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_api_logs_created
                    ON api_logs(created_at);
                CREATE INDEX IF NOT EXISTS idx_api_logs_request
                    ON api_logs(request_id);
                """
            )

    def ensure_session(self, session_id: str | None, first_question: str = "") -> str:
        session_id = session_id or uuid4().hex
        now = time.time()
        title = first_question.strip()[:80]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (session_id, title, now, now),
            )
        return session_id

    def add_message(
        self,
        session_id: str,
        turn_id: str,
        role: str,
        content: str,
        *,
        citations: list[dict] | None = None,
        status: str = "complete",
        error: str | None = None,
    ) -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages(
                    session_id, turn_id, role, content, citations_json, status, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_id,
                    role,
                    content,
                    json.dumps(citations or [], ensure_ascii=False),
                    status,
                    error,
                    now,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
            return int(cursor.lastrowid)

    def list_messages(self, session_id: str, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, turn_id, role, content, citations_json,
                       status, error, created_at
                FROM messages
                WHERE session_id=?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            {
                **dict(row),
                "citations": json.loads(row["citations_json"] or "[]"),
            }
            for row in rows
        ]

    def log_api_request(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        query: str,
        status_code: int,
        duration_ms: float,
        client_ip: str | None,
        session_id: str | None,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_logs(
                    request_id, method, path, query, status_code, duration_ms,
                    client_ip, session_id, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    method,
                    path,
                    query,
                    status_code,
                    duration_ms,
                    client_ip,
                    session_id,
                    error,
                    time.time(),
                ),
            )
