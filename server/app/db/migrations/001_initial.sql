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
