"""轻量级、版本化 SQLite SQL 迁移器。

用法：
    uv run python -m app.db.migrate
    uv run python -m app.db.migrate --status
    uv run python -m app.db.migrate --check
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _discover() -> list[tuple[int, str, Path, str]]:
    migrations = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        try:
            version = int(path.name.split("_", 1)[0])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"迁移文件名必须以数字开头：{path.name}") from exc
        sql = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        migrations.append((version, path.name, path, checksum))
    if len({item[0] for item in migrations}) != len(migrations):
        raise RuntimeError("迁移版本号重复")
    return migrations


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_meta(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def applied_versions(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    _ensure_meta(conn)
    return {
        int(row[0]): (str(row[1]), str(row[2]))
        for row in conn.execute("SELECT version, name, checksum FROM schema_migrations")
    }


def migrate(db_path: Path, *, check_only: bool = False) -> tuple[int, int]:
    migrations = _discover()
    with _connect(db_path) as conn:
        applied = applied_versions(conn)
        for version, name, _path, checksum in migrations:
            if version in applied:
                old_name, old_checksum = applied[version]
                if old_name != name or old_checksum != checksum:
                    raise RuntimeError(f"已执行迁移被修改：{name}")
                continue
            if check_only:
                continue
            sql = _path.read_text(encoding="utf-8")
            try:
                with conn:
                    conn.executescript(sql)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
                        (version, name, checksum, datetime.now(UTC).isoformat()),
                    )
            except Exception as exc:
                raise RuntimeError(f"迁移 {name} 执行失败，已回滚") from exc
        return len(applied), len(migrations)


def main() -> int:
    parser = argparse.ArgumentParser(description="执行或检查本地 SQLite 数据库迁移")
    parser.add_argument("--status", action="store_true", help="显示当前和最新版本")
    parser.add_argument("--check", action="store_true", help="只检查，不修改数据库")
    args = parser.parse_args()
    settings = get_settings()
    current, latest = migrate(settings.conversation_db_path, check_only=args.check or args.status)
    if args.status or args.check:
        print(f"当前已应用迁移：{current}，最新迁移：{latest}")
        if current < latest:
            print("数据库存在待执行迁移，请运行：uv run python -m app.db.migrate")
            return 1 if args.check else 0
        print("数据库结构已是最新版本")
        return 0
    print(f"数据库迁移完成：{current} -> {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
