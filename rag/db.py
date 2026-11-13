"""SQLite storage for chat sessions, messages and indexed files.

Every session belongs to a workspace, so visitors of a public demo never see
each other's documents or conversations.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    workspace   TEXT NOT NULL,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    filename    TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    pages       INTEGER,
    chunks      INTEGER,
    created_at  TEXT NOT NULL,
    UNIQUE (session_id, file_hash)
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self._conn()) as con:
            con.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _run(self, sql: str, params: tuple = ()) -> list[tuple]:
        with closing(self._conn()) as con, con:
            return con.execute(sql, params).fetchall()

    # Sessions
    def create_session(self, workspace: str, name: str) -> str:
        sid = uuid.uuid4().hex[:8]
        self._run("INSERT INTO sessions VALUES (?,?,?,?,?)", (sid, workspace, name, _now(), _now()))
        return sid

    def sessions(self, workspace: str) -> list[tuple[str, str]]:
        return self._run(
            "SELECT id, name FROM sessions WHERE workspace=? ORDER BY updated_at DESC", (workspace,)
        )

    def delete_session(self, workspace: str, sid: str) -> None:
        if not self._run("SELECT 1 FROM sessions WHERE id=? AND workspace=?", (sid, workspace)):
            return
        for table in ("messages", "files"):
            self._run(f"DELETE FROM {table} WHERE session_id=?", (sid,))
        self._run("DELETE FROM sessions WHERE id=?", (sid,))

    # Messages
    def add_message(self, sid: str, role: str, content: str) -> None:
        self._run("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                  (sid, role, content, _now()))
        self._run("UPDATE sessions SET updated_at=? WHERE id=?", (_now(), sid))

    def messages(self, sid: str) -> list[tuple[str, str]]:
        return self._run("SELECT role, content FROM messages WHERE session_id=? ORDER BY id", (sid,))

    def clear_messages(self, sid: str) -> None:
        self._run("DELETE FROM messages WHERE session_id=?", (sid,))

    # Files
    def has_file(self, sid: str, file_hash: str) -> bool:
        return bool(self._run("SELECT 1 FROM files WHERE session_id=? AND file_hash=?", (sid, file_hash)))

    def add_file(self, sid: str, filename: str, file_hash: str, pages: int, chunks: int) -> None:
        self._run(
            "INSERT OR IGNORE INTO files (session_id, filename, file_hash, pages, chunks, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (sid, filename, file_hash, pages, chunks, _now()),
        )

    def files(self, sid: str) -> list[tuple[str, int, int]]:
        return self._run("SELECT filename, pages, chunks FROM files WHERE session_id=? ORDER BY id", (sid,))
