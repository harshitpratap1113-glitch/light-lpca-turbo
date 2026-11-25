"""
Sovereign Local-First BYODB Vault (SQLite WAL + FTS5 Memory + TTL Cache + BYOK Registry).
Replaces slow remote round-trips with sub-millisecond local persistence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> Path:
    if os.environ.get("LPCA_TURBO_DB"):
        return Path(os.environ["LPCA_TURBO_DB"])
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return Path("/tmp/light_lpca_turbo/vault.db")
    return Path.home() / ".light_lpca_turbo" / "vault.db"


DEFAULT_DB_PATH = _default_db_path()


class SovereignVault:
    """Thread-safe, WAL-enabled SQLite vault for cache, FTS5 memory, chat, and Shadow Brain tasks."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS page_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_markdown TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    snippets_json TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS report_cache (
                    topic_key TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL DEFAULT 'general',
                    fact TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'chat',
                    importance INTEGER NOT NULL DEFAULT 5,
                    created_at REAL NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    fact, category, content='memories', content_rowid='id'
                );

                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, fact, category)
                    VALUES (new.id, new.fact, new.category);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, fact, category)
                    VALUES ('delete', old.id, old.fact, old.category);
                END;

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'CORE_STREAM',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    provider TEXT DEFAULT '',
                    latency_ms REAL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shadow_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    interval_minutes INTEGER NOT NULL DEFAULT 60,
                    last_run_at REAL DEFAULT 0,
                    last_result_summary TEXT DEFAULT '',
                    confidence_score REAL DEFAULT 0.0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shadow_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    # ---- Page & Report Caching ----
    def get_cached_page(self, url: str, ttl_seconds: int = 3600) -> Optional[Dict[str, Any]]:
        key = self._hash(url)
        cutoff = time.time() - ttl_seconds
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM page_cache WHERE url_hash = ? AND fetched_at >= ?",
                (key, cutoff),
            ).fetchone()
            if not row:
                return None
            return {
                "url": row["url"],
                "title": row["title"],
                "content_markdown": row["content_markdown"],
                "status_code": row["status_code"],
                "word_count": row["word_count"],
                "snippets": json.loads(row["snippets_json"]),
            }

    def set_cached_page(
        self,
        url: str,
        title: str,
        content_markdown: str,
        status_code: int,
        word_count: int,
        snippets: List[str],
    ) -> None:
        key = self._hash(url)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO page_cache (url_hash, url, title, content_markdown, status_code, word_count, snippets_json, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    title=excluded.title,
                    content_markdown=excluded.content_markdown,
                    status_code=excluded.status_code,
                    word_count=excluded.word_count,
                    snippets_json=excluded.snippets_json,
                    fetched_at=excluded.fetched_at
                """,
                (key, url, title, content_markdown, status_code, word_count, json.dumps(snippets), time.time()),
            )

    def get_cached_report(self, topic: str, ttl_seconds: int = 1800) -> Optional[Dict[str, Any]]:
        key = self._hash(topic)
        cutoff = time.time() - ttl_seconds
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM report_cache WHERE topic_key = ? AND created_at >= ?",
                (key, cutoff),
            ).fetchone()
            return json.loads(row["report_json"]) if row else None

    def set_cached_report(self, topic: str, report_dict: Dict[str, Any]) -> None:
        key = self._hash(topic)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO report_cache (topic_key, topic, report_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(topic_key) DO UPDATE SET
                    report_json=excluded.report_json,
                    created_at=excluded.created_at
                """,
                (key, topic, json.dumps(report_dict), time.time()),
            )

    # ---- FTS5 Cognitive Memory ----
    def add_memory(self, fact: str, category: str = "general", source: str = "chat", importance: int = 5) -> int:
        clean_fact = fact.strip()
        if not clean_fact:
            return 0
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM memories WHERE fact = ?", (clean_fact,)).fetchone()
            if existing:
                return int(existing["id"])
            cur = conn.execute(
                "INSERT INTO memories (category, fact, source, importance, created_at) VALUES (?, ?, ?, ?, ?)",
                (category, clean_fact, source, importance, time.time()),
            )
            return int(cur.lastrowid)

    def search_memories(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        tokens = [t for t in query.replace('"', " ").split() if len(t) >= 2]
        with self._connect() as conn:
            if not tokens:
                rows = conn.execute(
                    "SELECT id, category, fact, source, importance, created_at FROM memories ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            fts_query = " OR ".join(f'"{t}"' for t in tokens[:8])
            try:
                rows = conn.execute(
                    """
                    SELECT m.id, m.category, m.fact, m.source, m.importance, m.created_at
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25(memories_fts), m.importance DESC
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    "SELECT id, category, fact, source, importance, created_at FROM memories ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def list_memories(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, category, fact, source, importance, created_at FROM memories ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- Chat Stream ----
    def add_message(self, role: str, content: str, session_id: str = "CORE_STREAM", provider: str = "", latency_ms: float = 0.0) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, provider, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, provider, latency_ms, time.time()),
            )
            return int(cur.lastrowid)

    def get_history(self, session_id: str = "CORE_STREAM", limit: int = 40) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, role, content, provider, latency_ms, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    # ---- Shadow Brain Tasks & Logs ----
    def add_shadow_task(self, title: str, query: str, interval_minutes: int = 60) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO shadow_tasks (title, query, status, interval_minutes, created_at) VALUES (?, ?, 'pending', ?, ?)",
                (title, query, interval_minutes, time.time()),
            )
            return int(cur.lastrowid)

    def list_shadow_tasks(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM shadow_tasks ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_due_shadow_tasks(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM shadow_tasks
                WHERE status IN ('pending', 'active')
                  AND (last_run_at = 0 OR (? - last_run_at) >= (interval_minutes * 60))
                ORDER BY last_run_at ASC, id ASC
                """,
                (now,),
            ).fetchall()
            return [dict(r) for r in rows]

    def complete_shadow_task(self, task_id: int, summary: str, confidence_score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE shadow_tasks
                SET status = 'active', last_run_at = ?, last_result_summary = ?, confidence_score = ?
                WHERE id = ?
                """,
                (time.time(), summary, confidence_score, task_id),
            )

    def log_shadow_event(self, action: str, details: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO shadow_logs (action, details, created_at) VALUES (?, ?, ?)",
                (action, details, time.time()),
            )

    def get_shadow_logs(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, action, details, created_at FROM shadow_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ---- Settings / BYOK Config ----
    def set_setting(self, key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), time.time()),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return json.loads(row["value"]) if row else default

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            return {
                "cached_pages": conn.execute("SELECT COUNT(*) FROM page_cache").fetchone()[0],
                "cached_reports": conn.execute("SELECT COUNT(*) FROM report_cache").fetchone()[0],
                "memories": conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                "shadow_tasks": conn.execute("SELECT COUNT(*) FROM shadow_tasks").fetchone()[0],
                "shadow_enabled": bool(self.get_setting("shadow_enabled", True)),
                "db_path": str(self.db_path),
            }

    def db_diagnostics(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tables = [
            "messages",
            "memories",
            "page_cache",
            "report_cache",
            "shadow_tasks",
            "shadow_logs",
            "settings",
        ]
        collections = []
        total_records = 0
        with self._connect() as conn:
            for tbl in tables:
                cnt = int(conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0])
                total_records += cnt
                collections.append(
                    {
                        "name": tbl,
                        "records": cnt,
                        "size_mb": round(max(0.01, cnt * 0.0006), 2),
                    }
                )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        file_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "status": "Connected",
            "provider": "Sovereign SQLite WAL + FTS5",
            "type": "Local-First Embedded DB",
            "name": self.db_path.name,
            "path": str(self.db_path),
            "latency_ms": latency_ms,
            "used_mb": round(file_bytes / (1024 * 1024), 2),
            "total_records": total_records,
            "collections": collections,
        }
