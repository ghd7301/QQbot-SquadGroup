"""SQLite persistence: chat log, memory candidates, long-term memory, queue journal
(design doc §14, §9, queue persistence)."""
from __future__ import annotations

import json
import sqlite3
import threading

from .config import Config


class Store:
    def __init__(self, config: Config) -> None:
        self.conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with self._lock:
            c = self.conn
            c.execute(
                """CREATE TABLE IF NOT EXISTS chat_log (
                    id INTEGER PRIMARY KEY,
                    group_id INTEGER,
                    user_id TEXT,
                    message TEXT,
                    bot_reply TEXT,
                    reply_type TEXT,
                    ts REAL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS memory_candidate (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    source_turn TEXT,
                    evidence_count INTEGER DEFAULT 1,
                    first_seen REAL,
                    last_seen REAL,
                    confidence REAL DEFAULT 0.5,
                    explicit INTEGER DEFAULT 0,
                    expires_at REAL,
                    upgraded INTEGER DEFAULT 0
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    kind TEXT,
                    ts REAL
                )"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS queue_journal (
                    id INTEGER PRIMARY KEY,
                    payload TEXT,
                    created REAL
                )"""
            )
            c.commit()

    # ---- chat log ----
    def add_chat_log(self, group_id: int, user_id: str, message: str, bot_reply: str, reply_type: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO chat_log (group_id, user_id, message, bot_reply, reply_type, ts) VALUES (?,?,?,?,?,?)",
                (group_id, user_id, message, bot_reply, reply_type, __import__("time").time()),
            )
            self.conn.commit()

    # ---- memory ----
    def add_memory(self, content: str, kind: str = "fact") -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory (content, kind, ts) VALUES (?,?,?)",
                (content, kind, __import__("time").time()),
            )
            self.conn.commit()

    def add_memory_candidate(self, content: str, source_turn: str, explicit: bool = False, expires_at: float | None = None) -> None:
        import time
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "SELECT id, evidence_count FROM memory_candidate WHERE content=? AND upgraded=0",
                (content,),
            ).fetchone()
            if cur:
                self.conn.execute(
                    "UPDATE memory_candidate SET evidence_count=evidence_count+1, last_seen=?, confidence=MIN(1.0, confidence+0.1) WHERE id=?",
                    (now, cur[0]),
                )
            else:
                self.conn.execute(
                    "INSERT INTO memory_candidate (content, source_turn, evidence_count, first_seen, last_seen, explicit, expires_at) VALUES (?,?,?,?,?,?,?)",
                    (content, source_turn, 1, now, now, 1 if explicit else 0, expires_at),
                )
            self.conn.commit()

    def upgrade_candidates(self) -> int:
        """Promote eligible candidates to long-term memory. Returns count upgraded."""
        import time
        now = time.time()
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, content, explicit, evidence_count FROM memory_candidate WHERE upgraded=0", ()
            ).fetchall()
            upgraded = 0
            for cid, content, explicit, evidence in rows:
                eligible = explicit or evidence >= 2
                if eligible:
                    self.conn.execute("INSERT INTO memory (content, kind, ts) VALUES (?,?,?)", (content, "fact", now))
                    self.conn.execute("UPDATE memory_candidate SET upgraded=1 WHERE id=?", (cid,))
                    upgraded += 1
            self.conn.commit()
            return upgraded

    def memory_contents(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self.conn.execute("SELECT content FROM memory").fetchall()]

    # ---- queue journal ----
    def push_queue(self, item: dict) -> int:
        import time
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO queue_journal (payload, created) VALUES (?,?)",
                (json.dumps(item, ensure_ascii=False), time.time()),
            )
            self.conn.commit()
            return cur.lastrowid

    def pending_queue(self) -> list[tuple[int, dict]]:
        with self._lock:
            return [(r[0], json.loads(r[1])) for r in self.conn.execute("SELECT id, payload FROM queue_journal").fetchall()]

    def clear_queue_item(self, item_id: int) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM queue_journal WHERE id=?", (item_id,))
            self.conn.commit()
