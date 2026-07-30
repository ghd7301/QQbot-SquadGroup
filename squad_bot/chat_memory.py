from __future__ import annotations

import json
import logging
import math
import queue
import re
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]", re.I)
PATH_RE = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\)[^\s\"']+")
URL_RE = re.compile(r"https?://[^\s\"']+")
SECRET_RE = re.compile(r"(?i)(api[_ -]?key|token|authorization)\s*[:=]\s*\S+")
QQ_RE = re.compile(r"(?<!\d)\d{7,12}(?!\d)")


def lexical_terms(text: str) -> list[str]:
    normalized = "".join(TOKEN_RE.findall(str(text or "").lower()))
    terms: list[str] = []
    for size in (1, 2, 3):
        for index in range(max(0, len(normalized) - size + 1)):
            term = normalized[index : index + size]
            if term not in terms:
                terms.append(term)
    return terms[:160]


def redact_for_model(text: str) -> str:
    value = PATH_RE.sub("[本地路径已隐藏]", str(text or ""))
    value = SECRET_RE.sub("[密钥已隐藏]", value)
    value = URL_RE.sub("[链接已隐藏]", value)
    return QQ_RE.sub("[账号已隐藏]", value)


def pack_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes, dimensions: int) -> tuple[float, ...]:
    if not blob or dimensions <= 0 or len(blob) != dimensions * 4:
        return ()
    return struct.unpack(f"<{dimensions}f", blob)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


@dataclass(frozen=True)
class MemoryMessage:
    group_id: int
    message_id: str
    speaker_id: str
    display_name: str
    speaker_role: str
    text: str
    event_time: float
    reply_message_id: str = ""
    reply_speaker_id: str = ""
    quoted_text: str = ""
    mentions: tuple[str, ...] = ()
    generated_for_message_ids: tuple[str, ...] = ()
    turn_id: str = ""
    reply_mode: str = ""
    semantic_topic: str = ""
    sequence: int = 0
    received_time: float = 0.0
    content_segments: tuple[dict[str, str], ...] = ()
    message_status: str = "active"


@dataclass(frozen=True)
class MemoryHit:
    chunk_id: int
    group_id: int
    text: str
    speaker_ids: tuple[str, ...]
    event_time: float
    topic_id: int
    message_ids: tuple[str, ...]
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TopicAssignment:
    topic_id: int
    confidence: float
    relation_basis: str
    is_primary: bool
    anchor_message_ids: tuple[str, ...] = ()


class ChatMemoryStore:
    def __init__(self, path: str | Path, embedding: EmbeddingProvider):
        self.path = Path(path)
        self.embedding = embedding
        self._write_lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    message_id TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    speaker_role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    event_time REAL NOT NULL,
                    reply_message_id TEXT NOT NULL DEFAULT '',
                    reply_speaker_id TEXT NOT NULL DEFAULT '',
                    quoted_text TEXT NOT NULL DEFAULT '',
                    mentions_json TEXT NOT NULL DEFAULT '[]',
                    utterance_id INTEGER,
                    topic_id INTEGER,
                    recalled INTEGER NOT NULL DEFAULT 0,
                    searchable INTEGER NOT NULL DEFAULT 1,
                    sequence INTEGER NOT NULL DEFAULT 0,
                    received_time REAL NOT NULL DEFAULT 0,
                    content_segments_json TEXT NOT NULL DEFAULT '[]',
                    message_status TEXT NOT NULL DEFAULT 'active',
                    UNIQUE(group_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS chat_messages_group_time
                    ON chat_messages(group_id, event_time DESC);
                CREATE INDEX IF NOT EXISTS chat_messages_reply
                    ON chat_messages(group_id, reply_message_id);
                CREATE TABLE IF NOT EXISTS topic_sessions (
                    id INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lexical_terms TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS topic_sessions_group_time
                    ON topic_sessions(group_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    utterance_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    speaker_ids_json TEXT NOT NULL,
                    message_ids_json TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL NOT NULL,
                    search_text TEXT NOT NULL,
                    embedding BLOB,
                    dimensions INTEGER NOT NULL DEFAULT 0,
                    embedding_provider TEXT NOT NULL DEFAULT '',
                    active INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS memory_chunks_group_time
                    ON memory_chunks(group_id, ended_at DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
                    chunk_id UNINDEXED, group_id UNINDEXED, search_text,
                    tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS message_relations (
                    id INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    source_message_id TEXT NOT NULL,
                    target_message_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    relation_basis TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    UNIQUE(group_id, source_message_id, target_message_id, relation_type)
                );
                CREATE INDEX IF NOT EXISTS message_relations_source
                    ON message_relations(group_id, source_message_id, relation_type);
                CREATE INDEX IF NOT EXISTS message_relations_target
                    ON message_relations(group_id, target_message_id, relation_type);
                CREATE TABLE IF NOT EXISTS message_topic_relations (
                    id INTEGER PRIMARY KEY,
                    group_id INTEGER NOT NULL,
                    message_id TEXT NOT NULL,
                    topic_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    relation_basis TEXT NOT NULL,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    anchor_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    UNIQUE(group_id, message_id, topic_id)
                );
                CREATE INDEX IF NOT EXISTS message_topic_relations_message
                    ON message_topic_relations(group_id, message_id, is_primary DESC);
                CREATE INDEX IF NOT EXISTS message_topic_relations_topic
                    ON message_topic_relations(group_id, topic_id, confidence DESC);
                """
            )
            existing_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(chat_messages)")
            }
            migrations = {
                "generated_for_message_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "turn_id": "TEXT NOT NULL DEFAULT ''",
                "reply_mode": "TEXT NOT NULL DEFAULT ''",
                "semantic_topic": "TEXT NOT NULL DEFAULT ''",
                "sequence": "INTEGER NOT NULL DEFAULT 0",
                "received_time": "REAL NOT NULL DEFAULT 0",
                "content_segments_json": "TEXT NOT NULL DEFAULT '[]'",
                "message_status": "TEXT NOT NULL DEFAULT 'active'",
            }
            for column, definition in migrations.items():
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE chat_messages ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS chat_messages_topic_role_time "
                "ON chat_messages(group_id, topic_id, speaker_role, event_time DESC)"
            )
            connection.execute(
                """INSERT OR IGNORE INTO message_topic_relations
                   (group_id,message_id,topic_id,confidence,relation_basis,is_primary,
                    anchor_message_ids_json,created_at)
                   SELECT group_id,message_id,topic_id,0.7,'legacy_primary',1,'[]',event_time
                   FROM chat_messages
                   WHERE topic_id IS NOT NULL AND topic_id > 0 AND recalled=0"""
            )

    def add_message(self, message: MemoryMessage) -> bool:
        if not message.text.strip() or not message.message_id:
            return False
        message_status = str(message.message_status or "active").strip().lower()
        searchable = int(message_status not in {"recalled", "invalid"})
        with self._write_lock, self.connect() as connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO chat_messages
                   (group_id,message_id,speaker_id,display_name,speaker_role,text,event_time,
                    reply_message_id,reply_speaker_id,quoted_text,mentions_json,
                    generated_for_message_ids_json,turn_id,reply_mode,semantic_topic,
                    sequence,received_time,content_segments_json,message_status,
                    recalled,searchable)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    message.group_id, message.message_id, message.speaker_id,
                    message.display_name, message.speaker_role, message.text, message.event_time,
                    message.reply_message_id, message.reply_speaker_id, message.quoted_text,
                    json.dumps(message.mentions, ensure_ascii=False),
                    json.dumps(message.generated_for_message_ids, ensure_ascii=False),
                    message.turn_id, message.reply_mode, message.semantic_topic,
                    int(message.sequence),
                    float(message.received_time or message.event_time),
                    json.dumps(message.content_segments, ensure_ascii=False),
                    message_status,
                    int(message_status == "recalled"),
                    searchable,
                ),
            )
            if not cursor.rowcount:
                return False
            if not searchable:
                return True
            row_id = int(cursor.lastrowid)
            assignments = self._choose_topic_assignments(connection, message)
            topic_id = assignments[0].topic_id
            utterance_id, chunk_id = self._choose_utterance(connection, message, topic_id)
            connection.execute(
                "UPDATE chat_messages SET utterance_id=?, topic_id=? WHERE id=?",
                (utterance_id, topic_id, row_id),
            )
            self._record_topic_assignments(
                connection,
                message,
                assignments,
            )
            if chunk_id:
                self._append_to_chunk(connection, chunk_id, message)
            else:
                self._create_chunk(connection, utterance_id, topic_id, message)
            connection.execute(
                "UPDATE topic_sessions SET updated_at=? WHERE id=?",
                (message.event_time, topic_id),
            )
            if message.reply_message_id:
                self._add_relation(
                    connection,
                    group_id=message.group_id,
                    source_message_id=message.message_id,
                    target_message_id=message.reply_message_id,
                    relation_type="qq_reply",
                    relation_basis="onebot",
                    confidence=1.0,
                    created_at=message.event_time,
                )
                replied = connection.execute(
                    "SELECT speaker_role FROM chat_messages WHERE group_id=? AND message_id=?",
                    (message.group_id, message.reply_message_id),
                ).fetchone()
                if replied and str(replied["speaker_role"]) == "bot":
                    self._add_relation(
                        connection,
                        group_id=message.group_id,
                        source_message_id=message.message_id,
                        target_message_id=message.reply_message_id,
                        relation_type="feedback_to",
                        relation_basis="qq_reply",
                        confidence=1.0,
                        created_at=message.event_time,
                    )
            for target_message_id in message.generated_for_message_ids:
                self._add_relation(
                    connection,
                    group_id=message.group_id,
                    source_message_id=message.message_id,
                    target_message_id=target_message_id,
                    relation_type="generated_for",
                    relation_basis="program",
                    confidence=1.0,
                    created_at=message.event_time,
                )
            return True

    @staticmethod
    def _add_relation(
        connection: sqlite3.Connection,
        *,
        group_id: int,
        source_message_id: str,
        target_message_id: str,
        relation_type: str,
        relation_basis: str,
        confidence: float,
        created_at: float,
    ) -> bool:
        source = str(source_message_id or "").strip()
        target = str(target_message_id or "").strip()
        if not source or not target or source == target:
            return False
        cursor = connection.execute(
            """INSERT OR REPLACE INTO message_relations
               (group_id,source_message_id,target_message_id,relation_type,
                relation_basis,confidence,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                int(group_id), source, target, str(relation_type),
                str(relation_basis), max(0.0, min(1.0, float(confidence))),
                float(created_at),
            ),
        )
        return bool(cursor.rowcount)

    def add_relation(
        self,
        *,
        group_id: int,
        source_message_id: str,
        target_message_id: str,
        relation_type: str,
        relation_basis: str = "semantic",
        confidence: float = 1.0,
        created_at: float | None = None,
    ) -> bool:
        with self._write_lock, self.connect() as connection:
            return self._add_relation(
                connection,
                group_id=group_id,
                source_message_id=source_message_id,
                target_message_id=target_message_id,
                relation_type=relation_type,
                relation_basis=relation_basis,
                confidence=confidence,
                created_at=time.time() if created_at is None else created_at,
            )

    def _choose_topic_assignments(
        self,
        connection: sqlite3.Connection,
        message: MemoryMessage,
    ) -> tuple[TopicAssignment, ...]:
        terms = set(lexical_terms(message.text))
        hard_targets = tuple(dict.fromkeys(
            target
            for target in (*message.generated_for_message_ids, message.reply_message_id)
            if target
        ))
        inherited: list[TopicAssignment] = []
        for target_message_id in hard_targets:
            if not target_message_id:
                continue
            rows = connection.execute(
                """SELECT topic_id,confidence,is_primary FROM message_topic_relations
                   WHERE group_id=? AND message_id=?
                   ORDER BY is_primary DESC,confidence DESC LIMIT 2""",
                (message.group_id, target_message_id),
            ).fetchall()
            if not rows:
                legacy = connection.execute(
                    "SELECT topic_id FROM chat_messages WHERE group_id=? AND message_id=? AND recalled=0",
                    (message.group_id, target_message_id),
                ).fetchone()
                rows = [legacy] if legacy and legacy["topic_id"] else []
            basis = (
                "generated_for"
                if target_message_id in message.generated_for_message_ids
                else "qq_reply"
            )
            for row in rows:
                topic_id = int(row["topic_id"])
                if any(item.topic_id == topic_id for item in inherited):
                    continue
                inherited.append(TopicAssignment(
                    topic_id=topic_id,
                    confidence=1.0,
                    relation_basis=basis,
                    is_primary=not inherited,
                    anchor_message_ids=(target_message_id,),
                ))
                if len(inherited) >= 2:
                    break
            if len(inherited) >= 2:
                break
        if inherited:
            return tuple(inherited)
        if not message.mentions:
            previous = connection.execute(
                """SELECT message_id FROM chat_messages
                   WHERE group_id=? AND speaker_id=? AND recalled=0 AND searchable=1
                     AND message_id<>? AND event_time<=? AND event_time>=?
                   ORDER BY event_time DESC,id DESC LIMIT 1""",
                (
                    message.group_id,
                    message.speaker_id,
                    message.message_id,
                    message.event_time,
                    message.event_time - 12,
                ),
            ).fetchone()
            if previous:
                previous_id = str(previous["message_id"])
                rows = connection.execute(
                    """SELECT topic_id,confidence,is_primary FROM message_topic_relations
                       WHERE group_id=? AND message_id=?
                       ORDER BY is_primary DESC,confidence DESC LIMIT 2""",
                    (message.group_id, previous_id),
                ).fetchall()
                if rows:
                    return tuple(TopicAssignment(
                        topic_id=int(row["topic_id"]),
                        confidence=max(0.8, float(row["confidence"])),
                        relation_basis="continuation",
                        is_primary=index == 0,
                        anchor_message_ids=(previous_id,),
                    ) for index, row in enumerate(rows))
        candidates = connection.execute(
            "SELECT id,lexical_terms,updated_at FROM topic_sessions WHERE group_id=? AND updated_at>=? ORDER BY updated_at DESC LIMIT 6",
            (message.group_id, message.event_time - 600),
        ).fetchall()
        scored: list[tuple[int, float, str]] = []
        for row in candidates:
            previous = set(str(row["lexical_terms"]).split())
            score = len(terms & previous) / max(1, min(len(terms), len(previous)))
            scored.append((int(row["id"]), score, str(row["lexical_terms"])))
        scored.sort(key=lambda item: item[1], reverse=True)
        if scored and scored[0][1] >= 0.18:
            selected = [scored[0]]
            if (
                len(scored) > 1
                and scored[1][1] >= 0.24
                and scored[1][1] >= scored[0][1] * 0.75
            ):
                selected.append(scored[1])
            assignments = []
            for index, (topic_id, score, previous_terms) in enumerate(selected):
                merged = " ".join(sorted(terms | set(previous_terms.split())))
                connection.execute(
                    "UPDATE topic_sessions SET lexical_terms=? WHERE id=?",
                    (merged[:4000], topic_id),
                )
                assignments.append(TopicAssignment(
                    topic_id=topic_id,
                    confidence=max(0.18, min(0.95, score)),
                    relation_basis="lexical",
                    is_primary=index == 0,
                ))
            return tuple(assignments)
        cursor = connection.execute(
            "INSERT INTO topic_sessions(group_id,started_at,updated_at,lexical_terms) VALUES(?,?,?,?)",
            (message.group_id, message.event_time, message.event_time, " ".join(sorted(terms))),
        )
        return (TopicAssignment(
            topic_id=int(cursor.lastrowid),
            confidence=1.0,
            relation_basis="new_topic",
            is_primary=True,
        ),)

    @staticmethod
    def _record_topic_assignments(
        connection: sqlite3.Connection,
        message: MemoryMessage,
        assignments: Sequence[TopicAssignment],
    ) -> None:
        connection.execute(
            "DELETE FROM message_topic_relations WHERE group_id=? AND message_id=?",
            (message.group_id, message.message_id),
        )
        for assignment in assignments:
            connection.execute(
                """INSERT INTO message_topic_relations
                   (group_id,message_id,topic_id,confidence,relation_basis,is_primary,
                    anchor_message_ids_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    message.group_id,
                    message.message_id,
                    assignment.topic_id,
                    assignment.confidence,
                    assignment.relation_basis,
                    int(assignment.is_primary),
                    json.dumps(assignment.anchor_message_ids, ensure_ascii=False),
                    message.event_time,
                ),
            )

    def topic_assignments_for_message(
        self,
        group_id: int,
        message_id: str,
    ) -> tuple[TopicAssignment, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT topic_id,confidence,relation_basis,is_primary,anchor_message_ids_json
                   FROM message_topic_relations WHERE group_id=? AND message_id=?
                   ORDER BY is_primary DESC,confidence DESC""",
                (group_id, str(message_id)),
            ).fetchall()
        return tuple(TopicAssignment(
            topic_id=int(row["topic_id"]),
            confidence=float(row["confidence"]),
            relation_basis=str(row["relation_basis"]),
            is_primary=bool(row["is_primary"]),
            anchor_message_ids=tuple(json.loads(row["anchor_message_ids_json"] or "[]")),
        ) for row in rows)

    def _choose_utterance(self, connection, message: MemoryMessage, topic_id: int) -> tuple[int, int]:
        row = connection.execute(
            """SELECT id,utterance_id,speaker_ids_json,ended_at FROM memory_chunks
               WHERE group_id=? AND topic_id=? AND active=1
               ORDER BY ended_at DESC LIMIT 1""",
            (message.group_id, topic_id),
        ).fetchone()
        if row and message.event_time - float(row["ended_at"]) <= 12:
            speakers = json.loads(row["speaker_ids_json"])
            if (
                speakers
                and speakers[-1] == message.speaker_id
                and not message.reply_message_id
                and not message.generated_for_message_ids
            ):
                return int(row["utterance_id"]), int(row["id"])
        return int(time.time_ns() & 0x7FFFFFFFFFFFFFFF), 0

    def _embedding(self, text: str) -> tuple[bytes | None, int, str]:
        try:
            vector = self.embedding.embed([redact_for_model(text)])[0]
            return pack_vector(vector), len(vector), self.embedding.name
        except Exception as exc:
            logger.error("Chat memory embedding failed: %s %s", type(exc).__name__, repr(exc))
            return None, 0, ""

    def _create_chunk(self, connection, utterance_id: int, topic_id: int, message: MemoryMessage) -> None:
        search_text = " ".join(lexical_terms(message.text))
        blob, dimensions, provider = self._embedding(message.text)
        cursor = connection.execute(
            """INSERT INTO memory_chunks
               (group_id,utterance_id,topic_id,text,speaker_ids_json,message_ids_json,
                started_at,ended_at,search_text,embedding,dimensions,embedding_provider)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                message.group_id, utterance_id, topic_id, message.text,
                json.dumps([message.speaker_id]), json.dumps([message.message_id]),
                message.event_time, message.event_time, search_text, blob, dimensions, provider,
            ),
        )
        connection.execute(
            "INSERT INTO memory_chunks_fts(chunk_id,group_id,search_text) VALUES(?,?,?)",
            (int(cursor.lastrowid), message.group_id, search_text),
        )

    def _append_to_chunk(self, connection, chunk_id: int, message: MemoryMessage) -> None:
        row = connection.execute("SELECT * FROM memory_chunks WHERE id=?", (chunk_id,)).fetchone()
        text = str(row["text"]) + "\n" + message.text
        message_ids = json.loads(row["message_ids_json"])
        message_ids.append(message.message_id)
        search_text = " ".join(lexical_terms(text))
        blob, dimensions, provider = self._embedding(text)
        connection.execute(
            """UPDATE memory_chunks SET text=?,message_ids_json=?,ended_at=?,search_text=?,
               embedding=?,dimensions=?,embedding_provider=? WHERE id=?""",
            (text, json.dumps(message_ids), message.event_time, search_text, blob, dimensions, provider, chunk_id),
        )
        connection.execute("DELETE FROM memory_chunks_fts WHERE chunk_id=?", (chunk_id,))
        connection.execute(
            "INSERT INTO memory_chunks_fts(chunk_id,group_id,search_text) VALUES(?,?,?)",
            (chunk_id, message.group_id, search_text),
        )

    def recall(self, group_id: int, message_id: str) -> bool:
        found = False
        with self._write_lock, self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM chat_messages WHERE group_id=? AND message_id=?",
                (group_id, str(message_id)),
            ).fetchone()
            if not row:
                return False
            found = True
            connection.execute(
                "UPDATE chat_messages SET recalled=1,searchable=0,message_status='recalled' WHERE id=?",
                (row["id"],),
            )
            connection.execute(
                "DELETE FROM message_relations WHERE group_id=? "
                "AND (source_message_id=? OR target_message_id=?)",
                (group_id, str(message_id), str(message_id)),
            )
            connection.execute(
                "DELETE FROM message_topic_relations WHERE group_id=? AND message_id=?",
                (group_id, str(message_id)),
            )
            chunk_rows = connection.execute(
                "SELECT id,message_ids_json FROM memory_chunks WHERE group_id=? AND active=1",
                (group_id,),
            ).fetchall()
            for chunk in chunk_rows:
                if str(message_id) in json.loads(chunk["message_ids_json"]):
                    connection.execute("UPDATE memory_chunks SET active=0 WHERE id=?", (chunk["id"],))
                    connection.execute("DELETE FROM memory_chunks_fts WHERE chunk_id=?", (chunk["id"],))
        if found:
            self.rebuild_group(group_id)
        return found

    def lexical_probe(
        self,
        *,
        group_id: int,
        query: str,
        exclude_message_id: str = "",
        limit: int = 3,
        max_chars: int = 1600,
    ) -> tuple[MemoryHit, ...]:
        """Cheap local recall used before deciding whether full retrieval is worthwhile."""
        query_terms = tuple(term for term in lexical_terms(query) if len(term) >= 2)
        if len(query_terms) < 2:
            return ()
        match = " OR ".join(f'"{term}"' for term in query_terms[:40])
        with self.connect() as connection:
            try:
                rows = connection.execute(
                    """SELECT c.*, bm25(memory_chunks_fts) AS rank FROM memory_chunks_fts
                       JOIN memory_chunks c ON c.id=memory_chunks_fts.chunk_id
                       WHERE memory_chunks_fts MATCH ? AND c.group_id=? AND c.active=1
                       ORDER BY rank LIMIT 40""",
                    (match, group_id),
                ).fetchall()
            except sqlite3.OperationalError:
                return ()

            now = time.time()
            hits: list[MemoryHit] = []
            query_term_set = set(query_terms)
            for position, row in enumerate(rows):
                message_ids = tuple(json.loads(row["message_ids_json"]))
                if exclude_message_id and exclude_message_id in message_ids:
                    continue
                row_terms = set(str(row["search_text"]).split())
                matched = len(query_term_set.intersection(row_terms))
                coverage = matched / max(1, len(query_term_set))
                # One common bigram is too weak for automatic history injection.
                if matched < 2 or coverage < 0.24:
                    continue
                age_days = max(0.0, (now - float(row["ended_at"])) / 86400)
                recency = 1.0 / (1.0 + age_days / 7.0)
                lexical_score = min(1.0, 0.75 * coverage + 0.25 / (1.0 + position))
                hits.append(MemoryHit(
                    int(row["id"]),
                    int(row["group_id"]),
                    str(row["text"]),
                    tuple(json.loads(row["speaker_ids_json"])),
                    float(row["ended_at"]),
                    int(row["topic_id"]),
                    message_ids,
                    0.85 * lexical_score + 0.15 * recency,
                    ("lexical_probe", "recent"),
                ))
            hits.sort(key=lambda hit: hit.score, reverse=True)
            selected: list[MemoryHit] = []
            used = 0
            for hit in hits:
                size = len(hit.text)
                if selected and used + size > max_chars:
                    continue
                selected.append(hit)
                used += size
                if len(selected) >= max(1, limit) or used >= max_chars:
                    break
            return tuple(selected)

    def retrieve(self, *, group_id: int, query: str, speaker_id: str = "", reply_message_id: str = "", exclude_message_id: str = "", topic_id: int = 0, participant_scope: str = "group", time_scope: str = "", limit: int = 6, max_chars: int = 2400) -> tuple[MemoryHit, ...]:
        if not query.strip() and not reply_message_id:
            return ()
        query_terms = lexical_terms(query)
        query_vector: Sequence[float] = ()
        try:
            query_vector = self.embedding.embed([redact_for_model(query)])[0]
        except Exception:
            pass
        with self.connect() as connection:
            candidates: dict[int, dict] = {}
            reply_ids: set[str] = set()
            current_reply = reply_message_id
            for _ in range(8):
                if not current_reply or current_reply in reply_ids:
                    break
                reply_ids.add(current_reply)
                parent = connection.execute(
                    "SELECT reply_message_id FROM chat_messages WHERE group_id=? AND message_id=? AND recalled=0",
                    (group_id, current_reply),
                ).fetchone()
                current_reply = str(parent["reply_message_id"]) if parent else ""
            if query_terms:
                match = " OR ".join(f'"{term}"' for term in query_terms[:40])
                try:
                    rows = connection.execute(
                        """SELECT c.*, bm25(memory_chunks_fts) AS rank FROM memory_chunks_fts
                           JOIN memory_chunks c ON c.id=memory_chunks_fts.chunk_id
                           WHERE memory_chunks_fts MATCH ? AND c.group_id=? AND c.active=1
                           ORDER BY rank LIMIT 60""",
                        (match, group_id),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                for position, row in enumerate(rows):
                    candidates[int(row["id"])] = {"row": row, "lexical": 1.0 / (1.0 + position)}
            rows = connection.execute(
                "SELECT * FROM memory_chunks WHERE group_id=? AND active=1 ORDER BY ended_at DESC LIMIT 2000",
                (group_id,),
            ).fetchall()
            for row in rows:
                vector = unpack_vector(row["embedding"], int(row["dimensions"]))
                vector_score = max(0.0, cosine(query_vector, vector)) if query_vector else 0.0
                row_message_ids = set(json.loads(row["message_ids_json"]))
                if vector_score >= 0.12 or int(row["id"]) in candidates or reply_ids.intersection(row_message_ids):
                    candidates.setdefault(int(row["id"]), {"row": row, "lexical": 0.0})["vector"] = vector_score
            now = time.time()
            scope_seconds = {"day": 86400, "week": 7 * 86400, "month": 31 * 86400}.get(time_scope, 0)
            hits: list[MemoryHit] = []
            for candidate in candidates.values():
                row = candidate["row"]
                message_ids = tuple(json.loads(row["message_ids_json"]))
                if exclude_message_id and exclude_message_id in message_ids:
                    continue
                speakers = tuple(json.loads(row["speaker_ids_json"]))
                reply_bonus = 1.0 if reply_ids.intersection(message_ids) else 0.0
                if scope_seconds and now - float(row["ended_at"]) > scope_seconds and not reply_bonus:
                    continue
                if participant_scope == "current" and speaker_id not in speakers and not reply_bonus:
                    continue
                if participant_scope == "reply_chain" and not reply_bonus:
                    continue
                same_topic = 1.0 if topic_id and int(row["topic_id"]) == topic_id else 0.0
                same_speaker = 1.0 if speaker_id and speaker_id in speakers else 0.0
                age_days = max(0.0, (now - float(row["ended_at"])) / 86400)
                recency = 1.0 / (1.0 + age_days / 7.0)
                score = 0.40 * candidate.get("vector", 0.0) + 0.20 * candidate["lexical"] + 0.15 * reply_bonus + 0.10 * same_topic + 0.05 * same_speaker + 0.10 * recency
                reasons = tuple(name for name, value in (("semantic", candidate.get("vector", 0)), ("lexical", candidate["lexical"]), ("reply_chain", reply_bonus), ("same_topic", same_topic), ("same_participant", same_speaker), ("recent", recency)) if value)
                hits.append(MemoryHit(int(row["id"]), int(row["group_id"]), str(row["text"]), speakers, float(row["ended_at"]), int(row["topic_id"]), message_ids, score, reasons))
            hits.sort(key=lambda hit: hit.score, reverse=True)
            selected: list[MemoryHit] = []
            used = 0
            seen: set[str] = set()
            for hit in hits:
                normalized = " ".join(hit.text.split())
                if normalized in seen or (not set(hit.message_ids).intersection(reply_ids) and hit.score < 0.12):
                    continue
                if selected and used + len(hit.text) > max_chars:
                    continue
                selected.append(hit)
                seen.add(normalized)
                used += len(hit.text)
                if len(selected) >= max(1, limit):
                    break
            return tuple(selected)

    def format_hits(self, hits: Sequence[MemoryHit]) -> tuple[str, ...]:
        lines: list[str] = []
        with self.connect() as connection:
            for hit in hits:
                messages: list[dict] = []
                for message_id in hit.message_ids:
                    row = connection.execute(
                        """SELECT message_id,speaker_id,speaker_role,text,event_time,reply_message_id,
                                  reply_speaker_id,quoted_text,mentions_json,
                                  generated_for_message_ids_json,turn_id,reply_mode,semantic_topic,
                                  sequence,received_time,content_segments_json,message_status
                           FROM chat_messages
                           WHERE group_id=? AND message_id=? AND recalled=0 AND searchable=1""",
                        (hit.group_id, message_id),
                    ).fetchone()
                    if not row:
                        continue
                    reply_to = None
                    if row["reply_message_id"]:
                        reply_to = {
                            "message_id": str(row["reply_message_id"]),
                            "speaker_id": str(row["reply_speaker_id"] or "unknown_member"),
                            "speaker_role": (
                                "bot" if str(row["reply_speaker_id"]) == "bot" else "member"
                            ),
                            "quoted_text": redact_for_model(str(row["quoted_text"])),
                        }
                    role = str(row["speaker_role"])
                    topic_rows = connection.execute(
                        """SELECT topic_id,confidence,relation_basis,is_primary,
                                  anchor_message_ids_json
                           FROM message_topic_relations
                           WHERE group_id=? AND message_id=?
                           ORDER BY is_primary DESC,confidence DESC LIMIT 2""",
                        (hit.group_id, message_id),
                    ).fetchall()
                    messages.append({
                        "message_id": str(row["message_id"]),
                        "speaker": {
                            "id": str(row["speaker_id"]),
                            "role": role,
                            "is_self": role == "bot",
                        },
                        "text": redact_for_model(str(row["text"])),
                        "sequence": int(row["sequence"] or 0),
                        "event_time": float(row["event_time"] or 0),
                        "received_time": float(row["received_time"] or 0),
                        "content_segments": list(
                            json.loads(row["content_segments_json"] or "[]")
                        ),
                        "message_status": str(row["message_status"] or "active"),
                        "topic_candidates": [
                            {
                                "topic_id": int(topic_row["topic_id"]),
                                "confidence": round(float(topic_row["confidence"]), 4),
                                "basis": str(topic_row["relation_basis"]),
                                "is_primary": bool(topic_row["is_primary"]),
                                "anchor_message_ids": list(json.loads(
                                    topic_row["anchor_message_ids_json"] or "[]"
                                )),
                            }
                            for topic_row in topic_rows
                        ],
                        "reply_to": reply_to,
                        "mentions": list(json.loads(row["mentions_json"] or "[]")),
                        "generated_for_message_ids": list(
                            json.loads(row["generated_for_message_ids_json"] or "[]")
                        ),
                        "turn_id": str(row["turn_id"] or ""),
                        "reply_mode": str(row["reply_mode"] or ""),
                        "semantic_topic": redact_for_model(str(row["semantic_topic"] or "")),
                    })
                lines.append(json.dumps({
                    "source": "untrusted_group_chat_memory",
                    "chunk_id": hit.chunk_id,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hit.event_time)),
                    "speakers": list(hit.speaker_ids),
                    "messages": messages,
                    "reasons": list(hit.reasons),
                }, ensure_ascii=False, separators=(",", ":")))
        return tuple(lines)

    def format_self_history(
        self,
        *,
        group_id: int,
        related_message_ids: Sequence[str] = (),
        topic_ids: Sequence[int] = (),
        limit: int = 3,
        max_chars: int = 1200,
    ) -> tuple[str, ...]:
        """Load exact bot turns for selected messages/topics without treating them as facts."""
        message_ids = tuple(dict.fromkeys(
            str(value or "").strip() for value in related_message_ids if str(value or "").strip()
        ))
        selected_topics = {int(value) for value in topic_ids if int(value) > 0}
        direct_bot_ids: set[str] = set()
        with self.connect() as connection:
            if message_ids:
                placeholders = ",".join("?" for _ in message_ids)
                rows = connection.execute(
                    f"""SELECT message_id,speaker_role,topic_id,reply_message_id
                        FROM chat_messages
                        WHERE group_id=? AND recalled=0 AND message_id IN ({placeholders})""",
                    (group_id, *message_ids),
                ).fetchall()
                for row in rows:
                    if row["topic_id"]:
                        selected_topics.add(int(row["topic_id"]))
                    if str(row["speaker_role"]) == "bot":
                        direct_bot_ids.add(str(row["message_id"]))
                    replied_id = str(row["reply_message_id"] or "")
                    if replied_id:
                        replied = connection.execute(
                            "SELECT speaker_role FROM chat_messages WHERE group_id=? AND message_id=?",
                            (group_id, replied_id),
                        ).fetchone()
                        if replied and str(replied["speaker_role"]) == "bot":
                            direct_bot_ids.add(replied_id)
                relation_rows = connection.execute(
                    f"""SELECT target_message_id FROM message_relations
                        WHERE group_id=? AND source_message_id IN ({placeholders})
                          AND relation_type='feedback_to'""",
                    (group_id, *message_ids),
                ).fetchall()
                direct_bot_ids.update(str(row["target_message_id"]) for row in relation_rows)

            conditions = []
            parameters: list[object] = [group_id]
            if direct_bot_ids:
                placeholders = ",".join("?" for _ in direct_bot_ids)
                conditions.append(f"message_id IN ({placeholders})")
                parameters.extend(sorted(direct_bot_ids))
            if selected_topics:
                placeholders = ",".join("?" for _ in selected_topics)
                conditions.append(f"topic_id IN ({placeholders})")
                parameters.extend(sorted(selected_topics))
            if not conditions:
                return ()
            rows = connection.execute(
                f"""SELECT * FROM chat_messages
                    WHERE group_id=? AND speaker_role='bot' AND recalled=0 AND searchable=1
                      AND ({' OR '.join(conditions)})
                    ORDER BY event_time DESC,id DESC LIMIT ?""",
                (*parameters, max(1, int(limit) * 3)),
            ).fetchall()

            output: list[str] = []
            used = 0
            for row in rows:
                generated_for = tuple(json.loads(row["generated_for_message_ids_json"] or "[]"))
                triggers = []
                for target_id in generated_for:
                    trigger = connection.execute(
                        """SELECT message_id,speaker_id,speaker_role,text,event_time
                           FROM chat_messages WHERE group_id=? AND message_id=? AND recalled=0""",
                        (group_id, str(target_id)),
                    ).fetchone()
                    if trigger:
                        triggers.append({
                            "message_id": str(trigger["message_id"]),
                            "speaker": {
                                "id": str(trigger["speaker_id"]),
                                "role": str(trigger["speaker_role"]),
                                "is_self": str(trigger["speaker_role"]) == "bot",
                            },
                            "text": redact_for_model(str(trigger["text"])),
                        })
                feedback_rows = connection.execute(
                    """SELECT DISTINCT m.message_id,m.speaker_id,m.speaker_role,m.text,m.event_time
                       FROM message_relations r
                       JOIN chat_messages m
                         ON m.group_id=r.group_id AND m.message_id=r.source_message_id
                       WHERE r.group_id=? AND r.target_message_id=?
                         AND r.relation_type='feedback_to' AND m.recalled=0
                       ORDER BY m.event_time ASC LIMIT 4""",
                    (group_id, str(row["message_id"])),
                ).fetchall()
                feedback = [{
                    "message_id": str(item["message_id"]),
                    "speaker": {
                        "id": str(item["speaker_id"]),
                        "role": str(item["speaker_role"]),
                        "is_self": str(item["speaker_role"]) == "bot",
                    },
                    "text": redact_for_model(str(item["text"])),
                } for item in feedback_rows]
                payload = {
                    "source": "bot_self_history",
                    "authorship": "authoritative",
                    "fact_status": "untrusted_previous_output",
                    "topic_id": int(row["topic_id"] or 0),
                    "turn_id": str(row["turn_id"] or ""),
                    "reply_mode": str(row["reply_mode"] or ""),
                    "semantic_topic": redact_for_model(str(row["semantic_topic"] or "")),
                    "trigger_messages": triggers,
                    "bot_message": {
                        "message_id": str(row["message_id"]),
                        "speaker": {"id": "bot", "role": "bot", "is_self": True},
                        "text": redact_for_model(str(row["text"])),
                        "generated_for_message_ids": list(generated_for),
                    },
                    "feedback_messages": feedback,
                }
                line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                if output and used + len(line) > max(200, int(max_chars)):
                    continue
                output.append(line)
                used += len(line)
                if len(output) >= max(1, int(limit)) or used >= max_chars:
                    break
            return tuple(output)

    def status(self) -> dict[str, int | str]:
        with self.connect() as connection:
            messages = connection.execute("SELECT count(*) FROM chat_messages WHERE recalled=0").fetchone()[0]
            chunks = connection.execute("SELECT count(*) FROM memory_chunks WHERE active=1").fetchone()[0]
            topic_relations = connection.execute(
                "SELECT count(*) FROM message_topic_relations"
            ).fetchone()[0]
        return {
            "messages": int(messages),
            "chunks": int(chunks),
            "topic_relations": int(topic_relations),
            "provider": self.embedding.name,
        }

    def clear_group(self, group_id: int) -> None:
        with self._write_lock, self.connect() as connection:
            chunk_ids = [row[0] for row in connection.execute("SELECT id FROM memory_chunks WHERE group_id=?", (group_id,))]
            for chunk_id in chunk_ids:
                connection.execute("DELETE FROM memory_chunks_fts WHERE chunk_id=?", (chunk_id,))
            connection.execute("DELETE FROM memory_chunks WHERE group_id=?", (group_id,))
            connection.execute("DELETE FROM chat_messages WHERE group_id=?", (group_id,))
            connection.execute("DELETE FROM topic_sessions WHERE group_id=?", (group_id,))
            connection.execute("DELETE FROM message_relations WHERE group_id=?", (group_id,))
            connection.execute("DELETE FROM message_topic_relations WHERE group_id=?", (group_id,))

    def rebuild_group(self, group_id: int) -> int:
        with self._write_lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE group_id=? AND recalled=0 AND searchable=1 ORDER BY event_time,id",
                (group_id,),
            ).fetchall()
            chunk_ids = [row[0] for row in connection.execute("SELECT id FROM memory_chunks WHERE group_id=?", (group_id,))]
            for chunk_id in chunk_ids:
                connection.execute("DELETE FROM memory_chunks_fts WHERE chunk_id=?", (chunk_id,))
            connection.execute("DELETE FROM memory_chunks WHERE group_id=?", (group_id,))
            connection.execute("DELETE FROM topic_sessions WHERE group_id=?", (group_id,))
            connection.execute("DELETE FROM message_topic_relations WHERE group_id=?", (group_id,))
            for row in rows:
                message = MemoryMessage(
                    group_id=int(row["group_id"]), message_id=str(row["message_id"]),
                    speaker_id=str(row["speaker_id"]), display_name=str(row["display_name"]),
                    speaker_role=str(row["speaker_role"]), text=str(row["text"]),
                    event_time=float(row["event_time"]), reply_message_id=str(row["reply_message_id"]),
                    reply_speaker_id=str(row["reply_speaker_id"]), quoted_text=str(row["quoted_text"]),
                    mentions=tuple(json.loads(row["mentions_json"] or "[]")),
                    generated_for_message_ids=tuple(
                        json.loads(row["generated_for_message_ids_json"] or "[]")
                    ),
                    turn_id=str(row["turn_id"] or ""),
                    reply_mode=str(row["reply_mode"] or ""),
                    semantic_topic=str(row["semantic_topic"] or ""),
                    sequence=int(row["sequence"] or 0),
                    received_time=float(row["received_time"] or row["event_time"] or 0),
                    content_segments=tuple(
                        json.loads(row["content_segments_json"] or "[]")
                    ),
                    message_status=str(row["message_status"] or "active"),
                )
                assignments = self._choose_topic_assignments(connection, message)
                topic_id = assignments[0].topic_id
                utterance_id, chunk_id = self._choose_utterance(connection, message, topic_id)
                connection.execute(
                    "UPDATE chat_messages SET utterance_id=?,topic_id=? WHERE id=?",
                    (utterance_id, topic_id, row["id"]),
                )
                self._record_topic_assignments(connection, message, assignments)
                if chunk_id:
                    self._append_to_chunk(connection, chunk_id, message)
                else:
                    self._create_chunk(connection, utterance_id, topic_id, message)
            return len(rows)

    def cleanup(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        with self._write_lock, self.connect() as connection:
            rows = connection.execute("SELECT id FROM memory_chunks WHERE ended_at<?", (cutoff,)).fetchall()
            for row in rows:
                connection.execute("DELETE FROM memory_chunks_fts WHERE chunk_id=?", (row["id"],))
            connection.execute("DELETE FROM memory_chunks WHERE ended_at<?", (cutoff,))
            cursor = connection.execute("DELETE FROM chat_messages WHERE event_time<?", (cutoff,))
            connection.execute("DELETE FROM topic_sessions WHERE updated_at<?", (cutoff,))
            connection.execute("DELETE FROM message_relations WHERE created_at<?", (cutoff,))
            return cursor.rowcount


class ChatMemoryManager:
    def __init__(self, store: ChatMemoryStore, retention_days: int = 90):
        self.store = store
        self.retention_days = retention_days
        self.jobs: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=10000)
        self.paused = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._run, daemon=True, name="chat-memory-indexer").start()

    def enqueue(self, message: MemoryMessage) -> bool:
        try:
            self.jobs.put_nowait(("message", message))
            return True
        except queue.Full:
            logger.warning("Chat memory queue full; message not indexed %s %s", message.group_id, message.message_id)
            return False

    def enqueue_recall(self, group_id: int, message_id: str) -> bool:
        try:
            self.jobs.put_nowait(("recall", (group_id, message_id)))
            return True
        except queue.Full:
            return False

    def enqueue_rebuild(self, group_id: int) -> bool:
        try:
            self.jobs.put_nowait(("rebuild", int(group_id)))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        last_cleanup = 0.0
        while True:
            kind, payload = self.jobs.get()
            try:
                while self.paused.is_set():
                    time.sleep(0.25)
                if kind == "message":
                    self.store.add_message(payload)
                elif kind == "recall":
                    self.store.recall(*payload)
                elif kind == "rebuild":
                    self.store.rebuild_group(int(payload))
                if time.time() - last_cleanup > 3600:
                    self.store.cleanup(self.retention_days)
                    last_cleanup = time.time()
            except Exception as exc:
                logger.error("Chat memory job failed: %s %s", type(exc).__name__, repr(exc))
            finally:
                self.jobs.task_done()

    def status(self) -> dict:
        return {**self.store.status(), "queued": self.jobs.qsize(), "paused": self.paused.is_set()}
