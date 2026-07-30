from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .models import ConversationState, PendingFailureResult


def open_pending_queue_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            priority INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            payload TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    existing_pending_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(pending_messages)")
    }
    pending_migrations = {
        "status": "TEXT NOT NULL DEFAULT 'queued'",
        "attempts": "INTEGER NOT NULL DEFAULT 0",
        "next_attempt_at": "REAL NOT NULL DEFAULT 0",
        "last_error": "TEXT NOT NULL DEFAULT ''",
        "dispatch_id": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in pending_migrations.items():
        if column not in existing_pending_columns:
            connection.execute(
                f"ALTER TABLE pending_messages ADD COLUMN {column} {definition}"
            )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_status_time "
        "ON pending_messages (status, next_attempt_at, priority, sequence)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_reply_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            replied_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_reply_group_time "
        "ON chat_reply_history (group_id, replied_at)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS celebration_reply_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            target_key TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            replied_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_celebration_reply_lookup "
        "ON celebration_reply_history (group_id, target_key, event_kind, replied_at)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_message_id TEXT NOT NULL,
            bot_message_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            reply_mode TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_bot_message "
        "ON conversation_turns (group_id, bot_message_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_user_time "
        "ON conversation_turns (group_id, user_id, created_at)"
    )
    existing_turn_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(conversation_turns)")
    }
    turn_migrations = {
        "trigger_message_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "turn_id": "TEXT NOT NULL DEFAULT ''",
        "semantic_intent": "TEXT NOT NULL DEFAULT ''",
        "semantic_topic": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in turn_migrations.items():
        if column not in existing_turn_columns:
            connection.execute(
                f"ALTER TABLE conversation_turns ADD COLUMN {column} {definition}"
            )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_turn_id "
        "ON conversation_turns (group_id, turn_id)"
    )
    connection.commit()
    return connection


def persist_conversation_turn(
    group_id: int,
    state: ConversationState,
    *,
    db_path: str | Path,
) -> int:
    connection = open_pending_queue_db(db_path)
    try:
        cutoff = time.time() - 30 * 86400
        connection.execute("DELETE FROM conversation_turns WHERE created_at < ?", (cutoff,))
        cursor = connection.execute(
            """
            INSERT INTO conversation_turns (
                group_id, user_id, user_message_id, bot_message_id,
                question, answer, reply_mode, sources_json, created_at,
                trigger_message_ids_json,turn_id,semantic_intent,semantic_topic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(group_id),
                state.user_id,
                state.user_message_id,
                state.bot_message_id,
                state.last_question,
                state.last_answer,
                state.reply_mode,
                json.dumps(list(state.sources), ensure_ascii=False),
                state.timestamp,
                json.dumps(list(state.trigger_message_ids), ensure_ascii=False),
                state.turn_id,
                state.semantic_intent,
                state.semantic_topic,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def conversation_state_from_row(row) -> ConversationState:
    try:
        sources = tuple(json.loads(row[7]))
    except (TypeError, ValueError, json.JSONDecodeError):
        sources = ()
    return ConversationState(
        last_question=str(row[4] or ""),
        sources=sources,
        timestamp=float(row[8]),
        user_id=str(row[1] or ""),
        last_answer=str(row[5] or ""),
        reply_mode=str(row[6] or ""),
        bot_message_id=str(row[3] or ""),
        user_message_id=str(row[2] or ""),
        trigger_message_ids=tuple(json.loads(row[9] or "[]")),
        turn_id=str(row[10] or ""),
        semantic_intent=str(row[11] or ""),
        semantic_topic=str(row[12] or ""),
    )


def load_conversation_turn_by_bot_message_id(
    group_id: int,
    bot_message_id: str,
    *,
    db_path: str | Path,
) -> ConversationState | None:
    target = str(bot_message_id or "").strip()
    if not target:
        return None
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute(
            """
            SELECT id, user_id, user_message_id, bot_message_id,
                   question, answer, reply_mode, sources_json, created_at,
                   trigger_message_ids_json,turn_id,semantic_intent,semantic_topic
            FROM conversation_turns
            WHERE group_id = ? AND bot_message_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (str(group_id), target),
        ).fetchone()
    finally:
        connection.close()
    return conversation_state_from_row(row) if row else None


def persist_pending_message(
    priority: int,
    sequence: int,
    item: dict,
    *,
    db_path: str | Path,
) -> int:
    connection = open_pending_queue_db(db_path)
    try:
        cursor = connection.execute(
            "INSERT INTO pending_messages (priority, sequence, payload, created_at) VALUES (?, ?, ?, ?)",
            (priority, sequence, json.dumps(item, ensure_ascii=False), time.time()),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def load_pending_messages(
    *,
    db_path: str | Path,
    include_future: bool = False,
) -> list[tuple[int, int, dict]]:
    connection = open_pending_queue_db(db_path)
    try:
        query = """
            SELECT id, priority, sequence, payload, created_at,
                   status, attempts, next_attempt_at, last_error, dispatch_id
            FROM pending_messages
            WHERE status IN ('queued', 'retry')
        """
        parameters: tuple = ()
        if not include_future:
            query += " AND next_attempt_at <= ?"
            parameters = (time.time(),)
        query += " ORDER BY priority, sequence"
        rows = connection.execute(query, parameters).fetchall()
    finally:
        connection.close()

    pending: list[tuple[int, int, dict]] = []
    for (
        pending_id,
        priority,
        sequence,
        payload,
        created_at,
        status,
        attempts,
        next_attempt_at,
        last_error,
        dispatch_id,
    ) in rows:
        try:
            item = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            delete_pending_message(int(pending_id), db_path=db_path)
            continue
        if not isinstance(item, dict):
            delete_pending_message(int(pending_id), db_path=db_path)
            continue
        item["_pending_id"] = int(pending_id)
        item["_pending_created_at"] = float(created_at)
        item["_restored"] = True
        item["_pending_status"] = str(status or "queued")
        item["_pending_attempts"] = int(attempts or 0)
        item["_pending_next_attempt_at"] = float(next_attempt_at or 0)
        item["_pending_last_error"] = str(last_error or "")
        item["_pending_dispatch_id"] = str(dispatch_id or "")
        item["_queue_priority"] = int(priority)
        item["_queue_sequence"] = int(sequence)
        pending.append((int(priority), int(sequence), item))
    return pending


def mark_pending_failure(
    pending_id: int,
    error: str,
    *,
    db_path: str | Path,
    max_attempts: int,
    now: float | None = None,
) -> PendingFailureResult:
    current_time = time.time() if now is None else float(now)
    limit = max(1, int(max_attempts))
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute(
            "SELECT attempts FROM pending_messages WHERE id = ?",
            (pending_id,),
        ).fetchone()
        if not row:
            return PendingFailureResult("missing", 0, 0.0)
        attempts = int(row[0] or 0) + 1
        if attempts >= limit:
            status = "dead_letter"
            next_attempt_at = 0.0
        else:
            status = "retry"
            delays = (1.0, 3.0, 10.0)
            delay = delays[min(attempts - 1, len(delays) - 1)]
            next_attempt_at = current_time + delay
        connection.execute(
            """
            UPDATE pending_messages
            SET status = ?, attempts = ?, next_attempt_at = ?, last_error = ?
            WHERE id = ?
            """,
            (status, attempts, next_attempt_at, str(error or "")[:500], pending_id),
        )
        connection.commit()
        return PendingFailureResult(status, attempts, next_attempt_at)
    finally:
        connection.close()


def mark_pending_sent_unknown(
    pending_id: int,
    error: str,
    *,
    db_path: str | Path,
) -> None:
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute(
            """
            UPDATE pending_messages
            SET status = 'sent_unknown', last_error = ?, next_attempt_at = 0
            WHERE id = ?
            """,
            (str(error or "")[:500], pending_id),
        )
        connection.commit()
    finally:
        connection.close()


def mark_pending_dispatch_started(
    pending_id: int,
    dispatch_id: str,
    *,
    db_path: str | Path,
) -> None:
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute(
            """
            UPDATE pending_messages
            SET status = 'dispatching', dispatch_id = ?, last_error = ''
            WHERE id = ? AND status IN ('queued', 'retry')
            """,
            (str(dispatch_id), pending_id),
        )
        connection.commit()
    finally:
        connection.close()


def mark_pending_superseded(
    pending_id: int,
    *,
    db_path: str | Path,
) -> None:
    """Mark a queued or dispatching entry as superseded by a higher-priority duplicate."""
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute(
            """
            UPDATE pending_messages
            SET status = 'superseded', next_attempt_at = 0
            WHERE id = ? AND status IN ('queued', 'retry', 'dispatching')
            """,
            (pending_id,),
        )
        connection.commit()
    finally:
        connection.close()


def is_pending_superseded(
    pending_id: int,
    *,
    db_path: str | Path,
) -> bool:
    """Check if a pending entry has been marked as superseded."""
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute(
            "SELECT status FROM pending_messages WHERE id = ?",
            (pending_id,),
        ).fetchone()
        return bool(row and row[0] == "superseded")
    finally:
        connection.close()


def find_queued_duplicates(
    group_id: int,
    message_ids: list[str],
    *,
    db_path: str | Path,
) -> list[int]:
    """Find pending entries for the same group with overlapping message_ids."""
    if not message_ids:
        return []
    connection = open_pending_queue_db(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, payload FROM pending_messages
            WHERE status IN ('queued', 'retry', 'dispatching')
            """,
        ).fetchall()
    finally:
        connection.close()
    target_ids = set(message_ids)
    result: list[int] = []
    import json as _json
    for row in rows:
        try:
            payload = _json.loads(row[1])
        except (TypeError, ValueError):
            continue
        if int(payload.get("group_id") or 0) != group_id:
            continue
        existing_ids = set()
        for candidate in payload.get("message_ids") or ():
            value = str(candidate or "").strip()
            if value:
                existing_ids.add(value)
        msg_id = str(payload.get("message_id") or "").strip()
        if msg_id:
            existing_ids.add(msg_id)
        if existing_ids & target_ids:
            result.append(row[0])
    return result


def recover_incomplete_pending_dispatches(*, db_path: str | Path) -> int:
    connection = open_pending_queue_db(db_path)
    try:
        cursor = connection.execute(
            """
            UPDATE pending_messages
            SET status = 'sent_unknown', next_attempt_at = 0,
                last_error = 'service stopped during message dispatch'
            WHERE status = 'dispatching'
            """
        )
        connection.commit()
        return max(0, int(cursor.rowcount or 0))
    finally:
        connection.close()


def delete_pending_message(pending_id: int, *, db_path: str | Path) -> None:
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute("DELETE FROM pending_messages WHERE id = ?", (pending_id,))
        connection.commit()
    finally:
        connection.close()


def pending_message_count(*, db_path: str | Path) -> int:
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM pending_messages WHERE status IN ('queued', 'retry')"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


def pending_status_counts(*, db_path: str | Path) -> dict[str, int]:
    result = {
        "queued": 0,
        "retry": 0,
        "dispatching": 0,
        "dead_letter": 0,
        "sent_unknown": 0,
    }
    connection = open_pending_queue_db(db_path)
    try:
        for status, count in connection.execute(
            "SELECT status, COUNT(*) FROM pending_messages GROUP BY status"
        ):
            result[str(status or "queued")] = int(count or 0)
    finally:
        connection.close()
    return result


def cleanup_stale_pending_messages(
    *,
    db_path: str | Path,
    max_age_hours: int = 72,
) -> int:
    """Delete dead_letter, sent_unknown, and superseded entries older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    connection = open_pending_queue_db(db_path)
    try:
        cursor = connection.execute(
            """
            DELETE FROM pending_messages
            WHERE status IN ('dead_letter', 'sent_unknown', 'superseded')
              AND created_at < ?
            """,
            (cutoff,),
        )
        connection.commit()
        return cursor.rowcount
    finally:
        connection.close()


def chat_reply_quota_reason(
    group_id: int,
    *,
    db_path: str | Path,
    now: float,
    cooldown_seconds: int,
    max_per_hour: int,
) -> str:
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute(
            "SELECT MAX(replied_at), COUNT(*) FROM chat_reply_history "
            "WHERE group_id = ? AND replied_at > ?",
            (str(group_id), now - 3600),
        ).fetchone()
    finally:
        connection.close()
    previous = float(row[0]) if row and row[0] is not None else None
    recent_count = int(row[1]) if row else 0
    if cooldown_seconds > 0 and previous is not None and now - previous < cooldown_seconds:
        return "chat cooldown"
    if max_per_hour > 0 and recent_count >= max_per_hour:
        return "chat hourly limit"
    return ""


def mark_chat_replied(
    group_id: int,
    *,
    db_path: str | Path,
    now: float,
) -> None:
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute(
            "DELETE FROM chat_reply_history WHERE replied_at <= ?",
            (now - 3600,),
        )
        connection.execute(
            "INSERT INTO chat_reply_history (group_id, replied_at) VALUES (?, ?)",
            (str(group_id), now),
        )
        connection.commit()
    finally:
        connection.close()


def celebration_was_replied(
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    db_path: str | Path,
    now: float,
    window_seconds: int,
) -> bool:
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute(
            "SELECT 1 FROM celebration_reply_history "
            "WHERE group_id = ? AND target_key = ? AND event_kind = ? "
            "AND replied_at > ? LIMIT 1",
            (str(group_id), target_key, event_kind, now - window_seconds),
        ).fetchone()
    finally:
        connection.close()
    return row is not None


def mark_celebration_replied(
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    db_path: str | Path,
    now: float,
) -> None:
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute(
            "DELETE FROM celebration_reply_history WHERE replied_at <= ?",
            (now - 604800,),
        )
        connection.execute(
            "INSERT INTO celebration_reply_history "
            "(group_id, target_key, event_kind, replied_at) VALUES (?, ?, ?, ?)",
            (str(group_id), target_key, event_kind, now),
        )
        connection.commit()
    finally:
        connection.close()
