from __future__ import annotations

import sqlite3
from pathlib import Path

from ..models import ConversationState, PendingFailureResult


def _pending_db_path(deps, db_path: str | Path | None = None) -> str | Path:
    return db_path or deps.settings.pending_queue_db


def open_pending_queue_db(
    deps, db_path: str | Path | None = None
) -> sqlite3.Connection:
    return deps.pending_store.open_pending_queue_db(_pending_db_path(deps, db_path))


def persist_conversation_turn(
    deps, group_id: int, state: ConversationState, *, db_path: str | Path | None = None
) -> int:
    return deps.pending_store.persist_conversation_turn(
        group_id, state, db_path=_pending_db_path(deps, db_path)
    )


def _conversation_state_from_row(deps, row) -> ConversationState:
    return deps.pending_store.conversation_state_from_row(row)


def load_conversation_turn_by_bot_message_id(
    deps, group_id: int, bot_message_id: str, *, db_path: str | Path | None = None
) -> ConversationState | None:
    return deps.pending_store.load_conversation_turn_by_bot_message_id(
        group_id, bot_message_id, db_path=_pending_db_path(deps, db_path)
    )


def persist_pending_message(
    deps, priority: int, sequence: int, item: dict, db_path: str | Path | None = None
) -> int:
    return deps.pending_store.persist_pending_message(
        priority, sequence, item, db_path=_pending_db_path(deps, db_path)
    )


def load_pending_messages(
    deps, db_path: str | Path | None = None, *, include_future: bool = False
) -> list[tuple[int, int, dict]]:
    return deps.pending_store.load_pending_messages(
        db_path=_pending_db_path(deps, db_path), include_future=include_future
    )


def mark_pending_failure(
    deps,
    pending_id: int,
    error: str,
    *,
    db_path: str | Path | None = None,
    now: float | None = None,
    max_attempts: int | None = None,
) -> PendingFailureResult:
    retry_limit = (
        max_attempts
        if max_attempts is not None
        else getattr(deps.settings, "pending_retry_max_attempts", 3)
    )
    return deps.pending_store.mark_pending_failure(
        pending_id,
        error,
        db_path=_pending_db_path(deps, db_path),
        now=now,
        max_attempts=retry_limit,
    )


def mark_pending_sent_unknown(
    deps, pending_id: int, error: str, *, db_path: str | Path | None = None
) -> None:
    deps.pending_store.mark_pending_sent_unknown(
        pending_id, error, db_path=_pending_db_path(deps, db_path)
    )


def mark_pending_dispatch_started(
    deps, pending_id: int, dispatch_id: str, *, db_path: str | Path | None = None
) -> None:
    deps.pending_store.mark_pending_dispatch_started(
        pending_id, dispatch_id, db_path=_pending_db_path(deps, db_path)
    )


def recover_incomplete_pending_dispatches(
    deps, db_path: str | Path | None = None
) -> int:
    return deps.pending_store.recover_incomplete_pending_dispatches(
        db_path=_pending_db_path(deps, db_path)
    )


def delete_pending_message(
    deps, pending_id: int, db_path: str | Path | None = None
) -> None:
    deps.pending_store.delete_pending_message(
        pending_id, db_path=_pending_db_path(deps, db_path)
    )


def pending_message_count(deps, db_path: str | Path | None = None) -> int:
    return deps.pending_store.pending_message_count(
        db_path=_pending_db_path(deps, db_path)
    )


def pending_status_counts(deps, db_path: str | Path | None = None) -> dict[str, int]:
    return deps.pending_store.pending_status_counts(
        db_path=_pending_db_path(deps, db_path)
    )


def mark_pending_superseded(
    deps, pending_id: int, *, db_path: str | Path | None = None
) -> None:
    deps.pending_store.mark_pending_superseded(
        pending_id, db_path=_pending_db_path(deps, db_path)
    )


def is_pending_superseded(
    deps, pending_id: int, *, db_path: str | Path | None = None
) -> bool:
    return deps.pending_store.is_pending_superseded(
        pending_id, db_path=_pending_db_path(deps, db_path)
    )


def find_queued_duplicates(
    deps, group_id: int, message_ids: list[str], *, db_path: str | Path | None = None
) -> list[int]:
    return deps.pending_store.find_queued_duplicates(
        group_id, message_ids, db_path=_pending_db_path(deps, db_path)
    )
