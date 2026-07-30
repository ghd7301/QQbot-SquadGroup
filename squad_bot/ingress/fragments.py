from __future__ import annotations

import logging
import time
from typing import Sequence

from ..models import MessageFragmentBuffer

logger = logging.getLogger(__name__)


def classify_fragment_audience(deps, item: dict) -> str:
    """Classify who a fragment addresses without guessing from general pronouns."""
    return deps.classify_audience(item, bot_user_id=str(deps.settings.bot_qq or ""))


def fragment_items_compatible(
    deps, buffer: MessageFragmentBuffer, item: dict, audience: str
) -> bool:
    return deps.items_compatible(buffer, item, audience)


def _new_fragment_buffer(
    deps, item: dict, audience: str, now: float
) -> MessageFragmentBuffer:
    return deps.fragment_aggregator.new_buffer(
        item,
        audience,
        now,
        debounce_seconds=deps.settings.message_fragment_debounce_seconds,
        max_wait_seconds=deps.settings.message_fragment_max_wait_seconds,
    )


def _merge_fragment(
    deps, buffer: MessageFragmentBuffer, item: dict, audience: str, now: float
) -> None:
    deps.fragment_aggregator.merge(
        buffer,
        item,
        audience,
        now,
        debounce_seconds=deps.settings.message_fragment_debounce_seconds,
        max_wait_seconds=deps.settings.message_fragment_max_wait_seconds,
    )


def _fragment_prefix_item(deps, buffer: MessageFragmentBuffer, count: int) -> dict:
    return deps.fragment_aggregator.prefix_item(buffer, count)


def semantic_bot_fragment_count(deps, buffer: MessageFragmentBuffer) -> int:
    if buffer.audience != "bot" or len(buffer.fragments) < 2:
        return len(buffer.fragments)
    if not any(
        (
            fragment.get("fragment_audience") == "unknown"
            for fragment in buffer.fragments[1:]
        )
    ):
        return len(buffer.fragments)
    if not getattr(deps.settings, "message_fragment_semantic_enabled", True):
        return 1
    decision = deps.classify_bot_fragment_prefix(
        base_url=deps.settings.llm_base_url,
        api_key=deps.settings.llm_api_key,
        model=getattr(
            deps.settings, "message_fragment_semantic_model", deps.settings.llm_model
        ),
        fragments=buffer.parts,
        context=tuple(buffer.item.get("chat_context") or ()),
        timeout=getattr(deps.settings, "message_fragment_semantic_timeout_seconds", 8),
    )
    if not decision:
        return 1
    (count, confidence) = decision
    minimum = getattr(deps.settings, "message_fragment_semantic_min_confidence", 0.75)
    return count if confidence >= minimum else 1


def _dispatch_fragment_buffer(deps, buffer: MessageFragmentBuffer) -> int:
    count = deps.semantic_bot_fragment_count(buffer)
    item = deps._fragment_prefix_item(buffer, count)
    if count < len(buffer.fragments):
        logger.info(
            "Fragment audience split %s %s bot=%s/%s",
            buffer.group_id,
            buffer.user_id,
            count,
            len(buffer.fragments),
        )
    priority = 0 if item.get("mentioned") else 1
    return deps.enqueue_persistent_message(priority, item)


def _defer_fragment_buffers(deps, buffers: Sequence[MessageFragmentBuffer]) -> None:
    deps.fragment_aggregator.defer(buffers)


def clear_fragment_state(deps) -> None:
    deps.fragment_aggregator.clear()


def flush_group_fragment_buffer(
    deps, group_id: int, *, defer_dispatch: bool = False
) -> int | None:
    buffer = deps.fragment_aggregator.pop_group(group_id)
    if not buffer:
        return None
    if defer_dispatch:
        deps._defer_fragment_buffers((buffer,))
        return None
    return deps._dispatch_fragment_buffer(buffer)


def flush_fragment_buffer_for_new_speaker(
    deps, group_id: int, user_id, *, defer_dispatch: bool = False
) -> int | None:
    buffer = deps.fragment_aggregator.pop_for_new_speaker(group_id, user_id)
    if not buffer:
        return None
    if defer_dispatch:
        deps._defer_fragment_buffers((buffer,))
        return None
    return deps._dispatch_fragment_buffer(buffer)


def submit_message_fragment(
    deps, item: dict, *, now: float | None = None, defer_dispatch: bool = False
) -> list[int]:
    """Buffer a fragment, optionally deferring semantic dispatch to the worker."""
    current_time = time.monotonic() if now is None else float(now)
    audience = deps.classify_fragment_audience(item)
    is_admin_command = bool(
        deps.is_admin_user(item.get("user_id"), item.get("sender_role", ""))
        and deps.get_admin_command(str(item.get("question") or ""))
    )
    displaced = deps.fragment_aggregator.submit(
        item,
        audience,
        now=current_time,
        is_immediate=is_admin_command,
        max_parts=deps.settings.message_fragment_max_parts,
        max_chars=deps.settings.message_fragment_max_chars,
        debounce_seconds=deps.settings.message_fragment_debounce_seconds,
        max_wait_seconds=deps.settings.message_fragment_max_wait_seconds,
    )
    pending_ids: list[int] = []
    if defer_dispatch:
        deps._defer_fragment_buffers(displaced)
    else:
        pending_ids = [deps._dispatch_fragment_buffer(buffer) for buffer in displaced]
    if is_admin_command:
        pending_ids.append(
            deps.enqueue_persistent_message(0 if item.get("mentioned") else 1, item)
        )
    return pending_ids


def fragment_aggregation_worker(deps) -> None:
    while True:
        due = deps.fragment_aggregator.wait_for_due()
        for buffer in due:
            try:
                deps._dispatch_fragment_buffer(buffer)
            except Exception as exc:
                logger.error("Fragment queue write failed: %s", repr(exc))
