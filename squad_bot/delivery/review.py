from __future__ import annotations

import re
import time


def locked_send_context_change(
    deps,
    group_id: int,
    reviewed_revision: int,
    item: dict,
) -> tuple[bool, tuple[str, ...], str]:
    """Classify messages that arrived after review without model calls under the send lock."""
    bot_qq = str(deps.settings.bot_qq or "")
    original_user_id = str(item.get("user_id") or "")
    trigger_message_ids = set(deps._message_ids(item))
    direct_to_bot = bool(
        item.get("mentioned")
        or (bot_qq and str(item.get("reply_target_user_id") or "") == bot_qq)
    )
    with deps.chat_history_lock:
        delta = tuple(
            message
            for message in deps.group_chat_history.get(int(group_id), ())
            if message.sequence > reviewed_revision
            and message.user_id != bot_qq
            and message.message_status == "active"
        )
    if not delta:
        return False, (), "unchanged"

    delta_ids = tuple(
        message.message_id or f"sequence:{message.sequence}" for message in delta
    )
    human_thread_by_sender: dict[str, tuple[str, float]] = {}
    continuation_window = max(
        0.0,
        float(getattr(deps.settings, "message_fragment_max_wait_seconds", 8)),
    )

    for message in delta:
        reply_target = str(message.reply_target_user_id or "")
        if message.reply_message_id:
            if not reply_target:
                return True, delta_ids, "new reply target could not be resolved"
            if reply_target == bot_qq or message.reply_message_id in trigger_message_ids:
                return True, delta_ids, "new message directly extends or answers the bot turn"
            human_thread_by_sender[message.user_id] = (
                reply_target,
                message.received_time,
            )
            continue
        if message.mentioned_bot:
            if message.user_id == original_user_id:
                return True, delta_ids, "original sender sent another message to the bot"
            continue
        if message.mentioned_user_ids:
            human_thread_by_sender[message.user_id] = (
                str(message.mentioned_user_ids[0]),
                message.received_time,
            )
            continue
        if message.user_id == original_user_id:
            human_thread = human_thread_by_sender.get(message.user_id)
            if (
                human_thread
                and human_thread[0] != bot_qq
                and 0
                <= message.received_time - human_thread[1]
                <= continuation_window
            ):
                human_thread_by_sender[message.user_id] = (
                    human_thread[0],
                    message.received_time,
                )
                continue
            return True, delta_ids, "original sender added an ambiguous follow-up"
        if not direct_to_bot:
            return True, delta_ids, "unsolicited reply has a newer ambiguous group message"

    return False, delta_ids, "only unrelated directed messages arrived"


def validate_locked_send(
    deps,
    group_id: int,
    item: dict,
    reviewed_revision: int,
    *,
    check_context: bool = True,
) -> tuple[bool, str, str]:
    """Return whether a candidate may send, a blocking reason, and an audit note."""
    if deps.message_already_covered_by_bot(group_id, item.get("message_id")):
        return False, "message covered while waiting to send", ""
    if not check_context:
        return True, "", ""
    invalidated, delta_message_ids, relation = deps.locked_send_context_change(
        group_id,
        reviewed_revision,
        item,
    )
    delta_text = ",".join(delta_message_ids)
    if invalidated:
        return (
            False,
            "context invalidated while waiting for group send lock: "
            f"{relation}; delta={delta_text}",
            "",
        )
    note = (
        f"send-lock context preserved: {relation}; delta={delta_text}"
        if delta_message_ids
        else ""
    )
    return True, "", note


def unsafe_or_repeated_reply(deps, group_id: int, answer: str, *, limit: int = 10) -> str:
    if re.search(r"\bmember_[0-9a-f]{6,}\b", answer, flags=re.I):
        return "internal member id leaked"
    normalized = re.sub(r"[\W_]+", "", answer.lower())
    if len(normalized) < 6:
        return ""
    with deps.chat_history_lock:
        recent_bot_answers = [
            item.text
            for item in deps.group_chat_history.get(group_id, ())
            if item.user_id == deps.settings.bot_qq
        ][-limit:]
    for previous in recent_bot_answers:
        previous_normalized = re.sub(r"[\W_]+", "", previous.lower())
        if normalized == previous_normalized:
            return "duplicate recent bot reply"
    return ""


def is_recent_duplicate_group_message(
    deps,
    group_id: int,
    text: str,
    *,
    focus_sequence: int,
    event_time=None,
    window_seconds: int = 60,
) -> bool:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    if len(normalized) < 4 or focus_sequence <= 0:
        return False
    try:
        current_time = float(event_time)
    except (TypeError, ValueError):
        current_time = time.time()
    with deps.chat_history_lock:
        for item in reversed(deps.group_chat_history.get(group_id, ())):
            if item.sequence >= focus_sequence:
                continue
            if current_time - item.timestamp > max(0, window_seconds):
                break
            previous = re.sub(r"[\W_]+", "", item.text.lower())
            if previous == normalized:
                return True
    return False


def reply_deadline(deps, event_time, mentioned: bool) -> float:
    total = (
        getattr(deps.settings, "mentioned_reply_total_timeout_seconds", 15)
        if mentioned
        else getattr(deps.settings, "normal_reply_total_timeout_seconds", 10)
    )
    try:
        elapsed = max(0.0, time.time() - float(event_time))
    except (TypeError, ValueError):
        elapsed = 0.0
    return time.monotonic() + max(0.0, float(total) - elapsed)


def remaining_reply_timeout(
    deadline: float,
    *,
    cap: int,
    reserve: int = 0,
) -> int:
    remaining = deadline - time.monotonic() - max(0, reserve)
    if remaining < 1:
        return 0
    return max(1, min(int(cap), int(remaining)))

