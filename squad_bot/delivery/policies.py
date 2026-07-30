from __future__ import annotations

import time


def message_max_age_seconds(deps, mentioned: bool) -> int:
    if mentioned:
        return deps.settings.mentioned_message_max_age_seconds
    return deps.settings.normal_message_max_age_seconds


def is_message_too_old(
    deps,
    event_time,
    mentioned: bool,
    *,
    fallback_time=None,
    now: float | None = None,
) -> bool:
    timestamp_value = event_time if event_time is not None else fallback_time
    try:
        event_timestamp = float(timestamp_value)
    except (TypeError, ValueError):
        return False
    max_age = deps.message_max_age_seconds(mentioned)
    if max_age <= 0:
        return False
    current_time = time.time() if now is None else now
    return current_time - event_timestamp > max_age


def is_event_too_old(deps, event: dict) -> bool:
    raw_message = event.get("message", "")
    mentioned = deps.is_mentioned(deps.settings.bot_qq, raw_message)
    return deps.is_message_too_old(event.get("time"), mentioned)


def acquire_reply_slot(
    deps,
    *,
    block: bool = True,
    reserve_slots: int = 0,
    deadline: float | None = None,
) -> bool:
    while True:
        now = time.time()
        with deps.rate_limit_lock:
            deps.reply_timestamps[:] = [
                timestamp
                for timestamp in deps.reply_timestamps
                if now - timestamp < 60
            ]
            capacity = max(0, deps.settings.max_replies_per_minute - reserve_slots)
            if len(deps.reply_timestamps) < capacity:
                deps.reply_timestamps.append(now)
                return True
            if not block:
                return False
            sleep_for = (
                max(1, 60 - (now - deps.reply_timestamps[0]))
                if deps.reply_timestamps
                else 60
            )
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or sleep_for > remaining:
                return False
        time.sleep(sleep_for)


def wait_for_rate_limit(deps, deadline: float | None = None) -> bool:
    return deps.acquire_reply_slot(deadline=deadline)

