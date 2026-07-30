from __future__ import annotations

import json
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def message_ids(deps, item: dict) -> list[str]:
    return deps.message_ids(item)


def bot_turn_metadata(deps, item: dict, bot_message_id) -> tuple[tuple[str, ...], str]:
    trigger_ids = tuple(deps._message_ids(item))
    bot_id = str(bot_message_id or "").strip()
    turn_id = f"bot:{bot_id}" if bot_id else ""
    return trigger_ids, turn_id


def send_and_record_bot_turn(
    deps,
    *,
    group_id: int,
    item: dict,
    answer: str,
    reply_mode: str,
    semantic_topic: str = "",
    mention_user_id: str = "",
    reply_to_trigger: bool = False,
) -> tuple[object, tuple[str, ...], str]:
    trigger_message_id = str(item.get("message_id") or "")
    user_id = str(item.get("user_id") or "")
    deps.begin_pending_dispatch(item)
    bot_message_id = deps.send_group_msg(
        deps.settings.onebot_api_url,
        group_id,
        answer,
        deps.settings.onebot_access_token,
        mention_user_id=mention_user_id,
        reply_to_message_id=trigger_message_id if reply_to_trigger else "",
    )
    item["_dispatch_completed"] = True
    item["_sent_message_id"] = str(bot_message_id or "")
    trigger_message_ids, turn_id = deps.bot_turn_metadata(item, bot_message_id)
    try:
        deps.record_group_chat_message(
            group_id,
            deps.settings.bot_qq,
            answer,
            message_id=bot_message_id,
            reply_message_id=trigger_message_id if reply_to_trigger else "",
            reply_target_user_id=user_id if reply_to_trigger else "",
            reply_text=str(item.get("question") or "") if reply_to_trigger else "",
            generated_for_message_ids=trigger_message_ids,
            turn_id=turn_id,
            reply_mode=reply_mode,
            semantic_topic=semantic_topic,
        )
    except Exception as record_exc:
        logger.error("Post-send recording failed: %s", repr(record_exc))
        # Mark as sent_unknown so the pending entry is not retried
        # (message was already delivered, we just failed to record it)
        pending_id = item.get("_pending_id")
        if pending_id is not None:
            try:
                deps.mark_pending_sent_unknown(int(pending_id), repr(record_exc))
            except Exception:
                pass
        deps.write_message_audit(
            decision="error",
            reason=f"post-send recording failed: {record_exc!r}",
            group_id=group_id,
            user_id=user_id,
            question=str(item.get("question") or ""),
            event_time=item.get("time"),
        )
    try:
        deps.save_chat_history()
    except Exception as save_exc:
        logger.error("Chat history save failed: %s", repr(save_exc))
    return bot_message_id, trigger_message_ids, turn_id


def message_already_covered_by_bot(deps, group_id: int, message_id) -> bool:
    """Return whether a completed bot turn already covers this incoming message."""
    target = str(message_id or "").strip()
    if not target:
        return False
    with deps.chat_history_lock:
        return any(
            entry.user_id == str(deps.settings.bot_qq or "")
            and entry.message_status == "active"
            and target in entry.generated_for_message_ids
            for entry in deps.group_chat_history.get(int(group_id), ())
        )


def context_message_speakers(context: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in context:
        try:
            payload = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        message_id = str(payload.get("message_id") or "").strip()
        if message_id:
            speaker = (
                payload.get("speaker")
                if isinstance(payload.get("speaker"), dict)
                else {}
            )
            result[message_id] = str(speaker.get("id") or "").strip()
    return result


def merge_review_message_ids(
    deps, item: dict | None, review, latest_context: Sequence[str]
) -> None:
    """Attach only real context message IDs selected by the semantic reviewer."""
    if item is None:
        return
    existing = deps._message_ids(item)
    context_speakers = deps._context_message_speakers(latest_context)
    original_speakers = {
        context_speakers[message_id]
        for message_id in existing
        if context_speakers.get(message_id)
    }
    for message_id in review.related_message_ids:
        value = str(message_id or "").strip()
        same_sender = bool(
            original_speakers and context_speakers.get(value) in original_speakers
        )
        if value and same_sender and value not in existing:
            existing.append(value)
    item["message_ids"] = existing

