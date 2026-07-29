from __future__ import annotations

import json
import time
from pathlib import Path


def handle_onebot_event(deps, event: dict) -> tuple[int, dict]:
    if (
        event.get("post_type") == "notice"
        and event.get("notice_type") == "group_recall"
    ):
        group_id = event.get("group_id")
        if (
            not deps.settings.allowed_group_ids
            or str(group_id) in deps.settings.allowed_group_ids
        ):
            try:
                deps.recall_group_chat_message(
                    int(group_id), str(event.get("message_id") or "")
                )
                deps.schedule_chat_history_save()
            except (TypeError, ValueError):
                pass
        return (200, {"ok": True, "recalled": True})
    if event.get("message_type") != "group":
        print(
            "Ignored event: not group",
            event.get("post_type"),
            event.get("message_type"),
        )
        return (200, {"ok": True, "ignored": "not group message"})
    group_id = event.get("group_id")
    if (
        deps.settings.allowed_group_ids
        and str(group_id) not in deps.settings.allowed_group_ids
    ):
        print("Ignored event: group not allowed", group_id)
        (question, mentioned) = deps.extract_event_question(event)
        if mentioned:
            deps.write_message_audit(
                decision="ignored",
                reason="group not allowed",
                group_id=group_id,
                user_id=event.get("user_id"),
                question=question,
                mentioned=mentioned,
                event_time=event.get("time"),
            )
        return (200, {"ok": True, "ignored": "group not allowed"})
    raw_message = event.get("message", "")
    received_time = time.time()
    text = deps.extract_plain_text(raw_message)
    context_text = deps.extract_context_text(raw_message)
    content_segments = deps.extract_content_segments(raw_message)
    mentioned = deps.is_mentioned(deps.settings.bot_qq, raw_message)
    mentioned_user_ids = deps.extract_mentioned_user_ids(
        deps.settings.bot_qq, raw_message
    )
    user_id = event.get("user_id")
    if deps.settings.bot_qq and str(user_id) == deps.settings.bot_qq:
        return (200, {"ok": True, "ignored": "bot's own message"})
    numeric_group_id = int(group_id)
    reply_message_id = deps.extract_reply_message_id(raw_message)
    reply_target_user_id = ""
    reply_text = ""
    if reply_message_id:
        (reply_target_user_id, reply_text) = deps.resolve_reply_message_context(
            numeric_group_id, reply_message_id
        )
    chat_sequence = deps.record_group_chat_message(
        numeric_group_id,
        user_id,
        context_text,
        event.get("time"),
        message_id=str(event.get("message_id") or ""),
        reply_message_id=reply_message_id,
        reply_target_user_id=reply_target_user_id,
        reply_text=reply_text,
        mentioned_bot=mentioned,
        mentioned_user_ids=mentioned_user_ids,
        display_name=event.get("sender", {}).get("card")
        or event.get("sender", {}).get("nickname")
        or "",
        received_time=received_time,
        content_segments=content_segments,
    )
    deps.schedule_chat_history_save()
    deps.schedule_chat_scene_update(numeric_group_id, chat_sequence)
    try:
        deps.flush_fragment_buffer_for_new_speaker(
            numeric_group_id, user_id, defer_dispatch=True
        )
    except Exception as exc:
        print("Fragment queue write failed:", repr(exc))
        return (503, {"ok": False, "error": "queue unavailable"})
    if deps.is_event_too_old(event):
        deps.write_message_audit(
            decision="ignored",
            reason="message too old",
            group_id=group_id,
            user_id=user_id,
            question=text,
            mentioned=mentioned,
            event_time=event.get("time"),
        )
        return (200, {"ok": True, "ignored": "message too old"})
    (continue_processing, mentioned, reply_reason) = deps.classify_reply_target(
        reply_message_id, reply_target_user_id, mentioned, deps.settings.bot_qq
    )
    if not continue_processing:
        deps.flush_group_fragment_buffer(numeric_group_id, defer_dispatch=True)
        deps.write_message_audit(
            decision="ignored",
            reason=reply_reason,
            group_id=group_id,
            user_id=user_id,
            question=text,
            mentioned=mentioned,
            reply_message_id=reply_message_id,
            reply_target_user_id=reply_target_user_id,
            event_time=event.get("time"),
        )
        return (200, {"ok": True, "ignored": reply_reason})
    try:
        context_now = float(event.get("time"))
    except (TypeError, ValueError):
        context_now = time.time()
    chat_context = deps.recent_group_chat_context(
        numeric_group_id, now=context_now, focus_sequence=chat_sequence
    )
    explicit_knowledge_command = bool(
        deps.settings.command_prefix
        and text.strip().startswith(deps.settings.command_prefix)
    )
    (ok, question) = deps.should_respond(
        text, deps.settings.command_prefix, deps.settings.bot_qq, raw_message
    )
    print(
        "Group event",
        group_id,
        "mentioned",
        mentioned,
        "queued",
        ok,
        "question",
        question,
    )
    Path("work/last_onebot_event.json").parent.mkdir(exist_ok=True)
    Path("work/last_onebot_event.json").write_text(
        json.dumps(
            {
                "group_id": group_id,
                "raw_message": raw_message,
                "text": text,
                "mentioned": mentioned,
                "reply_message_id": reply_message_id,
                "reply_target_user_id": reply_target_user_id,
                "reply_reason": reply_reason,
                "queued": ok,
                "question": question,
                "mentioned_user_ids": list(mentioned_user_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not ok or not question:
        deps.write_message_audit(
            decision="ignored",
            reason="no trigger",
            group_id=group_id,
            user_id=event.get("user_id"),
            question=question,
            mentioned=mentioned,
            event_time=event.get("time"),
        )
        return (200, {"ok": True, "ignored": "no trigger"})
    item = {
        "group_id": group_id,
        "question": question,
        "mentioned": mentioned,
        "time": event.get("time"),
        "user_id": user_id,
        "sender_role": event.get("sender", {}).get("role", ""),
        "chat_context": list(chat_context),
        "mentions_other": bool(mentioned_user_ids),
        "mentioned_user_ids": list(mentioned_user_ids),
        "reply_message_id": reply_message_id,
        "reply_target_user_id": reply_target_user_id,
        "reply_text": reply_text,
        "message_id": str(event.get("message_id") or ""),
        "chat_sequence": chat_sequence,
        "received_time": received_time,
        "content_segments": list(content_segments),
        "message_status": "active",
        "explicit_knowledge_command": explicit_knowledge_command,
    }
    fragment_audience = deps.classify_fragment_audience(item)
    try:
        pending_ids = deps.submit_message_fragment(item, defer_dispatch=True)
    except Exception as exc:
        print("Pending queue write failed:", repr(exc))
        deps.write_message_audit(
            decision="error",
            reason=f"pending queue write failed: {exc!r}",
            group_id=group_id,
            user_id=event.get("user_id"),
            question=question,
            mentioned=mentioned,
            event_time=event.get("time"),
        )
        return (503, {"ok": False, "error": "queue unavailable"})
    if fragment_audience == "human":
        deps.write_message_audit(
            decision="ignored",
            reason="fragment directed at another member",
            group_id=group_id,
            user_id=user_id,
            question=question,
            mentioned=False,
            event_time=event.get("time"),
        )
        return (200, {"ok": True, "ignored": "directed at another member"})
    return (
        200,
        {
            "ok": True,
            "buffered": not pending_ids,
            "queued": bool(pending_ids),
            "mentioned": mentioned,
        },
    )
