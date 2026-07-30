from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from pathlib import Path
from typing import Sequence

from .chat_memory import MemoryMessage
from .models import GroupChatMessage


class ChatHistoryState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.messages: dict[int, list[GroupChatMessage]] = {}
        self.sequence = 0

    def record(
        self,
        group_id: int,
        user_id,
        text: str,
        event_time=None,
        *,
        context_seconds: int,
        context_messages: int,
        message_id: str = "",
        reply_message_id: str = "",
        reply_target_user_id: str = "",
        reply_text: str = "",
        mentioned_bot: bool = False,
        mentioned_user_ids: Sequence[str] = (),
        display_name: str = "",
        generated_for_message_ids: Sequence[str] = (),
        turn_id: str = "",
        reply_mode: str = "",
        semantic_topic: str = "",
        received_time=None,
        content_segments: Sequence[dict[str, str]] = (),
        message_status: str = "active",
    ) -> GroupChatMessage | None:
        normalized = text.strip()
        if not normalized:
            return None
        try:
            timestamp = float(event_time)
        except (TypeError, ValueError):
            timestamp = time.time()
        try:
            received_timestamp = float(received_time)
        except (TypeError, ValueError):
            received_timestamp = time.time()
        normalized_status = str(message_status or "active").strip().lower()
        if normalized_status not in {"active", "recalled", "edited", "invalid"}:
            normalized_status = "active"
        safe_segments = tuple(
            {
                str(key): str(value)[:1000]
                for key, value in segment.items()
                if str(key) in {"type", "text"}
            }
            for segment in content_segments
            if isinstance(segment, dict) and str(segment.get("type") or "").strip()
        )[:24]
        window = max(0, context_seconds)
        max_messages = max(1, context_messages)
        with self.lock:
            self.sequence += 1
            entry = GroupChatMessage(
                text=normalized,
                user_id=str(user_id or ""),
                timestamp=timestamp,
                sequence=self.sequence,
                message_id=str(message_id or ""),
                reply_message_id=str(reply_message_id or ""),
                reply_target_user_id=str(reply_target_user_id or ""),
                reply_text=str(reply_text or "").strip(),
                mentioned_bot=bool(mentioned_bot),
                mentioned_user_ids=tuple(
                    str(value) for value in mentioned_user_ids if str(value).strip()
                ),
                display_name=str(display_name or "").strip(),
                generated_for_message_ids=tuple(
                    dict.fromkeys(
                        str(value).strip()
                        for value in generated_for_message_ids
                        if str(value).strip()
                    )
                ),
                turn_id=str(turn_id or "").strip(),
                reply_mode=str(reply_mode or "").strip(),
                semantic_topic=str(semantic_topic or "").strip(),
                received_time=received_timestamp,
                content_segments=safe_segments,
                message_status=normalized_status,
            )
            history = self.messages.setdefault(group_id, [])
            history.append(entry)
            if window > 0:
                history[:] = [
                    item for item in history if timestamp - item.timestamp <= window
                ]
            history[:] = history[-max(max_messages * 4, 24):]
            return entry

    def find(self, group_id: int, message_id: str) -> GroupChatMessage | None:
        target = str(message_id or "").strip()
        if not target:
            return None
        with self.lock:
            for item in reversed(self.messages.get(group_id, ())):
                if item.message_id == target and item.message_status == "active":
                    return item
        return None

    def has_newer_user_message(
        self,
        group_id: int,
        sequence: int,
        *,
        bot_user_id: str,
    ) -> bool:
        with self.lock:
            return any(
                item.sequence > sequence and item.user_id != bot_user_id
                for item in self.messages.get(group_id, ())
            )

    def latest_user_sequence(self, group_id: int, *, bot_user_id: str) -> int:
        with self.lock:
            return max(
                (
                    item.sequence
                    for item in self.messages.get(group_id, ())
                    if item.user_id != bot_user_id
                ),
                default=0,
            )

    def recent(
        self,
        group_id: int,
        *,
        now: float,
        context_seconds: int,
        max_messages: int,
        focus_sequence: int = 0,
        through_sequence: int = 0,
    ) -> tuple[GroupChatMessage, ...]:
        if context_seconds <= 0 or max_messages <= 0:
            return ()
        with self.lock:
            history = self.messages.get(group_id, [])
            recent = [
                item for item in history if now - item.timestamp <= context_seconds
            ]
            if recent:
                self.messages[group_id] = recent
            else:
                self.messages.pop(group_id, None)
        active = [item for item in recent if item.message_status == "active"]
        available = (
            [item for item in active if item.sequence <= through_sequence]
            if through_sequence
            else active
        )
        selected = available[-max_messages:]
        if focus_sequence:
            focus = next(
                (item for item in available if item.sequence == focus_sequence),
                None,
            )
            if focus and focus.reply_message_id:
                replied = next(
                    (
                        item
                        for item in available
                        if item.message_id == focus.reply_message_id
                    ),
                    None,
                )
                if replied and replied not in selected:
                    tail = selected[-(max_messages - 1):] if max_messages > 1 else []
                    selected = sorted([replied, *tail], key=lambda item: item.sequence)
        return tuple(selected)

    def recall(self, group_id: int, message_id: str) -> None:
        target = str(message_id or "").strip()
        if not target:
            return
        with self.lock:
            for item in self.messages.get(group_id, []):
                if item.message_id == target:
                    item.message_status = "recalled"

    def snapshot(self) -> tuple[tuple[int, GroupChatMessage], ...]:
        with self.lock:
            return tuple(
                (group_id, item)
                for group_id, messages in self.messages.items()
                for item in messages
            )

    def clear(self) -> None:
        with self.lock:
            self.messages.clear()
            self.sequence = 0

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        with self.lock:
            data = {
                str(group_id): [self._message_to_dict(message) for message in messages]
                for group_id, messages in self.messages.items()
            }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path) -> int:
        load_path = Path(path)
        if not load_path.exists():
            return 0
        data = json.loads(load_path.read_text(encoding="utf-8"))
        count = 0
        with self.lock:
            for group_id_str, messages in data.items():
                group_id = int(group_id_str)
                history = self.messages.setdefault(group_id, [])
                for raw in messages:
                    entry = self._message_from_dict(raw)
                    history.append(entry)
                    self.sequence = max(self.sequence, entry.sequence)
                    count += 1
        return count

    @staticmethod
    def _message_to_dict(message: GroupChatMessage) -> dict:
        return {
            "text": message.text,
            "user_id": message.user_id,
            "timestamp": message.timestamp,
            "sequence": message.sequence,
            "message_id": message.message_id,
            "reply_message_id": message.reply_message_id,
            "reply_target_user_id": message.reply_target_user_id,
            "reply_text": message.reply_text,
            "mentioned_bot": message.mentioned_bot,
            "mentioned_user_ids": list(message.mentioned_user_ids),
            "display_name": message.display_name,
            "generated_for_message_ids": list(message.generated_for_message_ids),
            "turn_id": message.turn_id,
            "reply_mode": message.reply_mode,
            "semantic_topic": message.semantic_topic,
            "received_time": message.received_time,
            "content_segments": list(message.content_segments),
            "message_status": message.message_status,
        }

    @staticmethod
    def _message_from_dict(raw: dict) -> GroupChatMessage:
        return GroupChatMessage(
            text=raw["text"],
            user_id=raw["user_id"],
            timestamp=raw["timestamp"],
            sequence=raw["sequence"],
            message_id=raw.get("message_id", ""),
            reply_message_id=raw.get("reply_message_id", ""),
            reply_target_user_id=raw.get("reply_target_user_id", ""),
            reply_text=raw.get("reply_text", ""),
            mentioned_bot=bool(raw.get("mentioned_bot")),
            mentioned_user_ids=tuple(raw.get("mentioned_user_ids") or ()),
            display_name=raw.get("display_name", ""),
            generated_for_message_ids=tuple(raw.get("generated_for_message_ids") or ()),
            turn_id=raw.get("turn_id", ""),
            reply_mode=raw.get("reply_mode", ""),
            semantic_topic=raw.get("semantic_topic", ""),
            received_time=float(raw.get("received_time") or raw["timestamp"]),
            content_segments=tuple(raw.get("content_segments") or ()),
            message_status=str(raw.get("message_status") or "active"),
        )


def record_group_chat_message(
    deps,
    group_id: int,
    user_id,
    text: str,
    event_time=None,
    *,
    message_id: str = "",
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    reply_text: str = "",
    mentioned_bot: bool = False,
    mentioned_user_ids: Sequence[str] = (),
    display_name: str = "",
    generated_for_message_ids: Sequence[str] = (),
    turn_id: str = "",
    reply_mode: str = "",
    semantic_topic: str = "",
    received_time=None,
    content_segments: Sequence[dict[str, str]] = (),
    message_status: str = "active",
) -> int:
    with deps.chat_history_lock:
        entry = deps.chat_history_state.record(
            group_id,
            user_id,
            text,
            event_time,
            context_seconds=deps.settings.chat_context_seconds,
            context_messages=deps.settings.chat_context_messages,
            message_id=message_id,
            reply_message_id=reply_message_id,
            reply_target_user_id=reply_target_user_id,
            reply_text=reply_text,
            mentioned_bot=mentioned_bot,
            mentioned_user_ids=mentioned_user_ids,
            display_name=display_name,
            generated_for_message_ids=generated_for_message_ids,
            turn_id=turn_id,
            reply_mode=reply_mode,
            semantic_topic=semantic_topic,
            received_time=received_time,
            content_segments=content_segments,
            message_status=message_status,
        )
        if entry is None:
            return 0
        deps.chat_message_sequence = deps.chat_history_state.sequence
    if deps.chat_memory_manager:
        speaker_id = deps.stable_member_id(group_id, entry.user_id)
        deps.chat_memory_manager.enqueue(
            MemoryMessage(
                group_id=group_id,
                message_id=entry.message_id or f"local:{group_id}:{entry.sequence}",
                speaker_id=speaker_id,
                display_name=entry.display_name if speaker_id != "bot" else "机器人",
                speaker_role="bot" if speaker_id == "bot" else "member",
                text=entry.text,
                event_time=entry.timestamp,
                reply_message_id=entry.reply_message_id,
                reply_speaker_id=deps.stable_member_id(
                    group_id, entry.reply_target_user_id
                ),
                quoted_text=entry.reply_text,
                mentions=tuple(
                    deps.stable_member_id(group_id, value)
                    for value in entry.mentioned_user_ids
                ),
                generated_for_message_ids=entry.generated_for_message_ids,
                turn_id=entry.turn_id,
                reply_mode=entry.reply_mode,
                semantic_topic=entry.semantic_topic,
                sequence=entry.sequence,
                received_time=entry.received_time,
                content_segments=entry.content_segments,
                message_status=entry.message_status,
            )
        )
    return entry.sequence


def find_group_chat_message(deps, group_id: int, message_id: str) -> GroupChatMessage | None:
    return deps.chat_history_state.find(group_id, message_id)


def resolve_reply_message_context(
    deps,
    group_id: int,
    message_id: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    target = str(message_id or "").strip()
    if not target:
        return "", ""
    replied = deps.find_group_chat_message(group_id, target)
    if replied:
        return replied.user_id, replied.text
    sender_id, text = deps.get_message_info(
        deps.settings.onebot_api_url,
        target,
        deps.settings.onebot_access_token,
        deps.settings.onebot_message_lookup_timeout_seconds,
    )
    if sender_id:
        return sender_id, text
    turn = deps.load_conversation_turn_by_bot_message_id(
        group_id,
        target,
        db_path=db_path,
    )
    if turn:
        return deps.settings.bot_qq, turn.last_answer
    return "", text


def stable_member_id(deps, group_id: int, user_id: str) -> str:
    user_key = str(user_id or "").strip()
    if user_key == deps.settings.bot_qq:
        return "bot"
    if not user_key:
        return "unknown_member"
    secret = (
        getattr(deps.settings, "member_id_secret", "")
        or getattr(deps.settings, "onebot_access_token", "")
        or deps.settings.bot_qq
        or "local-member-id"
    )
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{group_id}:{user_key}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:10]
    return f"member_{digest}"


def context_message_payload(
    deps,
    group_id: int,
    item: GroupChatMessage,
    *,
    current: bool,
) -> dict:
    speaker_id = deps.stable_member_id(group_id, item.user_id)
    payload: dict = {
        "current": current,
        "message_id": item.message_id,
        "sequence": item.sequence,
        "event_time": item.timestamp,
        "received_time": item.received_time,
        "message_status": item.message_status,
        "speaker": {
            "id": speaker_id,
            "role": "bot" if speaker_id == "bot" else "member",
            "is_self": speaker_id == "bot",
            "display_name": item.display_name if speaker_id != "bot" else "机器人",
        },
        "text": item.text,
        "content_segments": list(item.content_segments),
        "mentions": [
            deps.stable_member_id(group_id, user_id)
            for user_id in item.mentioned_user_ids
        ],
        "mentions_bot": item.mentioned_bot,
        "generated_for_message_ids": list(item.generated_for_message_ids),
        "turn_id": item.turn_id,
        "reply_mode": item.reply_mode,
        "semantic_topic": item.semantic_topic,
    }
    if item.reply_message_id:
        target_id = deps.stable_member_id(group_id, item.reply_target_user_id)
        payload["reply_to"] = {
            "message_id": item.reply_message_id,
            "speaker_id": target_id,
            "speaker_role": "bot" if target_id == "bot" else "member",
            "quoted_text": item.reply_text,
        }
    else:
        payload["reply_to"] = None
    return payload


def recent_group_chat_context(
    deps,
    group_id: int,
    *,
    now: float | None = None,
    context_seconds: int | None = None,
    max_messages: int | None = None,
    focus_sequence: int = 0,
    through_sequence: int = 0,
) -> tuple[str, ...]:
    current_time = time.time() if now is None else now
    window = (
        deps.settings.chat_context_seconds
        if context_seconds is None
        else context_seconds
    )
    limit = (
        deps.settings.chat_context_messages
        if max_messages is None
        else max_messages
    )
    selected = deps.chat_history_state.recent(
        group_id,
        now=current_time,
        context_seconds=window,
        max_messages=limit,
        focus_sequence=focus_sequence,
        through_sequence=through_sequence,
    )
    return tuple(
        json.dumps(
            deps._context_message_payload(
                group_id,
                item,
                current=item.sequence == focus_sequence,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in selected
    )


def save_chat_history(deps, path: str | Path | None = None) -> None:
    """Persist chat history to disk for restart recovery."""
    save_path = Path(path or "work/chat_history.json")
    try:
        deps.chat_history_state.save(save_path)
    except Exception as exc:
        print("Save chat history failed:", repr(exc))


def schedule_chat_history_save(deps) -> None:
    deps.chat_history_save_event.set()


def chat_history_save_worker(deps) -> None:
    while True:
        deps.chat_history_save_event.wait()
        time.sleep(0.5)
        deps.chat_history_save_event.clear()
        deps.save_chat_history()


def initialize_chat_memory(deps) -> bool:
    if not getattr(deps.settings, "chat_memory_enabled", True):
        return False
    try:
        store = deps.ChatMemoryStore(
            getattr(deps.settings, "chat_memory_db", "work/chat_memory.sqlite3"),
            deps.build_embedding_provider(deps.settings),
        )
        deps.chat_memory_manager = deps.ChatMemoryManager(
            store,
            retention_days=max(
                0, getattr(deps.settings, "chat_memory_retention_days", 90)
            ),
        )
        deps.chat_memory_manager.start()
        return True
    except Exception as exc:
        deps.chat_memory_manager = None
        print("Chat memory initialization failed:", type(exc).__name__, repr(exc))
        return False


def recall_group_chat_message(deps, group_id: int, message_id: str) -> None:
    target = str(message_id or "").strip()
    if not target:
        return
    deps.chat_history_state.recall(group_id, target)
    if deps.chat_memory_manager:
        deps.chat_memory_manager.enqueue_recall(group_id, target)


def load_chat_history(deps, path: str | Path | None = None) -> int:
    """Load persisted chat history on startup."""
    load_path = Path(path or "work/chat_history.json")
    if not load_path.exists():
        return 0
    try:
        count = deps.chat_history_state.load(load_path)
        with deps.chat_history_lock:
            deps.chat_message_sequence = deps.chat_history_state.sequence
        print(f"Loaded {count} chat history entries from {load_path}")
        return count
    except Exception as exc:
        print("Load chat history failed:", repr(exc))
        return 0


def migrate_loaded_chat_history_to_memory(deps) -> int:
    if not deps.chat_memory_manager:
        return 0
    snapshot = deps.chat_history_state.snapshot()
    queued = 0
    for group_id, item in snapshot:
        speaker_id = deps.stable_member_id(group_id, item.user_id)
        queued += int(
            deps.chat_memory_manager.enqueue(
                MemoryMessage(
                    group_id=group_id,
                    message_id=(
                        item.message_id or f"local:{group_id}:{item.sequence}"
                    ),
                    speaker_id=speaker_id,
                    display_name=(
                        item.display_name if speaker_id != "bot" else "机器人"
                    ),
                    speaker_role="bot" if speaker_id == "bot" else "member",
                    text=item.text,
                    event_time=item.timestamp,
                    reply_message_id=item.reply_message_id,
                    reply_speaker_id=deps.stable_member_id(
                        group_id, item.reply_target_user_id
                    ),
                    quoted_text=item.reply_text,
                    mentions=tuple(
                        deps.stable_member_id(group_id, value)
                        for value in item.mentioned_user_ids
                    ),
                    generated_for_message_ids=item.generated_for_message_ids,
                    turn_id=item.turn_id,
                    reply_mode=item.reply_mode,
                    semantic_topic=item.semantic_topic,
                    sequence=item.sequence,
                    received_time=item.received_time,
                    content_segments=item.content_segments,
                    message_status=item.message_status,
                )
            )
        )
    return queued
