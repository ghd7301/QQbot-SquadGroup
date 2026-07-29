from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from .models import MessageFragmentBuffer


def message_ids(item: dict) -> list[str]:
    result: list[str] = []
    for candidate in item.get("message_ids") or (item.get("message_id"),):
        value = str(candidate or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def classify_audience(item: dict, *, bot_user_id: str) -> str:
    reply_message_id = str(item.get("reply_message_id") or "")
    reply_target_user_id = str(item.get("reply_target_user_id") or "")
    if item.get("mentioned") or item.get("explicit_knowledge_command"):
        return "bot"
    if reply_message_id:
        if bot_user_id and reply_target_user_id == bot_user_id:
            return "bot"
        return "human"
    if item.get("mentioned_user_ids"):
        return "human"
    return "unknown"


def items_compatible(
    buffer: MessageFragmentBuffer,
    item: dict,
    audience: str,
) -> bool:
    if str(item.get("user_id") or "") != buffer.user_id:
        return False
    if {buffer.audience, audience} == {"bot", "human"}:
        return False
    if audience == "human" or buffer.audience == "human":
        return audience == buffer.audience
    if audience == "bot" and buffer.audience == "unknown":
        return False
    if audience == "bot" and any(
        fragment.get("fragment_audience") == "unknown"
        for fragment in buffer.fragments[1:]
    ):
        return False
    old_reply_id = str(buffer.item.get("reply_message_id") or "")
    new_reply_id = str(item.get("reply_message_id") or "")
    if old_reply_id and new_reply_id and old_reply_id != new_reply_id:
        return False
    old_target = str(buffer.item.get("reply_target_user_id") or "")
    new_target = str(item.get("reply_target_user_id") or "")
    if old_target and new_target and old_target != new_target:
        return False
    return True


class FragmentAggregator:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.buffers: dict[int, MessageFragmentBuffer] = {}
        self.ready: list[MessageFragmentBuffer] = []

    def new_buffer(
        self,
        item: dict,
        audience: str,
        now: float,
        *,
        debounce_seconds: float,
        max_wait_seconds: float,
    ) -> MessageFragmentBuffer:
        buffered_item = dict(item)
        buffered_item["message_ids"] = message_ids(item)
        buffered_item["fragment_audience"] = audience
        question = str(item.get("question") or "").strip()
        fragment = dict(item)
        fragment["fragment_audience"] = audience
        max_wait = max(0.0, max_wait_seconds)
        debounce = max(0.0, debounce_seconds)
        deadline = min(now + debounce, now + max_wait) if max_wait else now + debounce
        return MessageFragmentBuffer(
            group_id=int(item["group_id"]),
            user_id=str(item.get("user_id") or ""),
            audience=audience,
            item=buffered_item,
            parts=[question],
            fragments=[fragment],
            started_at=now,
            deadline=deadline,
        )

    def merge(
        self,
        buffer: MessageFragmentBuffer,
        item: dict,
        audience: str,
        now: float,
        *,
        debounce_seconds: float,
        max_wait_seconds: float,
    ) -> None:
        question = str(item.get("question") or "").strip()
        buffer.parts.append(question)
        fragment = dict(item)
        fragment["fragment_audience"] = audience
        buffer.fragments.append(fragment)
        buffer.item["question"] = "\n".join(part for part in buffer.parts if part)
        if audience == "bot":
            buffer.audience = "bot"
        buffer.item["fragment_audience"] = buffer.audience
        buffer.item["mentioned"] = bool(
            buffer.item.get("mentioned") or item.get("mentioned")
        )
        mentioned_ids = list(buffer.item.get("mentioned_user_ids") or ())
        for candidate in item.get("mentioned_user_ids") or ():
            value = str(candidate or "").strip()
            if value and value not in mentioned_ids:
                mentioned_ids.append(value)
        buffer.item["mentioned_user_ids"] = mentioned_ids
        buffer.item["mentions_other"] = bool(mentioned_ids)
        combined_message_ids = message_ids(buffer.item)
        for message_id in message_ids(item):
            if message_id not in combined_message_ids:
                combined_message_ids.append(message_id)
        buffer.item["message_ids"] = combined_message_ids
        buffer.item["message_id"] = str(
            item.get("message_id") or buffer.item.get("message_id") or ""
        )
        for key in ("reply_message_id", "reply_target_user_id", "reply_text"):
            if not buffer.item.get(key) and item.get(key):
                buffer.item[key] = item[key]
        for key in ("time", "sender_role", "chat_context", "chat_sequence"):
            if key in item:
                buffer.item[key] = item[key]
        debounce = max(0.0, debounce_seconds)
        max_wait = max(0.0, max_wait_seconds)
        debounce_deadline = now + debounce
        hard_deadline = buffer.started_at + max_wait if max_wait else debounce_deadline
        buffer.deadline = min(debounce_deadline, hard_deadline)

    def submit(
        self,
        item: dict,
        audience: str,
        *,
        now: float,
        is_immediate: bool,
        max_parts: int,
        max_chars: int,
        debounce_seconds: float,
        max_wait_seconds: float,
    ) -> tuple[MessageFragmentBuffer, ...]:
        group_id = int(item["group_id"])
        displaced: list[MessageFragmentBuffer] = []
        with self.condition:
            current = self.buffers.get(group_id)
            if current and (
                is_immediate
                or audience == "human"
                or not items_compatible(current, item, audience)
            ):
                displaced.append(self.buffers.pop(group_id))
                current = None

            if audience != "human" and not is_immediate:
                part_limit = max(1, max_parts)
                char_limit = max(1, max_chars)
                would_exceed = bool(
                    current
                    and (
                        len(current.parts) >= part_limit
                        or len(
                            "\n".join(
                                (*current.parts, str(item.get("question") or ""))
                            )
                        )
                        > char_limit
                    )
                )
                if would_exceed:
                    displaced.append(self.buffers.pop(group_id))
                    current = None
                if current is None:
                    current = self.new_buffer(
                        item,
                        audience,
                        now,
                        debounce_seconds=debounce_seconds,
                        max_wait_seconds=max_wait_seconds,
                    )
                    self.buffers[group_id] = current
                else:
                    self.merge(
                        current,
                        item,
                        audience,
                        now,
                        debounce_seconds=debounce_seconds,
                        max_wait_seconds=max_wait_seconds,
                    )
                if (
                    len(current.parts) >= part_limit
                    or len(str(current.item.get("question") or "")) >= char_limit
                    or current.deadline <= now
                ):
                    displaced.append(self.buffers.pop(group_id))
            self.condition.notify_all()
        return tuple(displaced)

    def prefix_item(self, buffer: MessageFragmentBuffer, count: int) -> dict:
        selected = buffer.fragments[:count]
        item = dict(buffer.item)
        item["question"] = "\n".join(
            str(fragment.get("question") or "").strip() for fragment in selected
        )
        item["mentioned"] = any(fragment.get("mentioned") for fragment in selected)
        mentioned_ids: list[str] = []
        selected_message_ids: list[str] = []
        for fragment in selected:
            for candidate in fragment.get("mentioned_user_ids") or ():
                value = str(candidate or "").strip()
                if value and value not in mentioned_ids:
                    mentioned_ids.append(value)
            for message_id in message_ids(fragment):
                if message_id not in selected_message_ids:
                    selected_message_ids.append(message_id)
        item["mentioned_user_ids"] = mentioned_ids
        item["mentions_other"] = bool(mentioned_ids)
        item["message_ids"] = selected_message_ids
        item["message_id"] = str(selected[-1].get("message_id") or "")
        for key in ("reply_message_id", "reply_target_user_id", "reply_text"):
            item[key] = next(
                (fragment.get(key) for fragment in selected if fragment.get(key)),
                "",
            )
        return item

    def defer(self, buffers: Sequence[MessageFragmentBuffer]) -> None:
        if not buffers:
            return
        with self.condition:
            self.ready.extend(buffers)
            self.condition.notify_all()

    def pop_group(self, group_id: int) -> MessageFragmentBuffer | None:
        with self.condition:
            buffer = self.buffers.pop(int(group_id), None)
            self.condition.notify_all()
            return buffer

    def pop_for_new_speaker(
        self,
        group_id: int,
        user_id,
    ) -> MessageFragmentBuffer | None:
        with self.condition:
            buffer = self.buffers.get(int(group_id))
            if not buffer or buffer.user_id == str(user_id or ""):
                return None
            buffer = self.buffers.pop(int(group_id))
            self.condition.notify_all()
            return buffer

    def wait_for_due(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> tuple[MessageFragmentBuffer, ...]:
        with self.condition:
            while True:
                if self.ready:
                    due = tuple(self.ready)
                    self.ready.clear()
                    return due
                now = monotonic()
                due_group_ids = [
                    group_id
                    for group_id, buffer in self.buffers.items()
                    if buffer.deadline <= now
                ]
                if due_group_ids:
                    return tuple(self.buffers.pop(group_id) for group_id in due_group_ids)
                if not self.buffers:
                    self.condition.wait()
                else:
                    next_deadline = min(
                        buffer.deadline for buffer in self.buffers.values()
                    )
                    self.condition.wait(timeout=max(0.0, next_deadline - now))

    def buffered_count(self) -> int:
        with self.condition:
            return len(self.buffers)

    def clear(self) -> None:
        with self.condition:
            self.buffers.clear()
            self.ready.clear()
            self.condition.notify_all()
