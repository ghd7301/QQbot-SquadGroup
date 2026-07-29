from __future__ import annotations

import threading

from .models import GroupChatScene


class ChatSceneState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.scenes: dict[int, GroupChatScene] = {}
        self.requested_sequences: dict[int, int] = {}
        self.pending_messages: dict[int, int] = {}
        self.running: set[int] = set()

    def current_summary(
        self,
        group_id: int,
        *,
        focus_sequence: int,
        now: float,
        stale_seconds: int,
    ) -> str:
        with self.lock:
            scene = self.scenes.get(group_id)
            if not scene:
                return ""
            if stale_seconds > 0 and now - scene.updated_at > stale_seconds:
                return ""
            if focus_sequence and scene.sequence > focus_sequence:
                return ""
            return scene.summary

    def scene(self, group_id: int) -> GroupChatScene | None:
        with self.lock:
            return self.scenes.get(group_id)

    def request_update(
        self,
        group_id: int,
        sequence: int,
        *,
        min_messages: int,
    ) -> bool:
        with self.lock:
            self.requested_sequences[group_id] = sequence
            pending = self.pending_messages.get(group_id, 0) + 1
            self.pending_messages[group_id] = pending
            if group_id in self.running or pending < min_messages:
                return False
            self.running.add(group_id)
            return True

    def begin_update(
        self,
        group_id: int,
    ) -> tuple[int, GroupChatScene | None]:
        with self.lock:
            target_sequence = self.requested_sequences.get(group_id, 0)
            self.pending_messages[group_id] = 0
            return target_sequence, self.scenes.get(group_id)

    def set_scene(
        self,
        group_id: int,
        *,
        summary: str,
        updated_at: float,
        sequence: int,
    ) -> None:
        with self.lock:
            self.scenes[group_id] = GroupChatScene(
                summary=summary,
                updated_at=updated_at,
                sequence=sequence,
            )

    def should_continue(self, group_id: int, *, min_messages: int) -> bool:
        with self.lock:
            if self.pending_messages.get(group_id, 0) >= min_messages:
                return True
            self._finish_locked(group_id)
            return False

    def finish(self, group_id: int) -> None:
        with self.lock:
            self._finish_locked(group_id)

    def counts(self) -> tuple[int, int]:
        with self.lock:
            return len(self.scenes), len(self.running)

    def clear(self) -> None:
        with self.lock:
            self.scenes.clear()
            self.requested_sequences.clear()
            self.pending_messages.clear()
            self.running.clear()

    def _finish_locked(self, group_id: int) -> None:
        self.running.discard(group_id)
        self.requested_sequences.pop(group_id, None)
        self.pending_messages.pop(group_id, None)
