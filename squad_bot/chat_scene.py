from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import threading
import time

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


def current_group_chat_scene(
    deps,
    group_id: int,
    *,
    focus_sequence: int = 0,
    now: float | None = None,
    stale_seconds: int | None = None,
) -> str:
    current_time = time.time() if now is None else now
    max_age = (
        getattr(deps.settings, "chat_scene_stale_seconds", 600)
        if stale_seconds is None
        else stale_seconds
    )
    return deps.chat_scene_state.current_summary(
        group_id,
        focus_sequence=focus_sequence,
        now=current_time,
        stale_seconds=max_age,
    )


def chat_scene_enabled_for_group(deps, group_id: int) -> bool:
    if not deps.auto_reply_enabled or not deps.settings.chat_reply_enabled:
        return False
    if not getattr(deps.settings, "chat_scene_enabled", True):
        return False
    return (
        not deps.settings.chat_allowed_group_ids
        or str(group_id) in deps.settings.chat_allowed_group_ids
    )


def finish_chat_scene_update(deps, group_id: int) -> None:
    deps.chat_scene_state.finish(group_id)


def chat_scene_update_loop(deps, group_id: int) -> None:
    while True:
        debounce = max(
            0.0, getattr(deps.settings, "chat_scene_debounce_seconds", 3.0)
        )
        if debounce:
            time.sleep(debounce)

        existing = deps.chat_scene_state.scene(group_id)
        last_updated = existing.updated_at if existing else 0.0
        min_interval = max(
            0.0,
            getattr(deps.settings, "chat_scene_update_interval_seconds", 30.0),
        )
        wait_seconds = min_interval - (time.time() - last_updated)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        target_sequence, previous = deps.chat_scene_state.begin_update(group_id)
        context = deps.recent_group_chat_context(
            group_id,
            now=time.time(),
            focus_sequence=target_sequence,
            through_sequence=target_sequence,
        )
        min_messages = max(
            1, getattr(deps.settings, "chat_scene_min_messages", 3)
        )
        if len(context) < min_messages:
            deps._finish_chat_scene_update(group_id)
            return

        summary = deps.analyze_chat_scene(
            base_url=deps.settings.llm_base_url,
            api_key=deps.settings.llm_api_key,
            model=getattr(
                deps.settings, "chat_scene_model", deps.settings.llm_model
            ),
            context=context,
            previous_scene=previous.summary if previous else "",
            timeout=max(
                1, getattr(deps.settings, "chat_scene_timeout_seconds", 30)
            ),
        )
        if summary:
            deps.chat_scene_state.set_scene(
                group_id,
                summary=summary,
                updated_at=time.time(),
                sequence=target_sequence,
            )
            logger.info("Updated chat scene %s %s", group_id, target_sequence)
        else:
            logger.error("Chat scene update failed %s %s", group_id, target_sequence)

        if not deps.chat_scene_state.should_continue(
            group_id,
            min_messages=min_messages,
        ):
            return


def schedule_chat_scene_update(deps, group_id: int, sequence: int) -> bool:
    if not sequence or not deps.chat_scene_enabled_for_group(group_id):
        return False
    min_messages = max(1, getattr(deps.settings, "chat_scene_min_messages", 3))
    if not deps.chat_scene_state.request_update(
        group_id,
        sequence,
        min_messages=min_messages,
    ):
        return False
    threading.Thread(
        target=deps._chat_scene_update_loop,
        args=(group_id,),
        daemon=True,
        name=f"chat-scene-{group_id}",
    ).start()
    return True
