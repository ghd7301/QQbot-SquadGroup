from __future__ import annotations

import time
from collections.abc import Callable


class PendingItemLifecycle:
    def __init__(self, item: dict) -> None:
        self.item = item
        self.terminal = True

    def transfer(self) -> None:
        self.terminal = False

    def handle_failure(
        self,
        error: str,
        handler: Callable[[dict, str], str],
    ) -> str:
        action = handler(self.item, error)
        self.terminal = action == "delivered"
        return action

    def acknowledge(
        self,
        delete: Callable[[int], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        pending_id = self.item.get("_pending_id")
        if not self.terminal or pending_id is None:
            return
        try:
            delete(int(pending_id))
        except Exception as exc:
            on_error(exc)


def normal_lane_should_yield(lane: str, *, priority_pending: bool) -> bool:
    return lane == "normal" and priority_pending


def run_worker(deps, work_queue, lane: str) -> None:
    while True:
        priority, sequence, item = work_queue.get()
        if deps.normal_lane_should_yield(
            lane,
            priority_pending=not deps.message_queue.empty(),
        ):
            work_queue.put((priority, sequence, item))
            work_queue.task_done()
            time.sleep(0.05)
            continue
        lifecycle = deps.PendingItemLifecycle(item)
        try:
            deps.process_worker_item(item, lane, lifecycle)
        finally:
            lifecycle.acknowledge(
                deps.delete_pending_message,
                lambda exc: print("Pending queue acknowledge failed:", repr(exc)),
            )
            work_queue.task_done()


def run_chat_worker(deps) -> None:
    while True:
        item, decision = deps.chat_queue.get()
        lifecycle = deps.PendingItemLifecycle(item)
        try:
            deps.process_chat_item(item, decision, lifecycle)
        finally:
            lifecycle.acknowledge(
                deps.delete_pending_message,
                lambda exc: print(
                    "Chat pending queue acknowledge failed:",
                    repr(exc),
                ),
            )
            deps.chat_queue.task_done()
