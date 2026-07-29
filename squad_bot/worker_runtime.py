from __future__ import annotations

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
