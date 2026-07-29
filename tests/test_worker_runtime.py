import unittest
from unittest.mock import Mock

from squad_bot.worker_runtime import PendingItemLifecycle, normal_lane_should_yield


class PendingItemLifecycleTests(unittest.TestCase):
    def test_transfer_keeps_pending_item_for_chat_worker(self) -> None:
        delete = Mock()
        lifecycle = PendingItemLifecycle({"_pending_id": 7})
        lifecycle.transfer()

        lifecycle.acknowledge(delete, Mock())

        delete.assert_not_called()

    def test_failure_acknowledges_only_delivered_item(self) -> None:
        delete = Mock()
        lifecycle = PendingItemLifecycle({"_pending_id": "8"})

        action = lifecycle.handle_failure(
            "audit failed",
            lambda item, error: "delivered",
        )
        lifecycle.acknowledge(delete, Mock())

        self.assertEqual(action, "delivered")
        delete.assert_called_once_with(8)

    def test_retry_failure_does_not_delete_pending_item(self) -> None:
        delete = Mock()
        lifecycle = PendingItemLifecycle({"_pending_id": 9})
        lifecycle.handle_failure("timeout", lambda item, error: "retry")

        lifecycle.acknowledge(delete, Mock())

        delete.assert_not_called()

    def test_acknowledge_reports_delete_error(self) -> None:
        error = RuntimeError("database busy")
        on_error = Mock()
        lifecycle = PendingItemLifecycle({"_pending_id": 10})

        lifecycle.acknowledge(Mock(side_effect=error), on_error)

        on_error.assert_called_once_with(error)

    def test_only_normal_lane_yields_to_priority_work(self) -> None:
        self.assertTrue(normal_lane_should_yield("normal", priority_pending=True))
        self.assertFalse(normal_lane_should_yield("priority", priority_pending=True))
        self.assertFalse(normal_lane_should_yield("normal", priority_pending=False))


if __name__ == "__main__":
    unittest.main()
