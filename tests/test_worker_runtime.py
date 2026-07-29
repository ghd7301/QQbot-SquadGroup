import unittest
from unittest.mock import Mock, patch

from squad_bot import server
from squad_bot.models import ProcessingDecision
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


class WorkerItemProcessorTests(unittest.TestCase):
    def test_priority_item_processor_can_finish_without_queue_loop(self) -> None:
        item = {
            "question": "健康状态",
            "mentioned": True,
            "group_id": 1,
            "user_id": "100",
            "sender_role": "member",
            "time": 100,
        }
        with (
            patch.object(server, "reply_deadline", return_value=200),
            patch.object(server, "is_admin_user", return_value=False),
            patch.object(server, "is_restored_admin_command", return_value=True),
            patch.object(server, "write_message_audit") as audit,
        ):
            lifecycle = server.process_worker_item(item, "priority")

        self.assertTrue(lifecycle.terminal)
        self.assertEqual(
            audit.call_args.kwargs["reason"],
            "restored admin command discarded",
        )

    def test_chat_item_processor_can_finish_without_queue_loop(self) -> None:
        item = {
            "question": "已经过期的闲聊",
            "group_id": 1,
            "user_id": "100",
            "time": 100,
        }
        decision = ProcessingDecision(True, "chat", reply_mode="chat")
        with (
            patch.object(server, "reply_deadline", return_value=200),
            patch.object(server, "celebration_kind", return_value=""),
            patch.object(server, "response_mention_user_id", return_value=""),
            patch.object(server, "is_message_too_old", return_value=True),
            patch.object(server, "write_message_audit") as audit,
        ):
            lifecycle = server.process_chat_item(item, decision)

        self.assertTrue(lifecycle.terminal)
        self.assertEqual(audit.call_args.kwargs["reason"], "queued chat message too old")

    def test_item_processor_failure_uses_pending_failure_policy(self) -> None:
        item = {"_pending_id": 9}
        with (
            patch.object(
                server,
                "handle_pending_worker_failure",
                return_value="retry",
            ) as failure,
            patch.object(server, "write_message_audit"),
        ):
            lifecycle = server.process_worker_item(item, "normal")

        self.assertFalse(lifecycle.terminal)
        failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
