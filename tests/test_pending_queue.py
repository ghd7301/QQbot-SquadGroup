import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from squad_bot import server
from squad_bot.server import (
    delete_pending_message,
    is_message_too_old,
    is_restored_admin_command,
    load_pending_messages,
    mark_pending_dispatch_started,
    mark_pending_failure,
    pending_status_counts,
    pending_message_count,
    persist_pending_message,
)


class PendingQueueTests(unittest.TestCase):
    def test_priority_and_normal_messages_use_separate_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            with patch.object(
                server,
                "settings",
                SimpleNamespace(pending_queue_db=str(db_path)),
            ):
                priority_id = server.enqueue_persistent_message(
                    0,
                    {"group_id": 1, "question": "@问题", "mentioned": True},
                )
                normal_id = server.enqueue_persistent_message(
                    1,
                    {"group_id": 1, "question": "普通问题", "mentioned": False},
                )

                priority_item = server.message_queue.get_nowait()
                normal_item = server.normal_message_queue.get_nowait()
                server.message_queue.task_done()
                server.normal_message_queue.task_done()

                self.assertEqual(priority_item[2]["_pending_id"], priority_id)
                self.assertEqual(normal_item[2]["_pending_id"], normal_id)
                self.assertTrue(server.message_queue.empty())
                self.assertTrue(server.normal_message_queue.empty())

    def test_pending_message_survives_reopen_until_acknowledged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            item = {
                "group_id": 983063031,
                "question": "医疗兵怎么玩",
                "mentioned": True,
                "time": 1784690900,
                "user_id": 3466734955,
                "sender_role": "member",
            }

            pending_id = persist_pending_message(0, 7, item, db_path)

            self.assertEqual(pending_message_count(db_path), 1)
            loaded = load_pending_messages(db_path)
            self.assertEqual(len(loaded), 1)
            priority, sequence, restored_item = loaded[0]
            self.assertEqual(priority, 0)
            self.assertEqual(sequence, 7)
            self.assertEqual(restored_item["question"], item["question"])
            self.assertEqual(restored_item["_pending_id"], pending_id)
            self.assertTrue(restored_item["_restored"])
            self.assertIsInstance(restored_item["_pending_created_at"], float)

            delete_pending_message(pending_id, db_path)
            self.assertEqual(load_pending_messages(db_path), [])

    def test_existing_pending_database_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.execute(
                """
                CREATE TABLE pending_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    priority INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO pending_messages (priority, sequence, payload, created_at) "
                "VALUES (0, 1, '{\"question\": \"旧消息\"}', 100)"
            )
            connection.commit()
            connection.close()

            loaded = load_pending_messages(db_path, include_future=True)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0][2]["_pending_status"], "queued")
            self.assertEqual(loaded[0][2]["_pending_attempts"], 0)
            self.assertEqual(pending_status_counts(db_path)["queued"], 1)

    def test_pending_failure_retries_then_moves_to_dead_letter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            pending_id = persist_pending_message(
                0,
                7,
                {"group_id": 1, "question": "@问题", "mentioned": True},
                db_path,
            )

            first = mark_pending_failure(
                pending_id,
                "temporary failure",
                db_path=db_path,
                now=100,
                max_attempts=2,
            )
            self.assertEqual(first.status, "retry")
            self.assertEqual(first.attempts, 1)
            self.assertGreater(first.next_attempt_at, 100)

            second = mark_pending_failure(
                pending_id,
                "still failing",
                db_path=db_path,
                now=110,
                max_attempts=2,
            )
            self.assertEqual(second.status, "dead_letter")
            self.assertEqual(second.attempts, 2)
            self.assertEqual(load_pending_messages(db_path, include_future=True), [])
            self.assertEqual(pending_status_counts(db_path)["dead_letter"], 1)
            self.assertEqual(pending_message_count(db_path), 0)

    def test_interrupted_dispatch_becomes_sent_unknown_on_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            pending_id = persist_pending_message(
                0,
                7,
                {"group_id": 1, "question": "@问题", "mentioned": True},
                db_path,
            )
            mark_pending_dispatch_started(pending_id, "dispatch-1", db_path=db_path)

            self.assertEqual(pending_status_counts(db_path)["dispatching"], 1)
            recovered = server.recover_incomplete_pending_dispatches(db_path)

            self.assertEqual(recovered, 1)
            self.assertEqual(pending_status_counts(db_path)["dispatching"], 0)
            self.assertEqual(pending_status_counts(db_path)["sent_unknown"], 1)
            self.assertEqual(load_pending_messages(db_path, include_future=True), [])

    def test_begin_dispatch_persists_marker_before_network_send(self) -> None:
        item = {"_pending_id": 7}
        with patch.object(server, "mark_pending_dispatch_started") as mark_started:
            server.begin_pending_dispatch(item)

        self.assertTrue(item["_dispatch_started"])
        self.assertTrue(item["_pending_dispatch_id"].startswith("7:"))
        mark_started.assert_called_once_with(7, item["_pending_dispatch_id"])

    def test_worker_failure_before_dispatch_requeues_same_priority_item(self) -> None:
        item = {
            "_pending_id": 7,
            "_queue_priority": 0,
            "_queue_sequence": 11,
            "explicit_knowledge_command": True,
            "mentioned": False,
        }
        retry = server.PendingFailureResult("retry", 1, 101.0)
        with (
            patch.object(server, "mark_pending_failure", return_value=retry) as mark_failure,
            patch.object(server, "_queue_pending_item") as requeue,
            patch.object(server.time, "time", return_value=100.0),
        ):
            action = server.handle_pending_worker_failure(item, "planner timeout")

        self.assertEqual(action, "retry")
        mark_failure.assert_called_once_with(7, "planner timeout")
        requeue.assert_called_once_with(item, delay=1.0)

        with patch.object(server.message_queue, "put") as priority_put:
            server._queue_pending_item(item)
        priority_put.assert_called_once()
        self.assertEqual(priority_put.call_args.args[0][:2], (0, 11))

    def test_worker_failure_after_dispatch_started_is_not_retried(self) -> None:
        item = {"_pending_id": 8, "_dispatch_started": True}
        with (
            patch.object(server, "mark_pending_sent_unknown") as mark_unknown,
            patch.object(server, "mark_pending_failure") as mark_failure,
            patch.object(server, "_queue_pending_item") as requeue,
        ):
            action = server.handle_pending_worker_failure(item, "connection reset")

        self.assertEqual(action, "sent_unknown")
        mark_unknown.assert_called_once_with(8, "connection reset")
        mark_failure.assert_not_called()
        requeue.assert_not_called()

    def test_worker_failure_after_dispatch_completed_can_be_acknowledged(self) -> None:
        item = {
            "_pending_id": 9,
            "_dispatch_started": True,
            "_dispatch_completed": True,
        }
        with (
            patch.object(server, "mark_pending_sent_unknown") as mark_unknown,
            patch.object(server, "mark_pending_failure") as mark_failure,
        ):
            action = server.handle_pending_worker_failure(item, "audit write failed")

        self.assertEqual(action, "delivered")
        mark_unknown.assert_not_called()
        mark_failure.assert_not_called()

    def test_restore_pending_message_respects_future_retry_time(self) -> None:
        item = {"_pending_next_attempt_at": 105.0}
        with (
            patch.object(server, "recover_incomplete_pending_dispatches", return_value=0),
            patch.object(server, "load_pending_messages", return_value=[(0, 12, item)]),
            patch.object(server, "_queue_pending_item") as queue_item,
            patch.object(server.time, "time", return_value=100.0),
        ):
            restored = server.restore_pending_messages()

        self.assertEqual(restored, 1)
        queue_item.assert_called_once_with(item, delay=5.0)

    def test_normal_and_mentioned_messages_have_different_age_limits(self) -> None:
        now = 1_000.0

        self.assertFalse(is_message_too_old(now - 60, False, now=now))
        self.assertTrue(is_message_too_old(now - 61, False, now=now))
        self.assertFalse(is_message_too_old(now - 299, True, now=now))
        self.assertTrue(is_message_too_old(now - 301, True, now=now))

    def test_restored_admin_commands_are_not_replayed(self) -> None:
        with patch.object(
            server,
            "settings",
            SimpleNamespace(admin_qq_ids=("3466734955",)),
        ):
            self.assertTrue(
                is_restored_admin_command(
                    {
                        "_restored": True,
                        "question": "重载知识库",
                        "user_id": "3466734955",
                    }
                )
            )
            self.assertFalse(
                is_restored_admin_command(
                    {
                        "_restored": True,
                        "question": "重载知识库",
                        "user_id": "10001",
                        "sender_role": "owner",
                    }
                )
            )
            self.assertFalse(
                is_restored_admin_command(
                    {
                        "_restored": False,
                        "question": "重载知识库",
                        "user_id": "3466734955",
                    }
                )
            )

    def test_group_role_does_not_bypass_admin_whitelist(self) -> None:
        with patch.object(
            server,
            "settings",
            SimpleNamespace(admin_qq_ids=("3466734955",)),
        ):
            self.assertTrue(server.is_admin_user("3466734955", "member"))
            self.assertFalse(server.is_admin_user("10001", "owner"))
            self.assertFalse(server.is_admin_user("10002", "admin"))


if __name__ == "__main__":
    unittest.main()
