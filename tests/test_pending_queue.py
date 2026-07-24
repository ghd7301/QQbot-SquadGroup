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
            self.assertEqual(pending_message_count(db_path), 0)

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
