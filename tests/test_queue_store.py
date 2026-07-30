import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from squad_bot import pending_store
from squad_bot.queueing import store


class QueueStoreTests(unittest.TestCase):
    def test_configured_path_is_used_by_store_facade(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            deps = SimpleNamespace(
                settings=SimpleNamespace(
                    pending_queue_db=str(db_path),
                    pending_retry_max_attempts=2,
                ),
                pending_store=pending_store,
            )

            pending_id = store.persist_pending_message(
                deps,
                0,
                1,
                {"question": "测试消息"},
            )
            loaded = store.load_pending_messages(deps)
            failure = store.mark_pending_failure(
                deps,
                pending_id,
                "temporary",
                now=100,
            )

        self.assertEqual(loaded[0][2]["question"], "测试消息")
        self.assertEqual(failure.status, "retry")
        self.assertEqual(failure.next_attempt_at, 101)


if __name__ == "__main__":
    unittest.main()
