import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from squad_bot import server
from squad_bot.pending_store import (
    cleanup_stale_pending_messages,
    find_queued_duplicates,
    is_pending_superseded,
    mark_pending_dispatch_started,
    mark_pending_superseded,
    open_pending_queue_db,
    persist_pending_message,
)


class SupersededTests(unittest.TestCase):
    """Tests for the superseded dedup mechanism."""

    def _setup_db(self, temp_dir):
        db_path = Path(temp_dir) / "pending.sqlite3"
        return str(db_path)

    def test_mark_and_check_superseded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = self._setup_db(temp_dir)
            pid = persist_pending_message(
                1, 1, {"question": "test", "group_id": 1, "message_id": "m1"},
                db_path=db_path,
            )
            self.assertFalse(is_pending_superseded(pid, db_path=db_path))
            mark_pending_superseded(pid, db_path=db_path)
            self.assertTrue(is_pending_superseded(pid, db_path=db_path))

    def test_find_queued_duplicates_by_message_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = self._setup_db(temp_dir)
            persist_pending_message(
                1, 1,
                {"question": "q1", "group_id": 1, "message_id": "m1", "message_ids": ["m1"]},
                db_path=db_path,
            )
            # Same group, overlapping message_id
            dupes = find_queued_duplicates(1, ["m1"], db_path=db_path)
            self.assertEqual(len(dupes), 1)
            # Different group
            dupes = find_queued_duplicates(2, ["m1"], db_path=db_path)
            self.assertEqual(len(dupes), 0)
            # Different message_id
            dupes = find_queued_duplicates(1, ["m99"], db_path=db_path)
            self.assertEqual(len(dupes), 0)

    def test_find_queued_duplicates_includes_dispatching(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = self._setup_db(temp_dir)
            pid = persist_pending_message(
                1, 1,
                {"question": "q1", "group_id": 1, "message_id": "m1", "message_ids": ["m1"]},
                db_path=db_path,
            )
            mark_pending_dispatch_started(pid, "d1", db_path=db_path)
            # Should still find it even though status is 'dispatching'
            dupes = find_queued_duplicates(1, ["m1"], db_path=db_path)
            self.assertEqual(len(dupes), 1)

    def test_superseded_entry_excluded_from_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = self._setup_db(temp_dir)
            pid = persist_pending_message(
                1, 1, {"question": "test", "group_id": 1},
                db_path=db_path,
            )
            mark_pending_superseded(pid, db_path=db_path)
            from squad_bot.pending_store import load_pending_messages
            pending = load_pending_messages(db_path=db_path)
            self.assertEqual(len(pending), 0)


class CleanupTests(unittest.TestCase):
    """Tests for stale pending message cleanup."""

    def test_cleanup_removes_old_dead_letter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "pending.sqlite3")
            conn = open_pending_queue_db(db_path)
            conn.execute(
                "INSERT INTO pending_messages (priority, sequence, payload, created_at, status) "
                "VALUES (1, 1, '{}', ?, 'dead_letter')",
                (time.time() - 86400 * 4,),  # 4 days old
            )
            conn.commit()
            conn.close()
            cleaned = cleanup_stale_pending_messages(db_path=db_path, max_age_hours=72)
            self.assertEqual(cleaned, 1)

    def test_cleanup_preserves_recent_dead_letter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "pending.sqlite3")
            conn = open_pending_queue_db(db_path)
            conn.execute(
                "INSERT INTO pending_messages (priority, sequence, payload, created_at, status) "
                "VALUES (1, 1, '{}', ?, 'dead_letter')",
                (time.time() - 3600,),  # 1 hour old
            )
            conn.commit()
            conn.close()
            cleaned = cleanup_stale_pending_messages(db_path=db_path, max_age_hours=72)
            self.assertEqual(cleaned, 0)

    def test_cleanup_preserves_queued_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "pending.sqlite3")
            conn = open_pending_queue_db(db_path)
            conn.execute(
                "INSERT INTO pending_messages (priority, sequence, payload, created_at, status) "
                "VALUES (1, 1, '{}', ?, 'queued')",
                (time.time() - 86400 * 10,),  # 10 days old but queued
            )
            conn.commit()
            conn.close()
            cleaned = cleanup_stale_pending_messages(db_path=db_path, max_age_hours=72)
            self.assertEqual(cleaned, 0)


class IsModelErrorAnswerTests(unittest.TestCase):
    """Tests for extended error detection patterns."""

    def test_json_error_body(self):
        from squad_bot.answering import is_model_error_answer
        self.assertTrue(is_model_error_answer('{"error": "bad request"}'))
        self.assertTrue(is_model_error_answer('[{"message": "failed"}]'))

    def test_api_key_error(self):
        from squad_bot.answering import is_model_error_answer
        self.assertTrue(is_model_error_answer("Invalid API key provided"))
        self.assertTrue(is_model_error_answer("api key is required"))

    def test_rate_limit_error(self):
        from squad_bot.answering import is_model_error_answer
        self.assertTrue(is_model_error_answer("Rate limit exceeded, please retry"))

    def test_normal_answer_not_detected(self):
        from squad_bot.answering import is_model_error_answer
        self.assertFalse(is_model_error_answer("FOB就是兵站，用来部署和补给"))
        self.assertFalse(is_model_error_answer(""))

    def test_original_prefixes_still_work(self):
        from squad_bot.answering import is_model_error_answer
        self.assertTrue(is_model_error_answer("模型接口返回了错误"))
        self.assertTrue(is_model_error_answer("还没有配置模型 API Key"))


class DeterministicReviewFailureTests(unittest.TestCase):
    """Tests for deterministic_review_failure_answer edge cases."""

    def test_bot_meta_returns_generic_capability_info(self):
        from squad_bot.answering import deterministic_review_failure_answer
        from squad_bot.models import ProcessingDecision
        from unittest.mock import MagicMock
        deps = MagicMock()
        decision = ProcessingDecision(
            True, "bot_meta timeout", reply_mode="bot_meta",
            semantic_intent="bot_meta",
        )
        result = deterministic_review_failure_answer(deps, decision, "", mentioned=True)
        self.assertIn("知识库", result)
        self.assertNotIn("没判断清楚", result)

    def test_control_attempt_returns_control_message(self):
        from squad_bot.answering import deterministic_review_failure_answer
        from squad_bot.models import ProcessingDecision
        from unittest.mock import MagicMock
        deps = MagicMock()
        decision = ProcessingDecision(
            True, "control", reply_mode="fallback",
            semantic_intent="control_attempt",
        )
        result = deterministic_review_failure_answer(deps, decision, "", mentioned=True)
        self.assertIn("不能通过", result)


if __name__ == "__main__":
    unittest.main()
