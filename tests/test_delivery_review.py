import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from squad_bot.delivery import review
from squad_bot.models import GroupChatMessage, ProcessingDecision


class DeliveryReviewTests(unittest.TestCase):
    def test_locked_validation_stops_already_covered_message(self) -> None:
        covered = Mock(return_value=True)
        deps = SimpleNamespace(message_already_covered_by_bot=covered)

        allowed, reason, note = review.validate_locked_send(
            deps,
            100,
            {"message_id": "m1"},
            7,
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "message covered while waiting to send")
        self.assertEqual(note, "")
        covered.assert_called_once_with(100, "m1")

    def test_reply_safety_rejects_internal_member_id_and_duplicate(self) -> None:
        deps = SimpleNamespace(
            settings=SimpleNamespace(bot_qq="999"),
            chat_history_lock=threading.Lock(),
            group_chat_history={
                100: (GroupChatMessage("重复的机器人回答", "999", 1),)
            },
        )

        self.assertEqual(
            review.unsafe_or_repeated_reply(deps, 100, "member_4f92ac"),
            "internal member id leaked",
        )
        self.assertEqual(
            review.unsafe_or_repeated_reply(deps, 100, "重复的机器人回答"),
            "duplicate recent bot reply",
        )

    def test_remaining_timeout_honors_cap_and_reserve(self) -> None:
        with patch.object(review.time, "monotonic", return_value=100.0):
            self.assertEqual(
                review.remaining_reply_timeout(110.0, cap=20, reserve=3),
                7,
            )
            self.assertEqual(
                review.remaining_reply_timeout(103.0, cap=20, reserve=3),
                0,
            )

    def test_adaptive_review_skips_model_when_context_is_unchanged(self) -> None:
        deps = SimpleNamespace(
            latest_group_user_sequence=lambda _group_id: 7,
            settings=SimpleNamespace(final_reply_review_mode="adaptive"),
            unsafe_or_repeated_reply=lambda _group_id, _answer: "",
        )
        decision = ProcessingDecision(False, "test")

        answer, reason, revision = review.review_and_refresh_answer(
            deps,
            question="问题",
            answer="回答",
            decision=decision,
            group_id=100,
            mentioned=False,
            admin=False,
            deadline=200,
            baseline_revision=7,
        )

        self.assertEqual(answer, "回答")
        self.assertEqual(reason, "adaptive final review skipped")
        self.assertEqual(revision, 7)

    def test_late_context_refresh_returns_without_duplicate_review(self) -> None:
        deps = SimpleNamespace(latest_group_user_sequence=lambda _group_id: 9)

        answer, reason, revision = review.refresh_answer_for_late_context(
            deps,
            question="问题",
            answer="回答",
            decision=ProcessingDecision(False, "test"),
            group_id=100,
            mentioned=True,
            admin=False,
            deadline=200,
            reviewed_revision=9,
        )

        self.assertEqual(answer, "回答")
        self.assertEqual(reason, "context unchanged after final review")
        self.assertEqual(revision, 9)


if __name__ == "__main__":
    unittest.main()
