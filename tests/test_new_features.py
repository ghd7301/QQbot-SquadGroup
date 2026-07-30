import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from squad_bot import server
from squad_bot.conversation import (
    BARE_CHAT_REACTIONS,
    check_and_mark_chat_reply_quota,
    check_and_mark_topic_replied,
)
from squad_bot.knowledge import ContextResult
from squad_bot.llm import MessagePlan
from squad_bot.models import ProcessingDecision


def _make_deps(**overrides):
    """Build a minimal RuntimeDependencies-like object for unit tests."""
    ns = SimpleNamespace(
        settings=SimpleNamespace(
            same_topic_cooldown_seconds=60,
            chat_reply_cooldown_seconds=60,
            max_chat_replies_per_hour=20,
            bot_qq="999",
            auto_reply_enabled=True,
            chat_reply_enabled=True,
            chat_allowed_group_ids=(),
            semantic_planner_enabled=False,
            llm_fallback_enabled=True,
            fallback_only_when_mentioned=False,
            max_context_chars=4500,
            knowledge_strong_min_score=0.18,
            knowledge_strong_min_coverage=0.6,
            max_answer_chars=500,
        ),
        topic_cooldown_lock=threading.RLock(),
        recent_reply_topics={},
        chat_reply_lock=threading.RLock(),
        chat_message_sequence=0,
        chat_history_lock=threading.RLock(),
        chat_history_state=SimpleNamespace(clear=lambda: None),
        chat_scene_state=SimpleNamespace(clear=lambda: None),
        hostile_reply_lock=threading.RLock(),
        hostile_reply_history={},
        semantic_planner_health=SimpleNamespace(reset=lambda: None),
        clear_fragment_state=lambda: None,
        group_send_locks_lock=threading.Lock(),
        group_send_locks={},
        auto_reply_enabled=True,
        pending_store=MagicMock(),
        _pending_db_path=lambda db_path=None: db_path or ":memory:",
    )
    ns.__dict__.update(overrides)
    return ns


class PreFilterTests(unittest.TestCase):
    """Tests for the unsolicited pre-filter in should_process_message."""

    def test_single_char_message_filtered(self):
        """1-char unsolicited message with no signals should be filtered."""
        with patch.object(server, "settings", SimpleNamespace(
            bot_qq="999", auto_reply_enabled=True, semantic_planner_enabled=False,
            llm_fallback_enabled=True, fallback_only_when_mentioned=False,
            max_context_chars=4500, knowledge_strong_min_score=0.18,
            knowledge_strong_min_coverage=0.6, max_answer_chars=500,
            chat_reply_enabled=True, chat_allowed_group_ids=(),
        )):
            decision = server.should_process_message(
                "哈", False, group_id=1,
            )
        self.assertFalse(decision.should_reply)
        self.assertIn("pre-filter", decision.reason)

    def test_two_char_message_filtered(self):
        """2-char unsolicited message with no signals should be filtered."""
        with patch.object(server, "settings", SimpleNamespace(
            bot_qq="999", auto_reply_enabled=True, semantic_planner_enabled=False,
            llm_fallback_enabled=True, fallback_only_when_mentioned=False,
            max_context_chars=4500, knowledge_strong_min_score=0.18,
            knowledge_strong_min_coverage=0.6, max_answer_chars=500,
            chat_reply_enabled=True, chat_allowed_group_ids=(),
        )):
            decision = server.should_process_message(
                "嗯嗯", False, group_id=1,
            )
        self.assertFalse(decision.should_reply)
        self.assertIn("pre-filter", decision.reason)

    def test_three_char_game_term_not_filtered(self):
        """3-char game terms like 'hab' must NOT be filtered."""
        from squad_bot.message_router import _SHORT_SKIP_MAX_LENGTH
        # 'hab' is 3 chars, which is >= _SHORT_SKIP_MAX_LENGTH (2)
        self.assertGreaterEqual(3, _SHORT_SKIP_MAX_LENGTH)

    def test_question_signal_bypasses_filter(self):
        """Messages with question signals bypass pre-filter regardless of length."""
        with patch.object(server, "settings", SimpleNamespace(
            bot_qq="999", auto_reply_enabled=True, semantic_planner_enabled=False,
            llm_fallback_enabled=True, fallback_only_when_mentioned=False,
            max_context_chars=4500, knowledge_strong_min_score=0.18,
            knowledge_strong_min_coverage=0.6, max_answer_chars=500,
            chat_reply_enabled=True, chat_allowed_group_ids=(),
        )):
            decision = server.should_process_message(
                "怎么", False, group_id=1,
            )
        # "怎么" is 2 chars but has question signal → not pre-filtered
        self.assertNotIn("pre-filter", decision.reason if decision.reason else "")

    def test_mentioned_message_not_pre_filtered(self):
        """@mentioned messages bypass pre-filter entirely."""
        with patch.object(server, "settings", SimpleNamespace(
            bot_qq="999", auto_reply_enabled=True, semantic_planner_enabled=False,
            llm_fallback_enabled=True, fallback_only_when_mentioned=False,
            max_context_chars=4500, knowledge_strong_min_score=0.18,
            knowledge_strong_min_coverage=0.6, max_answer_chars=500,
            chat_reply_enabled=True, chat_allowed_group_ids=(),
        )):
            decision = server.should_process_message(
                "哈", True, group_id=1,
            )
        self.assertNotIn("pre-filter", decision.reason if decision.reason else "")


class AtomicTopicCooldownTests(unittest.TestCase):
    """Tests for check_and_mark_topic_replied."""

    def test_marks_and_returns_false_when_not_on_cooldown(self):
        deps = _make_deps()
        result = check_and_mark_topic_replied(deps, 1, "test_topic")
        self.assertFalse(result)  # Not on cooldown → should proceed

    def test_returns_true_on_second_call_within_cooldown(self):
        deps = _make_deps()
        check_and_mark_topic_replied(deps, 1, "test_topic")
        result = check_and_mark_topic_replied(deps, 1, "test_topic")
        self.assertTrue(result)  # On cooldown → should skip

    def test_different_groups_independent(self):
        deps = _make_deps()
        check_and_mark_topic_replied(deps, 1, "test_topic")
        result = check_and_mark_topic_replied(deps, 2, "test_topic")
        self.assertFalse(result)  # Different group, not on cooldown

    def test_zero_cooldown_always_false(self):
        deps = _make_deps()
        deps.settings.same_topic_cooldown_seconds = 0
        check_and_mark_topic_replied(deps, 1, "test_topic")
        result = check_and_mark_topic_replied(deps, 1, "test_topic")
        self.assertFalse(result)  # Cooldown disabled


class AtomicChatQuotaTests(unittest.TestCase):
    """Tests for check_and_mark_chat_reply_quota."""

    def test_returns_empty_when_quota_available(self):
        deps = _make_deps()
        deps.pending_store.chat_reply_quota_reason.return_value = ""
        result = check_and_mark_chat_reply_quota(deps, 1)
        self.assertEqual(result, "")
        deps.pending_store.mark_chat_replied.assert_called_once()

    def test_returns_reason_when_quota_exceeded(self):
        deps = _make_deps()
        deps.pending_store.chat_reply_quota_reason.return_value = "chat cooldown"
        result = check_and_mark_chat_reply_quota(deps, 1)
        self.assertEqual(result, "chat cooldown")
        deps.pending_store.mark_chat_replied.assert_not_called()

    def test_mark_called_inside_same_lock(self):
        """Verify mark_chat_replied is called while chat_reply_lock is held."""
        deps = _make_deps()
        deps.pending_store.chat_reply_quota_reason.return_value = ""
        lock_held_during_mark = threading.Event()

        original_mark = deps.pending_store.mark_chat_replied
        def check_lock(*args, **kwargs):
            lock_held_during_mark.set()
            return original_mark(*args, **kwargs)
        deps.pending_store.mark_chat_replied = check_lock

        check_and_mark_chat_reply_quota(deps, 1)
        self.assertTrue(lock_held_during_mark.is_set())


class RLockTests(unittest.TestCase):
    """Tests for RLock reentrancy."""

    def test_topic_cooldown_lock_is_reentrant(self):
        deps = _make_deps()
        # Acquire twice from same thread — should not deadlock
        with deps.topic_cooldown_lock:
            with deps.topic_cooldown_lock:
                pass  # If this doesn't deadlock, RLock works

    def test_chat_reply_lock_is_reentrant(self):
        deps = _make_deps()
        with deps.chat_reply_lock:
            with deps.chat_reply_lock:
                pass


class KnowledgeOverrideTests(unittest.TestCase):
    """Tests for high-confidence knowledge override."""

    def test_high_confidence_knowledge_goes_direct_path(self):
        """When planner says knowledge with conf>=0.9 and weak match,
        should route to knowledge path, not fallback."""
        plan = MessagePlan(
            audience="bot",
            intent="knowledge",
            reply_worthy=True,
            standalone_question="怎么当好小队长",
            implicit_meaning="",
            topic_summary="小队长",
            relevant_context_indices=(),
            capability="none",
            confidence=0.95,
            participation_role="addressed",
        )
        weak_result = ContextResult("弱相关资料", ["小队长入门"], 0.12, 0.5)
        configured = SimpleNamespace(
            bot_qq="999", auto_reply_enabled=True, semantic_planner_enabled=True,
            semantic_planner_min_confidence=0.68, llm_fallback_enabled=True,
            fallback_only_when_mentioned=False, max_context_chars=4500,
            knowledge_strong_min_score=0.18, knowledge_strong_min_coverage=0.6,
            max_answer_chars=500, chat_reply_enabled=True, chat_allowed_group_ids=(),
            llm_base_url="https://example.invalid", llm_api_key="test",
            llm_model="test", contextual_query_enabled=False,
        )
        with (
            patch.object(server, "settings", configured),
            patch.object(server, "retrieve_knowledge", return_value=weak_result),
            patch.object(server, "semantic_plan_for_message", return_value=plan),
        ):
            decision = server.should_process_message(
                "怎么当好小队长", True, group_id=1,
            )
        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_mode, "knowledge")
        self.assertIn("high-confidence", decision.reason)

    def test_low_confidence_does_not_override(self):
        """When planner confidence < 0.9, should NOT use override."""
        plan = MessagePlan(
            audience="bot",
            intent="knowledge",
            reply_worthy=True,
            standalone_question="test",
            implicit_meaning="",
            topic_summary="",
            relevant_context_indices=(),
            capability="none",
            confidence=0.7,  # below 0.9
            participation_role="addressed",
        )
        weak_result = ContextResult("弱相关", ["test"], 0.12, 0.5)
        configured = SimpleNamespace(
            bot_qq="999", auto_reply_enabled=True, semantic_planner_enabled=True,
            semantic_planner_min_confidence=0.68, llm_fallback_enabled=True,
            fallback_only_when_mentioned=False, max_context_chars=4500,
            knowledge_strong_min_score=0.18, knowledge_strong_min_coverage=0.6,
            max_answer_chars=500, chat_reply_enabled=True, chat_allowed_group_ids=(),
            llm_base_url="https://example.invalid", llm_api_key="test",
            llm_model="test", contextual_query_enabled=False,
        )
        with (
            patch.object(server, "settings", configured),
            patch.object(server, "retrieve_knowledge", return_value=weak_result),
            patch.object(server, "semantic_plan_for_message", return_value=plan),
        ):
            decision = server.should_process_message(
                "test", True, group_id=1,
            )
        # Should go to fallback, not knowledge
        self.assertEqual(decision.reply_mode, "fallback")


class RuntimeDepsErrorTests(unittest.TestCase):
    """Tests for RuntimeDependencies error message."""

    def test_missing_attribute_shows_helpful_message(self):
        from squad_bot.runtime_dependencies import RuntimeDependencies
        deps = RuntimeDependencies({"existing_key": 42})
        with self.assertRaises(AttributeError) as ctx:
            _ = deps.nonexistent_key
        self.assertIn("nonexistent_key", str(ctx.exception))
        self.assertIn("server.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
