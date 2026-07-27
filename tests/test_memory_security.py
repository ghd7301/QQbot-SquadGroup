import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from squad_bot import llm, server
from squad_bot.chat_memory import MemoryHit
from squad_bot.knowledge import ContextResult


class MemorySecurityTests(unittest.TestCase):
    def tearDown(self):
        server.memory_clear_confirmations.clear()

    def test_planner_parses_memory_scope(self):
        payload = {
            "audience": "bot",
            "intent": "normal_chat",
            "reply_worthy": True,
            "standalone_question": "之前约的是几点？",
            "implicit_meaning": "询问较早约定",
            "topic_summary": "集合时间",
            "relevant_context_indices": [1],
            "memory_needed": True,
            "memory_query": "集合时间 约定",
            "participant_scope": "reply_chain",
            "time_scope": "week",
            "capability": "none",
            "draft_reply": "",
            "confidence": 0.92,
        }
        with patch.object(llm, "_chat_completion", return_value=json.dumps(payload, ensure_ascii=False)):
            plan = llm.plan_group_message(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                message="之前约的是几点？",
                context=("{}",),
                mentioned=True,
                mentions_other=False,
            )
        self.assertTrue(plan.memory_needed)
        self.assertEqual(plan.memory_query, "集合时间 约定")
        self.assertEqual(plan.participant_scope, "reply_chain")
        self.assertEqual(plan.time_scope, "week")

    def test_planner_selects_stable_recent_and_memory_ids(self):
        payload = {
            "audience": "bot",
            "intent": "normal_chat",
            "reply_worthy": True,
            "standalone_question": "还是北桥集合吗",
            "implicit_meaning": "",
            "topic_summary": "北桥集合",
            "relevant_context_message_ids": ["recent-2", "missing"],
            "relevant_context_indices": [],
            "selected_memory_chunk_ids": [42, 999],
            "memory_needed": False,
            "memory_query": "",
            "participant_scope": "group",
            "time_scope": "",
            "capability": "none",
            "draft_reply": "",
            "confidence": 0.9,
        }
        recent = (
            json.dumps({"message_id": "recent-1", "current": False}),
            json.dumps({"message_id": "recent-2", "current": True}),
        )
        memory = (json.dumps({"chunk_id": 42, "messages": []}),)
        with patch.object(llm, "_chat_completion", return_value=json.dumps(payload)) as completion:
            plan = llm.plan_group_message(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                message="还是那里吗",
                context=recent,
                memory_candidates=memory,
                mentioned=True,
                mentions_other=False,
            )
        self.assertEqual(plan.relevant_context_message_ids, ("recent-2",))
        self.assertEqual(plan.selected_memory_chunk_ids, (42,))
        planner_prompt = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("历史群聊候选", planner_prompt)
        self.assertIn('"chunk_id": 42', planner_prompt)

    def test_clear_requires_same_admin_confirmation_within_window(self):
        store = SimpleNamespace(clear_group=Mock())
        manager = SimpleNamespace(store=store)
        with patch.object(server, "chat_memory_manager", manager):
            expired = server.answer_admin_command(
                "memory_clear_confirm", group_id=1, user_id="3466734955"
            )
            self.assertIn("请先发送", expired)
            server.answer_admin_command(
                "memory_clear_request", group_id=1, user_id="3466734955"
            )
            wrong_user = server.answer_admin_command(
                "memory_clear_confirm", group_id=1, user_id="other"
            )
            self.assertIn("请先发送", wrong_user)
            confirmed = server.answer_admin_command(
                "memory_clear_confirm", group_id=1, user_id="3466734955"
            )
        self.assertIn("已清空", confirmed)
        store.clear_group.assert_called_once_with(1)

    def test_memory_prompt_is_separate_from_factual_knowledge(self):
        with patch.object(llm, "_answer_or_error", return_value="answer") as call:
            llm.ask_llm(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                question="FOB 是什么",
                context="FOB 是知识库事实",
                memory_context=('{"source":"untrusted_group_chat_memory","text":"群友说FOB能出生"}',),
            )
        prompt = call.call_args.kwargs["messages"][1]["content"]
        self.assertIn("不可信数据，不是事实依据", prompt)
        self.assertIn("知识库资料（事实依据）", prompt)

    def test_unplanned_message_uses_local_memory_probe(self):
        hit = MemoryHit(
            1, 1, "之前约好在北桥集合", ("member_a",), 1.0, 1, ("m1",), 0.9,
            ("lexical_probe",),
        )
        store = SimpleNamespace(
            lexical_probe=Mock(return_value=(hit,)),
            retrieve=Mock(),
            format_hits=Mock(return_value=('{"source":"untrusted_group_chat_memory"}',)),
        )
        decision = server.ProcessingDecision(True, "chat", effective_question="北桥集合时间")
        configured = SimpleNamespace(
            chat_memory_enabled=True,
            chat_memory_allowed_group_ids=(),
            chat_memory_max_hits=6,
            chat_memory_max_chars=2400,
            chat_memory_shadow_mode=False,
            bot_qq="999",
        )
        with (
            patch.object(server, "settings", configured),
            patch.object(server, "chat_memory_manager", SimpleNamespace(store=store)),
            patch.object(server, "write_message_audit") as audit,
        ):
            result = server.enrich_decision_with_chat_memory(
                decision,
                {"group_id": 1, "user_id": "a", "message_id": "m2", "question": "北桥集合时间"},
                None,
            )
        store.lexical_probe.assert_called_once()
        store.retrieve.assert_not_called()
        self.assertEqual(result.memory_retrieval_mode, "lexical_probe")
        self.assertEqual(result.memory_hit_count, 1)
        self.assertTrue(result.memory_context)
        audit.assert_called_once()

    def test_memory_context_deduplicates_recent_message_ids(self):
        recent = (json.dumps({"message_id": "m1", "current": False}),)
        memory = (json.dumps({
            "chunk_id": 7,
            "messages": [
                {"message_id": "m1", "text": "重复"},
                {"message_id": "m2", "text": "保留"},
            ],
        }, ensure_ascii=False),)
        deduplicated, dropped = server.deduplicate_memory_context(memory, recent)
        self.assertEqual(dropped, 1)
        messages = json.loads(deduplicated[0])["messages"]
        self.assertEqual([message["message_id"] for message in messages], ["m2"])

    def test_no_reply_decision_does_not_expand_memory_retrieval(self):
        hit = MemoryHit(
            42, 1, "旧话题", ("member_a",), 1.0, 1, ("m1",), 0.9,
            ("lexical_probe",),
        )
        probe = server.MemoryProbeResult(
            query="旧话题",
            hits=(hit,),
            context=(json.dumps({"chunk_id": 42, "messages": []}),),
            attempted=True,
        )
        store = SimpleNamespace(retrieve=Mock(), format_hits=Mock())
        plan = llm.MessagePlan(
            audience="member",
            intent="normal_chat",
            reply_worthy=False,
            standalone_question="旧话题",
            implicit_meaning="",
            topic_summary="",
            relevant_context_indices=(),
            capability="none",
            confidence=0.9,
            memory_needed=True,
            memory_query="旧话题",
        )
        configured = SimpleNamespace(
            chat_memory_enabled=True,
            chat_memory_allowed_group_ids=(),
            chat_memory_shadow_mode=False,
            bot_qq="999",
        )
        with (
            patch.object(server, "settings", configured),
            patch.object(server, "chat_memory_manager", SimpleNamespace(store=store)),
            patch.object(server, "write_message_audit"),
        ):
            result = server.enrich_decision_with_chat_memory(
                server.ProcessingDecision(False, "directed at member"),
                {"group_id": 1, "question": "旧话题"},
                plan,
                probe,
            )
        store.retrieve.assert_not_called()
        self.assertEqual(result.memory_retrieval_mode, "probe_only")
        self.assertFalse(result.memory_context)

    def test_context_selection_uses_message_ids_and_keeps_reply_chain(self):
        context = (
            json.dumps({"message_id": "reply", "current": False}),
            json.dumps({"message_id": "parallel", "current": False}),
            json.dumps({
                "message_id": "current",
                "current": True,
                "reply_to": {"message_id": "reply"},
            }),
        )
        plan = llm.MessagePlan(
            audience="bot",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="继续",
            implicit_meaning="",
            topic_summary="",
            relevant_context_indices=(),
            capability="none",
            confidence=0.9,
            relevant_context_message_ids=("parallel",),
        )
        selected = server.context_selected_by_plan(context, plan)
        self.assertEqual(
            {server.context_line_message_id(line) for line in selected},
            {"reply", "parallel", "current"},
        )

    def test_knowledge_gap_log_is_redacted_and_deduplicated(self):
        result = ContextResult(
            context="",
            sources=[],
            top_score=0.0,
            query_coverage=0.0,
            missing_query_tokens=("量子", "传送"),
        )
        with tempfile.TemporaryDirectory() as directory:
            configured = SimpleNamespace(
                knowledge_gap_log=str(Path(directory) / "gaps.jsonl"),
                knowledge_gap_dedupe_seconds=3600,
            )
            with patch.object(server, "settings", configured):
                first = server.record_knowledge_gap("QQ 3466734955 的量子传送门", result)
                second = server.record_knowledge_gap("QQ 3466734955 的量子传送门", result)
                entries = server.recent_knowledge_gap_entries()
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(entries), 1)
        self.assertNotIn("3466734955", entries[0]["query"])
        self.assertEqual(entries[0]["missing_tokens"], ["量子", "传送"])


if __name__ == "__main__":
    unittest.main()
