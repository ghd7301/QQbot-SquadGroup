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

    def test_server_passes_scene_snapshot_to_semantic_planner(self):
        configured = SimpleNamespace(
            semantic_planner_enabled=True,
            bot_qq="999",
            llm_base_url="https://example.invalid",
            llm_api_key="key",
            llm_model="model",
            semantic_planner_model="planner",
            semantic_planner_timeout_seconds=4,
        )
        with (
            patch.object(server, "settings", configured),
            patch.object(server, "plan_group_message") as planner,
        ):
            server.semantic_plan_for_message(
                "我又优化了，大家再试试",
                ('{"message_id":"current","current":true}',),
                scene_context='{"topics":[{"summary":"当前机器人优化"}]}',
                mentioned=False,
                mentions_other=False,
            )

        self.assertEqual(
            planner.call_args.kwargs["scene_context"],
            '{"topics":[{"summary":"当前机器人优化"}]}',
        )

    def test_planner_parses_multiple_topic_candidates_with_real_anchors(self):
        payload = {
            "audience": "group",
            "intent": "normal_chat",
            "reply_worthy": True,
            "standalone_question": "边吃火锅边聊 IDE",
            "implicit_meaning": "连接两个并行话题",
            "topic_summary": "聚餐与 IDE 讨论交汇",
            "topic_candidates": [
                {
                    "key": "ide",
                    "label": "IDE 选择",
                    "confidence": 0.88,
                    "anchor_message_ids": ["m1", "missing"],
                    "basis": "bridge",
                },
                {
                    "key": "dinner",
                    "label": "周五聚餐",
                    "confidence": 0.84,
                    "anchor_message_ids": ["m2"],
                    "basis": "bridge",
                },
            ],
            "capability": "none",
            "draft_reply": "",
            "confidence": 0.9,
        }
        context = (
            json.dumps({"message_id": "m1", "current": False}),
            json.dumps({"message_id": "m2", "current": True}),
        )
        with patch.object(llm, "_chat_completion", return_value=json.dumps(payload)):
            plan = llm.plan_group_message(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                message="边吃边聊这个",
                context=context,
                mentioned=False,
                mentions_other=False,
            )

        self.assertEqual(len(plan.topic_candidates), 2)
        self.assertEqual(plan.topic_candidates[0].anchor_message_ids, ("m1",))
        self.assertEqual(plan.topic_candidates[1].label, "周五聚餐")
        self.assertEqual(plan.topic_candidates[1].basis, "bridge")

    def test_planner_parses_subject_candidates_and_scene_evidence(self):
        payload = {
            "audience": "group",
            "intent": "normal_chat",
            "reply_worthy": True,
            "standalone_question": "我又优化了机器人，大家再试试",
            "implicit_meaning": "邀请群友继续测试当前机器人",
            "topic_summary": "机器人优化测试",
            "subject_candidates": [
                {
                    "entity_type": "bot",
                    "entity_id": "invented",
                    "label": "当前机器人",
                    "confidence": 0.92,
                    "evidence_message_ids": ["bot-turn", "scene-anchor", "missing"],
                }
            ],
            "subject_ambiguity": "clear",
            "capability": "none",
            "draft_reply": "行，拿我再测几轮。",
            "confidence": 0.94,
        }
        context = (
            json.dumps({
                "message_id": "bot-turn",
                "speaker": {"id": "bot", "role": "bot"},
            }),
            json.dumps({
                "message_id": "current",
                "current": True,
                "speaker": {"id": "member_owner", "role": "member"},
            }),
        )
        scene = json.dumps({
            "topics": [{
                "participants": ["bot", "member_owner"],
                "anchor_message_ids": ["scene-anchor"],
            }]
        })
        with patch.object(llm, "_chat_completion", return_value=json.dumps(payload)) as completion:
            plan = llm.plan_group_message(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                message="我又做了优化，大伙还可以再来试试",
                context=context,
                scene_context=scene,
                mentioned=False,
                mentions_other=False,
            )

        self.assertEqual(plan.subject_ambiguity, "clear")
        self.assertEqual(plan.subject_candidates[0].entity_type, "bot")
        self.assertEqual(plan.subject_candidates[0].entity_id, "bot")
        self.assertEqual(
            plan.subject_candidates[0].evidence_message_ids,
            ("bot-turn", "scene-anchor"),
        )
        planner_input = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("滚动场景快照", planner_input)
        self.assertIn("scene-anchor", planner_input)

    def test_subject_relation_derives_safe_reply_perspective(self):
        bot_subject = llm.MessagePlan(
            audience="group",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="继续测试当前机器人",
            implicit_meaning="",
            topic_summary="机器人测试",
            relevant_context_indices=(),
            capability="none",
            confidence=0.9,
            subject_candidates=(
                llm.SubjectCandidate(
                    "bot",
                    "当前机器人",
                    0.92,
                    evidence_message_ids=("bot-turn",),
                ),
            ),
            subject_ambiguity="clear",
        )
        external_bot = llm.MessagePlan(
            audience="group",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="学校群里的机器人上下文很好",
            implicit_meaning="",
            topic_summary="外部机器人",
            relevant_context_indices=(),
            capability="none",
            confidence=0.9,
            subject_candidates=(
                llm.SubjectCandidate("external_project", "学校群机器人", 0.94),
            ),
            subject_ambiguity="clear",
        )
        ambiguous = llm.MessagePlan(
            audience="group",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="又优化了",
            implicit_meaning="",
            topic_summary="不明确的优化对象",
            relevant_context_indices=(),
            capability="none",
            confidence=0.8,
            subject_candidates=(
                llm.SubjectCandidate("external_project", "某个项目", 0.65),
                llm.SubjectCandidate("bot", "当前机器人", 0.6),
            ),
            subject_ambiguity="ambiguous",
        )

        self.assertEqual(
            server.derive_bot_reply_perspective(bot_subject, ()),
            ("subject", "first_person"),
        )
        self.assertEqual(
            server.derive_bot_reply_perspective(external_bot, ()),
            ("observer", "observer"),
        )
        self.assertEqual(
            server.derive_bot_reply_perspective(ambiguous, ()),
            ("uncertain", "neutral"),
        )

    def test_possible_bot_subject_preserves_recent_bot_turn(self):
        plan = llm.MessagePlan(
            audience="group",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="再测试当前机器人",
            implicit_meaning="",
            topic_summary="机器人测试",
            relevant_context_indices=(),
            capability="none",
            confidence=0.9,
            relevant_context_message_ids=("current",),
            subject_candidates=(
                llm.SubjectCandidate("bot", "当前机器人", 0.6),
            ),
            subject_ambiguity="ambiguous",
        )
        bot_line = json.dumps({
            "message_id": "bot-turn",
            "speaker": {"id": "bot", "role": "bot"},
        })
        current_line = json.dumps({
            "message_id": "current",
            "current": True,
            "speaker": {"id": "member_owner", "role": "member"},
        })

        selected = server.context_selected_by_plan((bot_line, current_line), plan)

        self.assertEqual(selected, (bot_line, current_line))

    def test_ambiguous_subject_clears_committed_planner_draft(self):
        plan = llm.MessagePlan(
            audience="group",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="又优化了，大家再试试",
            implicit_meaning="优化对象不明确",
            topic_summary="项目优化",
            relevant_context_indices=(),
            capability="none",
            confidence=0.85,
            draft_reply="好，我待会儿去试试看。",
            subject_candidates=(
                llm.SubjectCandidate("external_project", "某个项目", 0.65),
                llm.SubjectCandidate("bot", "当前机器人", 0.6),
            ),
            subject_ambiguity="ambiguous",
        )
        decision = server.ProcessingDecision(
            True,
            "semantic plan: chat candidate",
            reply_mode="chat",
            draft_reply=plan.draft_reply,
        )

        server.apply_semantic_plan_metadata(decision, plan)

        self.assertEqual(decision.bot_involvement, "uncertain")
        self.assertEqual(decision.reply_perspective, "neutral")
        self.assertEqual(decision.draft_reply, "")
        self.assertIn("self_identity", decision.risk_flags)

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
            format_self_history=Mock(return_value=()),
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
        store.format_self_history.assert_called_once()
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
        store = SimpleNamespace(retrieve=Mock(), format_hits=Mock(), format_self_history=Mock())
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
        store.format_self_history.assert_not_called()
        self.assertEqual(result.memory_retrieval_mode, "probe_only")
        self.assertFalse(result.memory_context)

    def test_self_history_prompt_keeps_authorship_separate_from_facts(self):
        self_history = (
            '{"source":"bot_self_history","bot_message":{"speaker":{"role":"bot","is_self":true},"text":"旧回复"},"generated_for_message_ids":["m1"]}',
        )
        with patch.object(llm, "_answer_or_error", return_value="answer") as call:
            llm.ask_llm(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                question="继续说",
                context="知识事实",
                self_history_context=self_history,
            )
        prompt = call.call_args.kwargs["messages"][1]["content"]
        self.assertIn("你此前参与当前话题的记录", prompt)
        self.assertIn("作者身份", prompt)
        self.assertIn("旧回复内容不是事实依据", prompt)
        self.assertIn("generated_for_message_ids", llm.FINAL_REPLY_REVIEW_PROMPT)

    def test_group_recruitment_prompts_redirect_to_ts_without_fake_participation(self):
        self.assertIn("向全群公开找人组队", llm.MESSAGE_PLAN_PROMPT)
        self.assertIn("TS 里对应游戏的语音频道", llm.MESSAGE_PLAN_PROMPT)
        self.assertIn("不得回答“有”“我来”“算我一个”", llm.MESSAGE_PLAN_PROMPT)
        self.assertIn("代替真实群友确认“有人”", llm.FINAL_REPLY_REVIEW_PROMPT)
        self.assertIn("这类候选绝不能 send", llm.FINAL_REPLY_REVIEW_PROMPT)
        self.assertIn("有啊，你打哪个版本", llm.FINAL_REPLY_REVIEW_PROMPT)
        self.assertIn("TS 里对应游戏的语音频道", llm.CHAT_PROMPT)

    def test_capability_prompt_requires_bot_meta_intent(self):
        self.assertIn("并且 intent=bot_meta", llm.MESSAGE_PLAN_PROMPT)
        self.assertIn("普通事实问题", llm.MESSAGE_PLAN_PROMPT)
        self.assertIn("一律返回 capability=none", llm.MESSAGE_PLAN_PROMPT)

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

    def test_context_selection_keeps_two_topic_anchors_and_full_reply_chain(self):
        context = (
            json.dumps({"message_id": "root", "reply_to": None}),
            json.dumps({"message_id": "middle", "reply_to": {"message_id": "root"}}),
            json.dumps({"message_id": "other-topic", "reply_to": None}),
            json.dumps({
                "message_id": "current",
                "current": True,
                "reply_to": {"message_id": "middle"},
            }),
        )
        plan = llm.MessagePlan(
            audience="group",
            intent="normal_chat",
            reply_worthy=True,
            standalone_question="连接两个话题",
            implicit_meaning="",
            topic_summary="两个并行话题交汇",
            relevant_context_indices=(),
            capability="none",
            confidence=0.9,
            topic_candidates=(
                llm.SemanticTopicCandidate("one", "第一话题", 1.0, ("middle",), "qq_reply"),
                llm.SemanticTopicCandidate("two", "第二话题", 0.8, ("other-topic",), "bridge"),
            ),
        )

        selected = server.context_selected_by_plan(context, plan)

        self.assertEqual(
            [json.loads(line)["message_id"] for line in selected],
            ["root", "middle", "other-topic", "current"],
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
