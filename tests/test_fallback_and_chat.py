import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from squad_bot import llm, server
from squad_bot.knowledge import ContextResult
from squad_bot.llm import (
    CHAT_PROMPT,
    CHAT_ROUTER_PROMPT,
    FALLBACK_PROMPT,
    SYSTEM_PROMPT,
    is_chat_no_reply,
    normalize_model_answer,
)


def routing_settings(**overrides):
    values = {
        "max_context_chars": 4500,
        "llm_fallback_enabled": True,
        "fallback_only_when_mentioned": True,
        "chat_reply_enabled": True,
        "llm_base_url": "https://example.invalid",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "chat_reply_cooldown_seconds": 180,
        "max_chat_replies_per_hour": 8,
        "chat_allowed_group_ids": (),
        "knowledge_strong_min_score": 0.18,
        "knowledge_strong_min_coverage": 0.6,
        "max_answer_chars": 500,
        "bot_qq": "999",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FallbackAndChatRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        server.clear_chat_state()

    def tearDown(self) -> None:
        server.clear_chat_state()

    def test_mentioned_knowledge_miss_uses_fallback(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
        ):
            decision = server.should_process_message("这个载具的弱点在哪", True, group_id=1)

        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_mode, "fallback")
        self.assertEqual(decision.reason, "mentioned llm fallback")

    def test_unmentioned_factual_miss_does_not_use_general_fallback(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "should_reply_to_chat") as chat_router,
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message("这个新武器要怎么玩？", False, group_id=1)

        self.assertFalse(decision.should_reply)
        self.assertNotEqual(decision.reply_mode, "fallback")
        chat_router.assert_not_called()

    def test_unmentioned_casual_question_can_use_chat(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "chat_reply_quota_reason", return_value=""),
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message(
                "不知道他们看到是什么感想",
                False,
                group_id=1,
                chat_context=("群友A：我给学弟留了一个手办",),
            )

        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_mode, "chat")

    def test_chat_no_reply_token_is_detected(self) -> None:
        self.assertTrue(is_chat_no_reply("NO_REPLY"))
        self.assertTrue(is_chat_no_reply(" no_reply\n"))
        self.assertFalse(is_chat_no_reply("学弟可能当场就沉默了"))

    def test_bare_reaction_requires_substantive_context(self) -> None:
        configured = routing_settings()
        with (
            patch.object(server, "settings", configured),
            patch.object(server, "chat_reply_quota_reason", return_value=""),
            patch.object(server, "auto_reply_enabled", True),
        ):
            without_context = server.consider_chat_reply(
                "笑死",
                group_id=1,
                chat_context=("群友A：笑死",),
                mentions_other=False,
            )
            with_context = server.consider_chat_reply(
                "笑死",
                group_id=1,
                chat_context=("群友A：学弟轭头把手办丢了", "群友B：笑死"),
                mentions_other=False,
            )

        self.assertFalse(without_context.should_reply)
        self.assertEqual(without_context.reason, "bare reaction without context")
        self.assertTrue(with_context.should_reply)

    def test_unmentioned_casual_message_can_use_chat_router(self) -> None:
        context = ("群友1：今晚好热闹", "群友2：终于凑齐一车人了")
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "chat_reply_quota_reason", return_value=""),
            patch.object(server, "should_reply_to_chat") as chat_router,
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message(
                "今晚这车坐得够满",
                False,
                group_id=1,
                chat_context=context,
            )

        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_mode, "chat")
        self.assertEqual(decision.chat_context, context)
        self.assertEqual(decision.reason, "chat candidate queued")
        chat_router.assert_not_called()

    def test_casual_message_can_chat_even_if_search_matches_incidentally(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("某个载具资料", ["载具入门"], 0.3, 1.0),
            ),
            patch.object(server, "chat_reply_quota_reason", return_value=""),
            patch.object(server, "should_reply_to_chat") as chat_router,
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message(
                "今晚这车坐得够满",
                False,
                group_id=1,
                chat_context=("群友1：终于凑齐一车人了",),
            )

        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_mode, "chat")
        chat_router.assert_not_called()

    def test_chat_filters_links_and_messages_directed_at_others(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "should_reply_to_chat") as chat_router,
            patch.object(server, "auto_reply_enabled", True),
        ):
            link = server.should_process_message("这个链接 https://example.com", False, group_id=1)
            directed = server.should_process_message(
                "晚上一起打两把",
                False,
                group_id=1,
                mentions_other=True,
            )

        self.assertFalse(link.should_reply)
        self.assertFalse(directed.should_reply)
        chat_router.assert_not_called()

    def test_birthday_celebration_can_join_even_when_someone_is_mentioned(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "chat_reply_quota_reason", return_value=""),
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message(
                "祝小王生日快乐",
                False,
                group_id=1,
                chat_context=("群友A：今天小王过生日", "群友B：生日快乐"),
                mentions_other=True,
            )

        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_mode, "chat")
        self.assertEqual(decision.reason, "social celebration candidate queued")
        self.assertEqual(len(decision.chat_context), 2)

    def test_graduation_celebration_can_join_and_target_mentioned_user(self) -> None:
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "chat_reply_quota_reason", return_value=""),
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message(
                "恭喜毕业",
                False,
                group_id=1,
                chat_context=("群友A：今天毕业典礼",),
                mentions_other=True,
            )
            target = server.response_mention_user_id(
                mentioned=False,
                user_id="20001",
                reply_mode=decision.reply_mode,
                question="恭喜毕业",
                mentioned_user_ids=("10001",),
            )

        self.assertTrue(decision.should_reply)
        self.assertEqual(server.celebration_kind("恭喜毕业"), "graduation")
        self.assertEqual(target, "10001")

    def test_response_mention_targets_asker_or_self_celebration(self) -> None:
        with patch.object(server, "settings", routing_settings(bot_qq="999")):
            self.assertEqual(
                server.response_mention_user_id(
                    mentioned=True,
                    user_id="10001",
                    reply_mode="knowledge",
                    question="医疗兵怎么玩",
                ),
                "10001",
            )
            self.assertEqual(
                server.response_mention_user_id(
                    mentioned=False,
                    user_id="10002",
                    reply_mode="chat",
                    question="我今天生日",
                ),
                "10002",
            )
            self.assertEqual(
                server.response_mention_user_id(
                    mentioned=False,
                    user_id="10003",
                    reply_mode="chat",
                    question="小王生日快乐",
                ),
                "",
            )

    def test_birthday_discussion_does_not_force_a_celebration(self) -> None:
        self.assertFalse(server.looks_like_birthday_celebration("去年生日收了什么礼物？"))
        self.assertFalse(server.looks_like_birthday_celebration("生日礼物买什么好"))
        self.assertFalse(server.looks_like_birthday_celebration("他什么时候生日"))
        self.assertTrue(server.looks_like_birthday_celebration("今天是小王生日"))
        self.assertTrue(server.looks_like_birthday_celebration("祝小王生日快乐"))

    def test_prompts_are_separate_and_keep_plain_term_rule(self) -> None:
        self.assertNotEqual(FALLBACK_PROMPT, SYSTEM_PROMPT)
        self.assertNotEqual(CHAT_PROMPT, CHAT_ROUTER_PROMPT)
        self.assertIn("队包", FALLBACK_PROMPT)
        self.assertIn("通读最近所有群聊", CHAT_PROMPT)
        self.assertIn("不要", CHAT_PROMPT)
        self.assertIn("强行转到 Squad", CHAT_PROMPT)
        self.assertIn("生日或毕业", CHAT_PROMPT)
        self.assertNotIn("教官", CHAT_PROMPT)

    def test_chat_generation_uses_history_without_knowledge_context(self) -> None:
        with patch.object(llm, "_answer_or_error", return_value="确实，刷久了容易困") as call:
            answer = llm.answer_chat(
                base_url="https://example.invalid",
                api_key="test-key",
                model="test-model",
                message="玩战甲其实后边也犯困",
                context=("群友A：最近又开始刷战甲了", "群友B：后期基本都在刷材料"),
            )

        self.assertEqual(answer, "确实，刷久了容易困")
        messages = call.call_args.kwargs["messages"]
        user_prompt = messages[1]["content"]
        self.assertIn("群友A：最近又开始刷战甲了", user_prompt)
        self.assertIn("群友B：后期基本都在刷材料", user_prompt)
        self.assertIn("当前消息：玩战甲其实后边也犯困", user_prompt)
        self.assertNotIn("知识库资料", user_prompt)

    def test_fallback_generation_receives_group_context(self) -> None:
        with patch.object(llm, "_answer_or_error", return_value="这里说的枪男就是专心练枪的玩法") as call:
            llm.ask_fallback_llm(
                base_url="https://example.invalid",
                api_key="test-key",
                model="test-model",
                question="怎么成为枪男",
                context=("群友A：这把别当伏地魔了", "群友B：练练对枪"),
            )

        user_prompt = call.call_args.kwargs["messages"][1]["content"]
        self.assertIn("群友A：这把别当伏地魔了", user_prompt)
        self.assertIn("当前问题：怎么成为枪男", user_prompt)

    def test_mentioned_weak_match_uses_fallback(self) -> None:
        weak = ContextResult("弱相关资料", ["工事与武器"], 0.28, 0.11)
        with (
            patch.object(server, "settings", routing_settings()),
            patch.object(server, "retrieve_knowledge", return_value=weak),
        ):
            decision = server.should_process_message("这个新武器怎么玩", True, group_id=1)

        self.assertEqual(decision.reply_mode, "fallback")
        self.assertEqual(decision.reason, "mentioned weak-context llm fallback")
        self.assertAlmostEqual(decision.retrieval_coverage, 0.11)

    def test_combined_mentioned_question_uses_knowledge(self) -> None:
        decision = server.should_process_message(
            "医疗要咋玩，还有榴弹要咋玩",
            True,
            group_id=983063031,
        )

        self.assertEqual(decision.reply_mode, "knowledge")
        self.assertGreaterEqual(decision.retrieval_coverage, 0.6)

    def test_chat_group_allowlist_is_independent(self) -> None:
        configured = routing_settings(chat_allowed_group_ids=("2",))
        with (
            patch.object(server, "settings", configured),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("", [], 0.0, 0.0),
            ),
            patch.object(server, "auto_reply_enabled", True),
        ):
            decision = server.should_process_message("今晚人挺多", False, group_id=1)

        self.assertFalse(decision.should_reply)
        self.assertEqual(decision.reason, "chat group not allowed")

    def test_model_answer_is_normalized_before_sending(self) -> None:
        answer = "# 说明\n- **RallyPoint** 放好了\n- 去 rally点复活"
        normalized = normalize_model_answer(answer, max_chars=100)

        self.assertNotIn("#", normalized)
        self.assertNotIn("**", normalized)
        self.assertNotIn("Rally", normalized)
        self.assertEqual(normalized.count("队包"), 2)
        self.assertNotIn("队包点", normalized)
        self.assertLessEqual(len(normalize_model_answer("很长" * 100, 30)), 30)


class ChatStateTests(unittest.TestCase):
    def setUp(self) -> None:
        server.clear_chat_state()

    def tearDown(self) -> None:
        server.clear_chat_state()

    def test_context_is_trimmed_by_time_and_message_count(self) -> None:
        with patch.object(
            server,
            "settings",
            SimpleNamespace(
                bot_qq="999",
                chat_context_seconds=120,
                chat_context_messages=6,
            ),
        ):
            server.record_group_chat_message(1, "1", "过期消息", 90)
            server.record_group_chat_message(1, "2", "第一条", 150)
            server.record_group_chat_message(1, "3", "第二条", 180)
            server.record_group_chat_message(1, "4", "第三条", 210)
            server.record_group_chat_message(1, "999", "机器人自己的话", 215)
            context = server.recent_group_chat_context(
                1,
                now=220,
                context_seconds=120,
                max_messages=2,
            )

            self.assertEqual(context, ("群友A：第三条", "机器人自己：机器人自己的话"))

    def test_newer_group_message_supersedes_chat_candidate(self) -> None:
        with patch.object(
            server,
            "settings",
            SimpleNamespace(
                bot_qq="999",
                chat_context_seconds=300,
                chat_context_messages=12,
            ),
        ):
            candidate_sequence = server.record_group_chat_message(1, "1", "第一句", 100)
            self.assertFalse(server.group_chat_has_newer_user_message(1, candidate_sequence))
            server.record_group_chat_message(1, "999", "机器人回复", 101)
            self.assertFalse(server.group_chat_has_newer_user_message(1, candidate_sequence))
            server.record_group_chat_message(1, "2", "话题已经继续了", 102)
            self.assertTrue(server.group_chat_has_newer_user_message(1, candidate_sequence))

    def test_context_preserves_reply_relation_and_marks_current_message(self) -> None:
        with patch.object(
            server,
            "settings",
            SimpleNamespace(
                bot_qq="999",
                chat_context_seconds=300,
                chat_context_messages=12,
            ),
        ):
            server.record_group_chat_message(
                1,
                "100",
                "我电脑重装系统了",
                100,
                message_id="m1",
            )
            server.record_group_chat_message(1, "200", "今晚吃什么", 101, message_id="m2")
            current = server.record_group_chat_message(
                1,
                "300",
                "那语音得重新装了",
                102,
                message_id="m3",
                reply_message_id="m1",
                reply_target_user_id="100",
                reply_text="我电脑重装系统了",
            )

            context = server.recent_group_chat_context(
                1,
                now=102,
                max_messages=2,
                focus_sequence=current,
            )

            self.assertEqual(
                context,
                (
                    "群友A：我电脑重装系统了",
                    "【当前消息】群友B（回复群友A“我电脑重装系统了”）：那语音得重新装了",
                ),
            )
            self.assertEqual(server.find_group_chat_message(1, "m1").text, "我电脑重装系统了")

    def test_chat_cooldown_and_hourly_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            server.mark_chat_replied(1, now=1000, db_path=db_path)

            self.assertEqual(
                server.chat_reply_quota_reason(
                    1,
                    now=1100,
                    cooldown_seconds=180,
                    max_per_hour=8,
                    db_path=db_path,
                ),
                "chat cooldown",
            )
            self.assertEqual(
                server.chat_reply_quota_reason(
                    1,
                    now=1200,
                    cooldown_seconds=180,
                    max_per_hour=1,
                    db_path=db_path,
                ),
                "chat hourly limit",
            )
            self.assertEqual(
                server.chat_reply_quota_reason(
                    1,
                    now=4601,
                    cooldown_seconds=180,
                    max_per_hour=1,
                    db_path=db_path,
                ),
                "",
            )
            self.assertEqual(
                server.chat_reply_quota_reason(
                    2,
                    now=1100,
                    cooldown_seconds=180,
                    max_per_hour=1,
                    db_path=db_path,
                ),
                "",
            )

    def test_chat_audit_records_anonymized_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_path = Path(temp_dir) / "audit.jsonl"
            with patch.object(
                server,
                "settings",
                SimpleNamespace(message_audit_log=str(audit_path)),
            ):
                server.write_message_audit(
                    decision="answered",
                    reason="chat router accepted",
                    group_id=1,
                    user_id=123,
                    question="玩战甲后边也犯困",
                    reply_mode="chat",
                    chat_context=("群友A：最近在刷战甲", "群友B：材料刷累了"),
                    answer="刷到后面确实容易犯困，材料循环太重复了。",
                )

            record = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(
                record["chat_context"],
                ["群友A：最近在刷战甲", "群友B：材料刷累了"],
            )
            self.assertNotIn("123", "".join(record["chat_context"]))
            self.assertEqual(record["answer"], "刷到后面确实容易犯困，材料循环太重复了。")

    def test_celebration_dedup_is_per_target_and_event_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            server.mark_celebration_replied(
                1,
                "10001",
                "birthday",
                now=1000,
                db_path=db_path,
            )

            self.assertTrue(
                server.celebration_was_replied(
                    1,
                    "10001",
                    "birthday",
                    now=2000,
                    db_path=db_path,
                )
            )
            self.assertFalse(
                server.celebration_was_replied(
                    1,
                    "10001",
                    "graduation",
                    now=2000,
                    db_path=db_path,
                )
            )
            self.assertFalse(
                server.celebration_was_replied(
                    1,
                    "10002",
                    "birthday",
                    now=2000,
                    db_path=db_path,
                )
            )
            self.assertFalse(
                server.celebration_was_replied(
                    1,
                    "10001",
                    "birthday",
                    now=87401,
                    db_path=db_path,
                )
            )


if __name__ == "__main__":
    unittest.main()
