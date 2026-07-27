import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from squad_bot import server
from squad_bot.knowledge import ContextResult


def followup_settings() -> SimpleNamespace:
    return SimpleNamespace()


class FollowupContextTests(unittest.TestCase):
    def test_explicit_reply_uses_exact_persisted_bot_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            medical = server.ConversationState(
                last_question="医疗兵怎么玩",
                sources=("医疗兵",),
                timestamp=time.time(),
                user_id="100",
                last_answer="先保证自己安全，再救人。",
                reply_mode="knowledge",
                bot_message_id="bot-medical",
            )
            fob = server.ConversationState(
                last_question="FOB怎么放",
                sources=("FOB",),
                timestamp=time.time() + 1,
                user_id="200",
                last_answer="先找隐蔽位置。",
                reply_mode="knowledge",
                bot_message_id="bot-fob",
            )
            server.persist_conversation_turn(1, medical, db_path=db_path)
            server.persist_conversation_turn(1, fob, db_path=db_path)

            with patch.object(server, "settings", followup_settings()):
                match = server.followup_context_for(
                    1,
                    "100",
                    "那倒一片先救谁？",
                    True,
                    reply_message_id="bot-medical",
                    reply_target_user_id="999",
                    reply_text="先保证自己安全，再救人。",
                    bot_qq="999",
                    db_path=db_path,
                )

        self.assertIsNotNone(match)
        self.assertEqual(match.scope, "reply")
        self.assertEqual(match.state.last_question, "医疗兵怎么玩")
        self.assertEqual(match.state.last_answer, "先保证自己安全，再救人。")

    def test_reply_target_recovers_from_sqlite_when_get_msg_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            state = server.ConversationState(
                last_question="队包怎么用",
                sources=("队包",),
                timestamp=time.time(),
                user_id="100",
                last_answer="队包按波次提供复活。",
                reply_mode="knowledge",
                bot_message_id="bot-rally",
            )
            server.persist_conversation_turn(1, state, db_path=db_path)
            configured = SimpleNamespace(
                bot_qq="999",
                onebot_api_url="http://127.0.0.1:3000",
                onebot_access_token="",
                onebot_message_lookup_timeout_seconds=3,
            )
            with (
                patch.object(server, "settings", configured),
                patch.object(server, "find_group_chat_message", return_value=None),
                patch.object(server, "get_message_info", return_value=("", "")),
            ):
                sender_id, quoted = server.resolve_reply_message_context(
                    1,
                    "bot-rally",
                    db_path=db_path,
                )

        self.assertEqual(sender_id, "999")
        self.assertEqual(quoted, "队包按波次提供复活。")

    def test_explicit_reply_falls_back_to_quote_not_recent_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            with patch.object(server, "settings", followup_settings()):
                match = server.followup_context_for(
                    1,
                    "100",
                    "那多久？",
                    True,
                    reply_message_id="missing",
                    reply_target_user_id="999",
                    reply_text="队包每个人死亡后有强制复活冷却。",
                    bot_qq="999",
                    db_path=db_path,
                )
                missing = server.followup_context_for(
                    1,
                    "100",
                    "那多久？",
                    True,
                    reply_message_id="missing-again",
                    reply_target_user_id="999",
                    reply_text="",
                    bot_qq="999",
                    db_path=db_path,
                )

        self.assertEqual(match.scope, "reply_text")
        self.assertEqual(match.state.last_question, "")
        self.assertIn("强制复活冷却", match.state.last_answer)
        self.assertIsNone(missing)

    def test_implicit_followup_is_left_to_semantic_planner(self) -> None:
        with patch.object(server, "settings", followup_settings()):
            continued = server.followup_context_for(1, "100", "那倒一片先救谁", False)
            complete_new_question = server.followup_context_for(
                1, "100", "TS地址是什么", True
            )
            different_topic = server.followup_context_for(
                1, "100", "这个TS地址是什么", True
            )

        self.assertIsNone(continued)
        self.assertIsNone(complete_new_question)
        self.assertIsNone(different_topic)

    def test_generation_context_contains_previous_answer(self) -> None:
        match = server.FollowupMatch(
            state=server.ConversationState(
                last_question="队包怎么用",
                sources=("队包",),
                timestamp=time.time(),
                last_answer="每个人死亡后有18秒强制复活冷却。",
            ),
            scope="user",
        )

        retrieval = server.build_effective_question("那多久一轮", match)
        generation = server.build_generation_question("那多久一轮", match)

        self.assertIn("上一轮问题：队包怎么用", retrieval)
        self.assertNotIn("18秒", retrieval)
        self.assertIn("上一轮回答：每个人死亡后有18秒强制复活冷却。", generation)
        self.assertIn("当前追问：那多久一轮", generation)

    def test_retrieval_uses_questions_while_model_also_sees_previous_answer(self) -> None:
        configured = SimpleNamespace(
            max_context_chars=4500,
            knowledge_strong_min_score=0.18,
            knowledge_strong_min_coverage=0.6,
            llm_fallback_enabled=True,
            llm_base_url="https://example.invalid",
            llm_api_key="test-key",
            llm_model="test-model",
            max_answer_chars=500,
        )
        retrieval_question = "上一轮问题：队包怎么用\n当前追问：那多久一轮"
        generation_question = (
            "上一轮问题：队包怎么用\n"
            "上一轮回答：每个人死亡后有18秒强制复活冷却。\n"
            "当前追问：那多久一轮"
        )
        with (
            patch.object(server, "settings", configured),
            patch.object(
                server,
                "retrieve_knowledge",
                return_value=ContextResult("队包资料", ("队包",), 1.0, 1.0),
            ) as retrieve,
            patch.object(server, "ask_llm", return_value="60秒一轮。") as ask,
        ):
            answer = server.answer_question(
                "那多久一轮",
                generation_question,
                retrieval_question=retrieval_question,
                allow_fallback=False,
            )

        self.assertEqual(answer, "60秒一轮。")
        retrieve.assert_called_once_with(retrieval_question, 4500)
        self.assertEqual(ask.call_args.kwargs["question"], generation_question)

    def test_persisted_turn_is_available_for_explicit_reply(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            state = server.ConversationState(
                last_question="FOB和HAB有什么区别",
                sources=("FOB/HAB",),
                timestamp=time.time(),
                user_id="100",
                last_answer="FOB是范围，HAB才是出生建筑。",
                reply_mode="knowledge",
                bot_message_id="bot-1",
                user_message_id="user-1",
                trigger_message_ids=("user-1", "user-2"),
                turn_id="bot:bot-1",
                semantic_intent="knowledge",
                semantic_topic="FOB 与 HAB",
            )
            server.persist_conversation_turn(1, state, db_path=db_path)

            match = server.followup_context_for(
                1,
                "100",
                "那兵站呢",
                True,
                reply_message_id="bot-1",
                reply_target_user_id="999",
                bot_qq="999",
                db_path=db_path,
            )

        self.assertIsNotNone(match)
        self.assertEqual(match.state.last_answer, "FOB是范围，HAB才是出生建筑。")
        self.assertEqual(match.state.trigger_message_ids, ("user-1", "user-2"))
        self.assertEqual(match.state.turn_id, "bot:bot-1")
        self.assertEqual(match.state.semantic_topic, "FOB 与 HAB")


if __name__ == "__main__":
    unittest.main()
