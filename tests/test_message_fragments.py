import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from squad_bot import llm, server
from squad_bot.message_fragments import FragmentAggregator


def fragment_settings(**overrides):
    values = {
        "bot_qq": "999",
        "message_fragment_debounce_seconds": 3.0,
        "message_fragment_max_wait_seconds": 8.0,
        "message_fragment_max_parts": 6,
        "message_fragment_max_chars": 800,
        "message_fragment_semantic_enabled": True,
        "message_fragment_semantic_model": "classifier-model",
        "message_fragment_semantic_timeout_seconds": 8,
        "message_fragment_semantic_min_confidence": 0.75,
        "llm_base_url": "https://example.invalid",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "onebot_api_url": "http://127.0.0.1:3000",
        "onebot_access_token": "",
        "chat_context_seconds": 300,
        "chat_context_messages": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def item(
    question,
    *,
    user_id="200",
    mentioned=False,
    message_id="1",
    reply_message_id="",
    reply_target_user_id="",
    mentioned_user_ids=(),
    chat_sequence=1,
):
    return {
        "group_id": 100,
        "question": question,
        "mentioned": mentioned,
        "time": 1000,
        "user_id": user_id,
        "sender_role": "member",
        "chat_context": [],
        "mentions_other": bool(mentioned_user_ids),
        "mentioned_user_ids": list(mentioned_user_ids),
        "reply_message_id": reply_message_id,
        "reply_target_user_id": reply_target_user_id,
        "reply_text": "",
        "message_id": message_id,
        "chat_sequence": chat_sequence,
    }


class MessageFragmentTests(unittest.TestCase):
    def setUp(self):
        server.clear_fragment_state()

    def tearDown(self):
        server.clear_fragment_state()

    def test_aggregator_returns_due_buffer_to_worker(self):
        aggregator = FragmentAggregator()
        aggregator.submit(
            item("等到截止时间", message_id="due-1"),
            "unknown",
            now=0,
            is_immediate=False,
            max_parts=6,
            max_chars=800,
            debounce_seconds=3,
            max_wait_seconds=8,
        )

        due = aggregator.wait_for_due(monotonic=lambda: 3)

        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].item["message_ids"], ["due-1"])
        self.assertEqual(aggregator.buffered_count(), 0)

    def test_bot_mention_and_following_unknown_fragment_merge(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append((priority, payload)) or 1,
            ),
            patch.object(server, "classify_bot_fragment_prefix", return_value=(2, 0.95)),
        ):
            server.submit_message_fragment(
                item("我刚来不太会", mentioned=True, message_id="11"), now=0
            )
            server.submit_message_fragment(
                item("队包应该怎么放", message_id="12", chat_sequence=2), now=1
            )
            server.flush_group_fragment_buffer(100)

        self.assertEqual(len(dispatched), 1)
        priority, payload = dispatched[0]
        self.assertEqual(priority, 0)
        self.assertTrue(payload["mentioned"])
        self.assertEqual(payload["question"], "我刚来不太会\n队包应该怎么放")
        self.assertEqual(payload["message_ids"], ["11", "12"])
        self.assertEqual(payload["chat_sequence"], 2)

    def test_human_directed_second_fragment_does_not_inherit_bot_mention(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append((priority, payload)) or len(dispatched),
            ),
            patch.object(server, "classify_bot_fragment_prefix", return_value=(1, 0.95)),
        ):
            server.submit_message_fragment(item("我刚来不太会", user_id="A"), now=0)
            server.submit_message_fragment(
                item("来新人了，教一教", user_id="B", mentioned=True, message_id="2"),
                now=1,
            )
            result = server.submit_message_fragment(
                item("有啥不会的都可以问他", user_id="B", message_id="3"),
                now=2,
            )
            flushed_id = server.flush_group_fragment_buffer(100)

        self.assertEqual(result, [])
        self.assertEqual(flushed_id, 2)
        self.assertEqual(len(dispatched), 2)
        bot_items = [payload for priority, payload in dispatched if priority == 0]
        self.assertEqual(len(bot_items), 1)
        self.assertEqual(bot_items[0]["question"], "来新人了，教一教")
        self.assertTrue(bot_items[0]["mentioned"])
        self.assertNotIn("有啥不会的都可以问他", bot_items[0]["question"])
        self.assertNotIn(100, server.group_fragment_buffers)

    def test_wording_does_not_directly_determine_audience(self):
        with patch.object(server, "settings", fragment_settings()):
            self.assertEqual(
                server.classify_fragment_audience(item("有啥不会的都可以问他")),
                "unknown",
            )
            self.assertEqual(
                server.classify_fragment_audience(item("让教官跟你说明白")),
                "unknown",
            )

    def test_explicit_knowledge_command_uses_priority_audience(self):
        command_item = item("队包多久一轮")
        command_item["explicit_knowledge_command"] = True
        with patch.object(server, "settings", fragment_settings()):
            self.assertEqual(server.classify_fragment_audience(command_item), "bot")

    def test_low_confidence_semantic_result_does_not_inherit_bot_target(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(server, "classify_bot_fragment_prefix", return_value=(2, 0.4)),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append(payload) or 1,
            ),
        ):
            server.submit_message_fragment(item("教一下新人", mentioned=True), now=0)
            server.submit_message_fragment(item("我再补充一句", message_id="2"), now=1)
            server.flush_group_fragment_buffer(100)

        self.assertEqual(dispatched[0]["question"], "教一下新人")

    def test_semantic_failure_does_not_inherit_bot_target(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(server, "classify_bot_fragment_prefix", return_value=None),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append(payload) or 1,
            ),
        ):
            server.submit_message_fragment(item("教一下新人", mentioned=True), now=0)
            server.submit_message_fragment(item("目标不明确", message_id="2"), now=1)
            server.flush_group_fragment_buffer(100)

        self.assertEqual(dispatched[0]["message_ids"], ["1"])

    def test_deferred_dispatch_does_not_call_classifier_in_request_path(self):
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(server, "classify_bot_fragment_prefix") as classifier,
            patch.object(server, "enqueue_persistent_message") as enqueue,
        ):
            server.submit_message_fragment(item("第一人", user_id="A"), now=0)
            server.submit_message_fragment(
                item("第二人", user_id="B"),
                now=1,
                defer_dispatch=True,
            )

        classifier.assert_not_called()
        enqueue.assert_not_called()
        self.assertEqual(len(server.ready_fragment_buffers), 1)

    def test_different_user_flushes_current_buffer(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append(payload) or 1,
            ),
        ):
            server.submit_message_fragment(item("第一段", user_id="A"), now=0)
            server.submit_message_fragment(item("另一人的话", user_id="B"), now=1)

        self.assertEqual([payload["question"] for payload in dispatched], ["第一段"])
        self.assertEqual(server.group_fragment_buffers[100].parts, ["另一人的话"])

    def test_different_explicit_reply_targets_split(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append(payload) or 1,
            ),
        ):
            server.submit_message_fragment(
                item(
                    "回复第一条",
                    mentioned=True,
                    reply_message_id="bot-msg-1",
                    reply_target_user_id="999",
                ),
                now=0,
            )
            server.submit_message_fragment(
                item(
                    "回复第二条",
                    mentioned=True,
                    reply_message_id="bot-msg-2",
                    reply_target_user_id="999",
                ),
                now=1,
            )

        self.assertEqual([payload["question"] for payload in dispatched], ["回复第一条"])
        self.assertEqual(
            server.group_fragment_buffers[100].item["reply_message_id"], "bot-msg-2"
        )

    def test_later_bot_mention_does_not_retroactively_target_previous_text(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append((priority, payload)) or 1,
            ),
        ):
            server.submit_message_fragment(item("这是跟群里说的", message_id="1"), now=0)
            server.submit_message_fragment(
                item("教官回答这个", mentioned=True, message_id="2"), now=1
            )

        self.assertEqual(dispatched[0][0], 1)
        self.assertEqual(dispatched[0][1]["question"], "这是跟群里说的")
        self.assertEqual(server.group_fragment_buffers[100].parts, ["教官回答这个"])

    def test_explicit_bot_mention_overrides_reply_to_human(self):
        configured = fragment_settings()
        payload = item(
            "教官你看这个",
            mentioned=True,
            reply_message_id="human-msg",
            reply_target_user_id="123",
        )
        with patch.object(server, "settings", configured):
            self.assertEqual(server.classify_fragment_audience(payload), "bot")

    def test_non_text_message_from_new_speaker_flushes_buffer(self):
        dispatched = []
        with (
            patch.object(server, "settings", fragment_settings()),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append(payload) or 1,
            ),
        ):
            server.submit_message_fragment(item("还没说完", user_id="A"), now=0)
            server.flush_fragment_buffer_for_new_speaker(100, "B")

        self.assertEqual([payload["question"] for payload in dispatched], ["还没说完"])
        self.assertNotIn(100, server.group_fragment_buffers)

    def test_deadline_uses_debounce_and_hard_max_wait(self):
        with patch.object(server, "settings", fragment_settings()):
            server.submit_message_fragment(item("一", message_id="1"), now=0)
            self.assertEqual(server.group_fragment_buffers[100].deadline, 3)
            server.submit_message_fragment(item("二", message_id="2"), now=2)
            self.assertEqual(server.group_fragment_buffers[100].deadline, 5)
            server.submit_message_fragment(item("三", message_id="3"), now=7)
            self.assertEqual(server.group_fragment_buffers[100].deadline, 8)

    def test_part_and_character_limits_force_dispatch(self):
        dispatched = []
        configured = fragment_settings(message_fragment_max_parts=2, message_fragment_max_chars=5)
        with (
            patch.object(server, "settings", configured),
            patch.object(
                server,
                "enqueue_persistent_message",
                side_effect=lambda priority, payload: dispatched.append(payload) or 1,
            ),
        ):
            server.submit_message_fragment(item("ab", message_id="1"), now=0)
            server.submit_message_fragment(item("cd", message_id="2"), now=1)

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["question"], "ab\ncd")

    def test_merged_utterance_creates_one_persistent_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "pending.sqlite3"
            configured = fragment_settings(pending_queue_db=str(db_path))
            with patch.object(server, "settings", configured):
                server.submit_message_fragment(item("第一段", message_id="1"), now=0)
                server.submit_message_fragment(item("第二段", message_id="2"), now=1)
                server.flush_group_fragment_buffer(100)
                pending = server.load_pending_messages(db_path)
                queued = server.normal_message_queue.get_nowait()
                server.normal_message_queue.task_done()

            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0][2]["question"], "第一段\n第二段")
            self.assertEqual(queued[2]["question"], "第一段\n第二段")

    def test_raw_fragments_stay_separate_in_chat_history(self):
        with patch.object(server, "settings", fragment_settings()):
            server.clear_chat_state()
            server.record_group_chat_message(100, "A", "第一段", 1000, message_id="1")
            server.record_group_chat_message(100, "A", "第二段", 1001, message_id="2")

        history = server.group_chat_history[100]
        self.assertEqual([entry.text for entry in history], ["第一段", "第二段"])

    def test_bot_turn_fields_survive_chat_history_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "chat_history.json"
            with patch.object(server, "settings", fragment_settings(bot_qq="999")):
                server.clear_chat_state()
                server.record_group_chat_message(
                    100,
                    "999",
                    "我刚才回的是第一段",
                    1002,
                    message_id="bot-1",
                    generated_for_message_ids=("1", "2"),
                    turn_id="bot:bot-1",
                    reply_mode="chat",
                    semantic_topic="分段消息",
                )
                server.save_chat_history(history_path)
                server.clear_chat_state()
                self.assertEqual(server.load_chat_history(history_path), 1)

        restored = server.group_chat_history[100][0]
        self.assertEqual(restored.generated_for_message_ids, ("1", "2"))
        self.assertEqual(restored.turn_id, "bot:bot-1")
        self.assertEqual(restored.reply_mode, "chat")
        self.assertEqual(restored.semantic_topic, "分段消息")

    def test_sending_bot_turn_immediately_persists_short_history(self):
        with (
            patch.object(server, "settings", fragment_settings(bot_qq="999")),
            patch.object(server, "send_group_msg", return_value="bot-1"),
            patch.object(server, "save_chat_history") as save_history,
        ):
            server.clear_chat_state()
            server.send_and_record_bot_turn(
                group_id=100,
                item=item("原问题", mentioned=True, message_id="user-1"),
                answer="我的回答",
                reply_mode="fallback",
                reply_to_trigger=True,
            )

        save_history.assert_called_once()

    def test_review_message_ids_accept_only_real_context_messages(self):
        payload = item("队标怎么考", mentioned=True, message_id="m1")
        review = llm.FinalReplyReview(
            "regenerate",
            "同一发送者补充了问题",
            0.95,
            updated_question="队标怎么考，晋升路线是什么？",
            context_relation="same_topic_update",
            related_message_ids=("m1", "m2", "m3", "invented"),
        )
        context = (
            '{"message_id":"m1","speaker":{"id":"member_a","role":"member"},"text":"队标怎么考"}',
            '{"message_id":"m2","speaker":{"id":"member_a","role":"member"},"text":"晋升路线是什么"}',
            '{"message_id":"m3","speaker":{"id":"member_b","role":"member"},"text":"另一个人的并行话题"}',
        )

        server.merge_review_message_ids(payload, review, context)

        self.assertEqual(payload["message_ids"], ["m1", "m2"])

    def test_completed_bot_turn_covers_following_fragment(self):
        configured = fragment_settings(bot_qq="999")
        with patch.object(server, "settings", configured):
            server.clear_chat_state()
            server.record_group_chat_message(
                100,
                "999",
                "合并回答",
                1002,
                message_id="bot-1",
                generated_for_message_ids=("m1", "m2"),
            )

            self.assertTrue(server.message_already_covered_by_bot(100, "m2"))
            self.assertFalse(server.message_already_covered_by_bot(100, "m3"))

    def test_semantic_classifier_parses_structured_result(self):
        with patch.object(
            llm,
            "_chat_completion",
            return_value='```json\n{"bot_part_count": 1, "confidence": 0.92}\n```',
        ):
            decision = llm.classify_bot_fragment_prefix(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                fragments=("教一下新人", "有问题可以问他"),
                context=("A：我刚来不太会",),
            )

        self.assertEqual(decision, (1, 0.92))

    def test_semantic_classifier_rejects_unstructured_result(self):
        with patch.object(llm, "_chat_completion", return_value="我觉得是第一段"):
            decision = llm.classify_bot_fragment_prefix(
                base_url="https://example.invalid",
                api_key="key",
                model="model",
                fragments=("第一段", "第二段"),
            )

        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
