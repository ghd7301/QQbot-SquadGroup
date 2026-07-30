import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from squad_bot import chat_history


class ChatHistoryServiceTests(unittest.TestCase):
    def test_record_updates_runtime_sequence(self) -> None:
        state = chat_history.ChatHistoryState()
        deps = SimpleNamespace(
            chat_history_lock=threading.RLock(),
            chat_history_state=state,
            chat_message_sequence=0,
            chat_memory_manager=None,
            settings=SimpleNamespace(
                chat_context_seconds=300,
                chat_context_messages=20,
            ),
        )

        sequence = chat_history.record_group_chat_message(
            deps,
            100,
            "200",
            "测试消息",
            10,
            message_id="m1",
        )

        self.assertEqual(sequence, 1)
        self.assertEqual(deps.chat_message_sequence, 1)
        self.assertEqual(state.find(100, "m1").text, "测试消息")

    def test_reply_resolution_prefers_local_history(self) -> None:
        state = chat_history.ChatHistoryState()
        state.record(
            100,
            "200",
            "本地消息",
            10,
            context_seconds=300,
            context_messages=20,
            message_id="m1",
        )
        remote_lookup = Mock()
        deps = SimpleNamespace(
            find_group_chat_message=state.find,
            get_message_info=remote_lookup,
            settings=SimpleNamespace(
                onebot_api_url="",
                onebot_access_token="",
                onebot_message_lookup_timeout_seconds=1,
                bot_qq="999",
            ),
        )

        sender, text = chat_history.resolve_reply_message_context(
            deps, 100, "m1"
        )

        self.assertEqual((sender, text), ("200", "本地消息"))
        remote_lookup.assert_not_called()

    def test_recent_context_marks_focus_and_anonymizes_members(self) -> None:
        state = chat_history.ChatHistoryState()
        state.record(
            100,
            "200",
            "上下文消息",
            10,
            context_seconds=300,
            context_messages=20,
            message_id="m1",
        )
        settings = SimpleNamespace(
            bot_qq="999",
            member_id_secret="secret",
            onebot_access_token="",
            chat_context_seconds=300,
            chat_context_messages=20,
        )
        deps = SimpleNamespace(settings=settings, chat_history_state=state)
        deps.stable_member_id = lambda group_id, user_id: (
            chat_history.stable_member_id(deps, group_id, user_id)
        )
        deps._context_message_payload = lambda group_id, item, current: (
            chat_history.context_message_payload(
                deps, group_id, item, current=current
            )
        )

        context = chat_history.recent_group_chat_context(
            deps,
            100,
            now=10,
            focus_sequence=1,
        )
        payload = json.loads(context[0])

        self.assertTrue(payload["current"])
        self.assertEqual(payload["message_id"], "m1")
        self.assertTrue(payload["speaker"]["id"].startswith("member_"))
        self.assertNotEqual(payload["speaker"]["id"], "200")


if __name__ == "__main__":
    unittest.main()
