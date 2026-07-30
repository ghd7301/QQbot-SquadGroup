import threading
import unittest
from types import SimpleNamespace

from squad_bot.delivery import replies
from squad_bot.models import GroupChatMessage


class ReplyDeliveryTests(unittest.TestCase):
    def test_bot_turn_metadata_tracks_all_trigger_messages(self) -> None:
        deps = SimpleNamespace(_message_ids=lambda _item: ["m1", "m2"])

        trigger_ids, turn_id = replies.bot_turn_metadata(deps, {}, "bot-7")

        self.assertEqual(trigger_ids, ("m1", "m2"))
        self.assertEqual(turn_id, "bot:bot-7")

    def test_coverage_requires_active_message_from_current_bot(self) -> None:
        history = (
            GroupChatMessage(
                "有效回复",
                "999",
                1,
                generated_for_message_ids=("m1",),
            ),
            GroupChatMessage(
                "已撤回回复",
                "999",
                2,
                generated_for_message_ids=("m2",),
                message_status="recalled",
            ),
            GroupChatMessage(
                "其他成员",
                "200",
                3,
                generated_for_message_ids=("m3",),
            ),
        )
        deps = SimpleNamespace(
            settings=SimpleNamespace(bot_qq="999"),
            chat_history_lock=threading.Lock(),
            group_chat_history={100: history},
        )

        self.assertTrue(replies.message_already_covered_by_bot(deps, 100, "m1"))
        self.assertFalse(replies.message_already_covered_by_bot(deps, 100, "m2"))
        self.assertFalse(replies.message_already_covered_by_bot(deps, 100, "m3"))

    def test_review_ids_only_include_real_messages_from_original_sender(self) -> None:
        item = {"message_id": "m1"}
        context = (
            '{"message_id":"m1","speaker":{"id":"member_a"}}',
            '{"message_id":"m2","speaker":{"id":"member_a"}}',
            '{"message_id":"m3","speaker":{"id":"member_b"}}',
        )
        deps = SimpleNamespace(
            _message_ids=lambda _item: ["m1"],
            _context_message_speakers=replies.context_message_speakers,
        )
        review = SimpleNamespace(related_message_ids=("m2", "m3", "invented"))

        replies.merge_review_message_ids(deps, item, review, context)

        self.assertEqual(item["message_ids"], ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()

