import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from squad_bot.ingress.events import handle_onebot_event


class OneBotIngressTests(unittest.TestCase):
    def test_non_group_event_is_ignored(self) -> None:
        status, payload = handle_onebot_event(
            SimpleNamespace(),
            {"post_type": "meta_event"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["ignored"], "not group message")

    def test_allowed_group_recall_updates_history(self) -> None:
        recall = Mock()
        schedule_save = Mock()
        deps = SimpleNamespace(
            settings=SimpleNamespace(allowed_group_ids={"100"}),
            recall_group_chat_message=recall,
            schedule_chat_history_save=schedule_save,
        )

        status, payload = handle_onebot_event(
            deps,
            {
                "post_type": "notice",
                "notice_type": "group_recall",
                "group_id": 100,
                "message_id": "200",
            },
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["recalled"])
        recall.assert_called_once_with(100, "200")
        schedule_save.assert_called_once_with()

    def test_disallowed_group_only_audits_bot_mention(self) -> None:
        audit = Mock()
        deps = SimpleNamespace(
            settings=SimpleNamespace(allowed_group_ids={"100"}),
            extract_event_question=lambda _event: ("问题", True),
            write_message_audit=audit,
        )
        event = {
            "message_type": "group",
            "group_id": 200,
            "user_id": 300,
            "time": 400,
        }

        status, payload = handle_onebot_event(deps, event)

        self.assertEqual(status, 200)
        self.assertEqual(payload["ignored"], "group not allowed")
        audit.assert_called_once_with(
            decision="ignored",
            reason="group not allowed",
            group_id=200,
            user_id=300,
            question="问题",
            mentioned=True,
            event_time=400,
        )


if __name__ == "__main__":
    unittest.main()
