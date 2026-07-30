import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from squad_bot.delivery import policies


class DeliveryPolicyTests(unittest.TestCase):
    def test_message_age_uses_mention_specific_limit(self) -> None:
        deps = SimpleNamespace(
            message_max_age_seconds=lambda mentioned: 300 if mentioned else 60
        )

        self.assertFalse(
            policies.is_message_too_old(deps, 940, False, now=1_000)
        )
        self.assertTrue(
            policies.is_message_too_old(deps, 939, False, now=1_000)
        )
        self.assertFalse(
            policies.is_message_too_old(deps, 701, True, now=1_000)
        )

    def test_event_age_uses_onebot_mention_detection(self) -> None:
        calls = []
        deps = SimpleNamespace(
            settings=SimpleNamespace(bot_qq="999"),
            is_mentioned=lambda bot_id, message: calls.append((bot_id, message))
            or True,
            is_message_too_old=lambda event_time, mentioned: (
                event_time == 10 and mentioned
            ),
        )

        self.assertTrue(
            policies.is_event_too_old(deps, {"time": 10, "message": "[CQ:at]"})
        )
        self.assertEqual(calls, [("999", "[CQ:at]")])

    def test_nonblocking_rate_limit_respects_reserved_slots(self) -> None:
        deps = SimpleNamespace(
            settings=SimpleNamespace(max_replies_per_minute=2),
            rate_limit_lock=threading.Lock(),
            reply_timestamps=[99.0],
        )

        with patch.object(policies.time, "time", return_value=100.0):
            self.assertFalse(
                policies.acquire_reply_slot(
                    deps,
                    block=False,
                    reserve_slots=1,
                )
            )
            self.assertTrue(policies.acquire_reply_slot(deps, block=False))


if __name__ == "__main__":
    unittest.main()

