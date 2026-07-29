import unittest
from types import SimpleNamespace

from squad_bot.observability.health import build_health_payload


class SizedState:
    def __init__(self, size: int) -> None:
        self.size = size

    def qsize(self) -> int:
        return self.size

    def buffered_count(self) -> int:
        return self.size


class HealthPayloadTests(unittest.TestCase):
    def test_disabled_memory_uses_zero_defaults(self) -> None:
        deps = SimpleNamespace(
            chat_scene_state=SimpleNamespace(counts=lambda: (1, 0)),
            chat_memory_manager=None,
            pending_status_counts=lambda: {"queued": 2, "retry": 3},
            semantic_planner_health_snapshot=lambda: {"addressed": {}},
            kb=SimpleNamespace(chunks=[1, 2]),
            message_queue=SizedState(1),
            normal_message_queue=SizedState(2),
            chat_queue=SizedState(3),
            fragment_aggregator=SizedState(4),
        )

        payload = build_health_payload(deps)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["queued"], 6)
        self.assertEqual(payload["pending"], 5)
        self.assertEqual(payload["fragment_buffered"], 4)
        self.assertEqual(payload["memory_messages"], 0)
        self.assertFalse(payload["memory_paused"])


if __name__ == "__main__":
    unittest.main()
