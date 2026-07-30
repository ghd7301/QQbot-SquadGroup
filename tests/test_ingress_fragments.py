import unittest
from types import SimpleNamespace

from squad_bot.ingress import fragments
from squad_bot.models import MessageFragmentBuffer


class FragmentIngressTests(unittest.TestCase):
    def test_audience_classification_receives_bot_id(self) -> None:
        calls = []
        deps = SimpleNamespace(
            settings=SimpleNamespace(bot_qq="999"),
            classify_audience=lambda item, **kwargs: calls.append((item, kwargs))
            or "bot",
        )
        item = {"question": "教官回答一下"}

        audience = fragments.classify_fragment_audience(deps, item)

        self.assertEqual(audience, "bot")
        self.assertEqual(calls, [(item, {"bot_user_id": "999"})])

    def test_dispatch_uses_mention_priority(self) -> None:
        queued = []
        buffer = MessageFragmentBuffer(
            group_id=100,
            user_id="200",
            audience="bot",
            item={},
            parts=["问题"],
            fragments=[{"question": "问题"}],
            started_at=0,
            deadline=1,
        )
        deps = SimpleNamespace(
            semantic_bot_fragment_count=lambda _buffer: 1,
            _fragment_prefix_item=lambda _buffer, _count: {
                "question": "问题",
                "mentioned": True,
            },
            enqueue_persistent_message=lambda priority, item: queued.append(
                (priority, item)
            )
            or 7,
        )

        pending_id = fragments._dispatch_fragment_buffer(deps, buffer)

        self.assertEqual(pending_id, 7)
        self.assertEqual(queued[0][0], 0)
        self.assertTrue(queued[0][1]["mentioned"])


if __name__ == "__main__":
    unittest.main()
