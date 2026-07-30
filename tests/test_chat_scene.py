import unittest
from types import SimpleNamespace

from squad_bot import chat_scene
from squad_bot.chat_scene import ChatSceneState


class ChatSceneStateTests(unittest.TestCase):
    def test_request_threshold_and_followup_iteration(self) -> None:
        state = ChatSceneState()

        self.assertFalse(state.request_update(1, 1, min_messages=3))
        self.assertFalse(state.request_update(1, 2, min_messages=3))
        self.assertTrue(state.request_update(1, 3, min_messages=3))

        sequence, previous = state.begin_update(1)
        self.assertEqual(sequence, 3)
        self.assertIsNone(previous)

        self.assertFalse(state.request_update(1, 4, min_messages=3))
        self.assertFalse(state.request_update(1, 5, min_messages=3))
        self.assertFalse(state.request_update(1, 6, min_messages=3))
        self.assertTrue(state.should_continue(1, min_messages=3))

        sequence, previous = state.begin_update(1)
        self.assertEqual(sequence, 6)
        self.assertIsNone(previous)
        self.assertFalse(state.should_continue(1, min_messages=3))
        self.assertNotIn(1, state.running)
        self.assertNotIn(1, state.requested_sequences)
        self.assertNotIn(1, state.pending_messages)

    def test_scene_freshness_sequence_and_counts(self) -> None:
        state = ChatSceneState()
        state.set_scene(1, summary="当前话题", updated_at=100, sequence=5)

        self.assertEqual(
            state.current_summary(
                1,
                focus_sequence=5,
                now=150,
                stale_seconds=120,
            ),
            "当前话题",
        )
        self.assertEqual(
            state.current_summary(
                1,
                focus_sequence=4,
                now=150,
                stale_seconds=120,
            ),
            "",
        )
        self.assertEqual(
            state.current_summary(
                1,
                focus_sequence=5,
                now=221,
                stale_seconds=120,
            ),
            "",
        )
        self.assertEqual(state.counts(), (1, 0))

    def test_scene_service_respects_group_policy(self) -> None:
        deps = SimpleNamespace(
            auto_reply_enabled=True,
            settings=SimpleNamespace(
                chat_reply_enabled=True,
                chat_scene_enabled=True,
                chat_allowed_group_ids=("100",),
            ),
        )

        self.assertTrue(chat_scene.chat_scene_enabled_for_group(deps, 100))
        self.assertFalse(chat_scene.chat_scene_enabled_for_group(deps, 200))
        deps.auto_reply_enabled = False
        self.assertFalse(chat_scene.chat_scene_enabled_for_group(deps, 100))


if __name__ == "__main__":
    unittest.main()
