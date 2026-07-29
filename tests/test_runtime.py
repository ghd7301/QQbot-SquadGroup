import tempfile
import unittest
from pathlib import Path

from squad_bot import server
from squad_bot.runtime import BotRuntime
from squad_bot.runtime_dependencies import RuntimeDependencies


class BotRuntimeTests(unittest.TestCase):
    def test_runtime_dependencies_read_and_write_live_namespace(self) -> None:
        namespace = {"enabled": False}
        dependencies = RuntimeDependencies(namespace)

        namespace["enabled"] = True
        self.assertTrue(dependencies.enabled)

        dependencies.enabled = False
        self.assertFalse(namespace["enabled"])

    def test_runtime_instances_do_not_share_mutable_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            knowledge_dir = Path(temp_dir) / "knowledge"
            first = BotRuntime(knowledge_dir)
            second = BotRuntime(knowledge_dir)

        first.queues.reply_timestamps.append(1.0)
        first.conversation.recent_reply_topics[(1, "topic")] = 2.0
        first.knowledge.recent_gap_queries["query"] = 3.0

        self.assertEqual(second.queues.reply_timestamps, [])
        self.assertEqual(second.conversation.recent_reply_topics, {})
        self.assertEqual(second.knowledge.recent_gap_queries, {})
        self.assertIsNot(first.conversation.history, second.conversation.history)
        self.assertIsNot(first.conversation.fragments, second.conversation.fragments)

    def test_server_compatibility_aliases_reference_bot_runtime(self) -> None:
        runtime = server.bot_runtime

        self.assertIs(server.kb, runtime.knowledge.base)
        self.assertIs(server.message_queue, runtime.queues.priority)
        self.assertIs(server.normal_message_queue, runtime.queues.normal)
        self.assertIs(server.chat_queue, runtime.queues.chat)
        self.assertIs(server.chat_history_state, runtime.conversation.history)
        self.assertIs(server.chat_scene_state, runtime.conversation.scene)
        self.assertIs(server.fragment_aggregator, runtime.conversation.fragments)
        self.assertIs(server.semantic_planner_health, runtime.planner_health)


if __name__ == "__main__":
    unittest.main()
