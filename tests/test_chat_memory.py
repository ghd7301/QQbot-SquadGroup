import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from squad_bot.chat_memory import ChatMemoryStore, MemoryMessage, redact_for_model
from squad_bot.embedding import HashedNgramEmbedding


class ChatMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ChatMemoryStore(
            Path(self.temp_dir.name) / "chat.sqlite3",
            HashedNgramEmbedding(dimensions=128),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def message(self, group, message_id, speaker, text, timestamp, **kwargs):
        return MemoryMessage(
            group_id=group,
            message_id=message_id,
            speaker_id=speaker,
            display_name="群友",
            speaker_role="member",
            text=text,
            event_time=timestamp,
            **kwargs,
        )

    def test_same_speaker_fragments_form_one_utterance_chunk(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "我刚来", now))
        self.store.add_message(self.message(1, "m2", "member_a", "不太会玩", now + 2))

        status = self.store.status()
        self.assertEqual(status["messages"], 2)
        self.assertEqual(status["chunks"], 1)
        hits = self.store.retrieve(group_id=1, query="刚来不会玩")
        self.assertIn("我刚来\n不太会玩", hits[0].text)

    def test_explicit_reply_chain_is_hard_retrieval(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "今晚八点集合", now))
        self.store.add_message(self.message(
            1, "m2", "member_b", "收到", now + 20,
            reply_message_id="m1", reply_speaker_id="member_a", quoted_text="今晚八点集合",
        ))
        hits = self.store.retrieve(
            group_id=1,
            query="完全不相干的检索文字",
            reply_message_id="m2",
            participant_scope="reply_chain",
        )
        self.assertTrue(hits)
        self.assertTrue(any("今晚八点集合" in hit.text for hit in hits))
        self.assertTrue(all("reply_chain" in hit.reasons for hit in hits))

    def test_lexical_probe_recalls_related_history_without_embedding_query(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "今晚八点在北桥集合", now))

        class FailingEmbedding:
            name = "must-not-run"
            dimensions = 128

            def embed(self, texts):
                raise AssertionError("lexical probe must stay local")

        self.store.embedding = FailingEmbedding()
        hits = self.store.lexical_probe(group_id=1, query="北桥集合时间")
        self.assertTrue(hits)
        self.assertIn("北桥", hits[0].text)

    def test_lexical_probe_rejects_single_weak_overlap(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "今天晚上一起吃饭", now))
        hits = self.store.lexical_probe(group_id=1, query="晚上开黑地图")
        self.assertFalse(hits)

    def test_retrieval_never_crosses_group_boundary(self):
        now = time.time()
        self.store.add_message(self.message(1, "g1", "member_a", "秘密集合点在北桥", now))
        self.store.add_message(self.message(2, "g2", "member_b", "另一个群在聊晚饭", now))
        hits = self.store.retrieve(group_id=2, query="秘密集合点 北桥")
        self.assertFalse(any("北桥" in hit.text for hit in hits))

    def test_recall_removes_message_but_preserves_other_fragment(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "第一段还要保留", now))
        self.store.add_message(self.message(1, "m2", "member_a", "第二段要撤回", now + 1))
        self.assertTrue(self.store.recall(1, "m2"))
        hits = self.store.retrieve(group_id=1, query="第一段保留")
        self.assertTrue(any("第一段还要保留" in hit.text for hit in hits))
        self.assertFalse(any("第二段要撤回" in hit.text for hit in hits))

    def test_model_format_redacts_accounts_paths_tokens_and_urls(self):
        text = "QQ 3466734955 路径 /Users/test/work key token=secret https://internal.example/x"
        redacted = redact_for_model(text)
        self.assertNotIn("3466734955", redacted)
        self.assertNotIn("/Users/test", redacted)
        self.assertNotIn("secret", redacted)
        self.assertNotIn("internal.example", redacted)

    def test_formatted_hits_are_explicitly_untrusted(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "以前聊过北桥", now))
        hit = self.store.retrieve(group_id=1, query="北桥")[0]
        payload = json.loads(self.store.format_hits([hit])[0])
        self.assertEqual(payload["source"], "untrusted_group_chat_memory")
        self.assertEqual(payload["chunk_id"], hit.chunk_id)
        self.assertEqual(payload["speakers"], ["member_a"])
        self.assertEqual(payload["messages"][0]["speaker"]["id"], "member_a")

    def test_formatted_reply_keeps_quote_ownership(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "bot", "你之前问的是北桥", now))
        self.store.add_message(self.message(
            1, "m2", "member_a", "对，就是这个", now + 20,
            reply_message_id="m1", reply_speaker_id="bot", quoted_text="你之前问的是北桥",
        ))
        hits = self.store.retrieve(
            group_id=1, query="这个", reply_message_id="m2", participant_scope="reply_chain"
        )
        payloads = [json.loads(line) for line in self.store.format_hits(hits)]
        reply = next(
            message
            for payload in payloads
            for message in payload["messages"]
            if message["message_id"] == "m2"
        )
        self.assertEqual(reply["speaker"]["id"], "member_a")
        self.assertEqual(reply["reply_to"]["speaker_id"], "bot")

    def test_generated_for_keeps_bot_reply_on_actual_trigger_topic(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "咪咪怎么糖糖的", now))
        self.store.add_message(self.message(1, "m2", "member_b", "今天食堂几点关门", now + 1))
        self.store.add_message(MemoryMessage(
            group_id=1,
            message_id="b1",
            speaker_id="bot",
            display_name="机器人",
            speaker_role="bot",
            text="谁糖了？我这叫大智若愚。",
            event_time=now + 2,
            generated_for_message_ids=("m1",),
            turn_id="bot:b1",
            reply_mode="chat",
            semantic_topic="调侃机器人",
        ))
        with self.store.connect() as connection:
            topics = {
                row["message_id"]: int(row["topic_id"])
                for row in connection.execute(
                    "SELECT message_id,topic_id FROM chat_messages WHERE group_id=1"
                )
            }
            relation = connection.execute(
                "SELECT relation_type,target_message_id FROM message_relations "
                "WHERE group_id=1 AND source_message_id='b1'"
            ).fetchone()
        self.assertEqual(topics["b1"], topics["m1"])
        self.assertNotEqual(topics["b1"], topics["m2"])
        self.assertEqual((relation["relation_type"], relation["target_message_id"]), ("generated_for", "m1"))

    def test_self_history_contains_trigger_bot_authorship_and_explicit_feedback(self):
        now = time.time()
        self.store.add_message(self.message(1, "m1", "member_a", "咪咪怎么糖糖的", now))
        self.store.add_message(MemoryMessage(
            group_id=1,
            message_id="b1",
            speaker_id="bot",
            display_name="机器人",
            speaker_role="bot",
            text="谁糖了？我这叫大智若愚。",
            event_time=now + 1,
            generated_for_message_ids=("m1",),
            turn_id="bot:b1",
            reply_mode="chat",
        ))
        self.store.add_message(self.message(
            1,
            "m2",
            "member_b",
            "你刚才接的是第一句",
            now + 2,
            reply_message_id="b1",
            reply_speaker_id="bot",
            quoted_text="谁糖了？我这叫大智若愚。",
        ))
        payload = json.loads(self.store.format_self_history(
            group_id=1,
            related_message_ids=("m2",),
        )[0])
        self.assertEqual(payload["source"], "bot_self_history")
        self.assertTrue(payload["bot_message"]["speaker"]["is_self"])
        self.assertEqual(payload["bot_message"]["generated_for_message_ids"], ["m1"])
        self.assertEqual(payload["trigger_messages"][0]["message_id"], "m1")
        self.assertEqual(payload["feedback_messages"][0]["message_id"], "m2")

    def test_existing_database_is_migrated_for_bot_turn_fields(self):
        old_path = Path(self.temp_dir.name) / "old.sqlite3"
        with sqlite3.connect(old_path) as connection:
            connection.execute(
                """CREATE TABLE chat_messages (
                    id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL,
                    message_id TEXT NOT NULL, speaker_id TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '', speaker_role TEXT NOT NULL,
                    text TEXT NOT NULL, event_time REAL NOT NULL,
                    reply_message_id TEXT NOT NULL DEFAULT '',
                    reply_speaker_id TEXT NOT NULL DEFAULT '', quoted_text TEXT NOT NULL DEFAULT '',
                    mentions_json TEXT NOT NULL DEFAULT '[]', utterance_id INTEGER,
                    topic_id INTEGER, recalled INTEGER NOT NULL DEFAULT 0,
                    searchable INTEGER NOT NULL DEFAULT 1, UNIQUE(group_id,message_id)
                )"""
            )
        migrated = ChatMemoryStore(old_path, HashedNgramEmbedding(dimensions=32))
        with migrated.connect() as connection:
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(chat_messages)")}
            relation_table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='message_relations'"
            ).fetchone()
        self.assertTrue({"generated_for_message_ids_json", "turn_id", "reply_mode", "semantic_topic"} <= columns)
        self.assertIsNotNone(relation_table)


if __name__ == "__main__":
    unittest.main()
