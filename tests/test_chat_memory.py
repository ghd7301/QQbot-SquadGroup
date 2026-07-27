import json
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


if __name__ == "__main__":
    unittest.main()
