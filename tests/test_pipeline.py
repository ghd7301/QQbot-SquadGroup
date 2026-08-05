import asyncio
import os
import time

from qqbot.audit import Auditor
from qqbot.config import Config
from qqbot.embedding import HashedNgramEmbedding
from qqbot.envelope import MessageEnvelope
from qqbot.knowledge import KnowledgeBase
from qqbot.llm import LLMClient
from qqbot.memory import MemoryStore
from qqbot.send import Sender
from qqbot.server import Bot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeIngress:
    """Stub so Sender (dry_run) never needs a real OneBot connection."""


class CapAuditor(Auditor):
    def __init__(self, path):
        super().__init__(path=path)
        self.entries = []

    def log(self, entry):
        self.entries.append(entry)
        super().log(entry)


def _make_fake():
    async def fake(system, user):
        if "判断一条群消息是否需要你回复" in system:
            return '{"task":"knowledge","knowledge_query":"医疗兵 玩法","confidence":0.9}'
        if "基于以下知识资料" in system:
            return '{"should_reply":true,"reply":"医疗兵是支援兵种，负责救治伤员。"}'
        if "群里的老兵" in system and "闲聊" in system:
            return '{"should_reply":true,"reply":"哈哈"}'
        return '{"should_reply":false,"reply":""}'

    return fake


def test_full_pipeline_knowledge(tmp_path):
    async def run():
        cfg = Config.from_env()
        cfg.dry_run = True
        cfg.whitelisted_groups = [1]
        cfg.bot_qq = "b"
        cfg.knowledge_dir = os.path.join(ROOT, "knowledge")
        cfg.db_path = str(tmp_path / "t.db")

        llm = LLMClient(cfg, fake=_make_fake())
        kb = KnowledgeBase(HashedNgramEmbedding())
        kb.load(cfg.knowledge_dir)
        await kb.embed_all()

        from qqbot.store import Store

        store = Store(cfg)
        memory = MemoryStore(store)
        sender = Sender(cfg, FakeIngress())
        auditor = CapAuditor(str(tmp_path / "audit.jsonl"))
        bot = Bot(cfg, llm, kb, store, memory, sender, auditor, queue=None)

        env = MessageEnvelope(
            group_id=1, user_id="u", message_id="m", event_time=int(time.time()),
            question="医疗兵怎么玩", mentioned=True,
        )
        await bot.handle(env)

        assert any(e.get("action") == "reply" for e in auditor.entries), auditor.entries
        assert any("医疗兵" in (e.get("reply") or "") for e in auditor.entries)

    asyncio.run(run())


def test_pipeline_skip_when_not_addressed():
    async def run():
        cfg = Config.from_env()
        cfg.dry_run = True
        cfg.whitelisted_groups = [1]
        cfg.bot_qq = "b"
        cfg.knowledge_dir = os.path.join(ROOT, "knowledge")
        cfg.db_path = ":memory:"

        llm = LLMClient(cfg, fake=_make_fake())
        kb = KnowledgeBase(HashedNgramEmbedding())
        kb.load(cfg.knowledge_dir)
        await kb.embed_all()

        from qqbot.store import Store

        store = Store(cfg)
        memory = MemoryStore(store)
        sender = Sender(cfg, FakeIngress())
        auditor = CapAuditor("/tmp/_qqbot_audit_test.jsonl")
        bot = Bot(cfg, llm, kb, store, memory, sender, auditor, queue=None)

        # Not mentioned, no question -> gate should drop it before any LLM call.
        env = MessageEnvelope(
            group_id=1, user_id="u", message_id="m2", event_time=int(time.time()), question="哈哈哈哈"
        )
        await bot.handle(env)
        assert auditor.entries == [], auditor.entries

    asyncio.run(run())
