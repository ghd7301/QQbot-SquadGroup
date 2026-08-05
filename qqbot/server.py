"""Main processing pipeline and worker (design doc §4, §18).

Flow per message:
  prefilter -> candidate gate -> single classifier
    -> chat: generate
    -> knowledge: hybrid retrieve -> generate (deterministic-uncertain if no chunks)
  -> safety checks -> send -> record (chat log + memory) -> audit
"""
from __future__ import annotations

import asyncio
import logging
import time

from .audit import Auditor
from .classifier import classify
from .config import Config
from .envelope import MessageEnvelope
from .gate import gate
from .generate import generate_chat, generate_knowledge
from .ingress import OneBotIngress
from .knowledge import Chunk, KnowledgeBase
from .llm import LLMClient
from .memory import MemoryStore
from .prefilter import should_skip
from .queue import WorkQueue
from .safety import UNCERTAIN_REPLY, check
from .send import Sender
from .store import Store

logger = logging.getLogger("qqbot.server")

HISTORY_LIMIT = 60


class Bot:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        kb: KnowledgeBase,
        store: Store,
        memory: MemoryStore,
        sender: Sender,
        auditor: Auditor,
        queue: WorkQueue,
    ) -> None:
        self.config = config
        self.llm = llm
        self.kb = kb
        self.store = store
        self.memory = memory
        self.sender = sender
        self.auditor = auditor
        self.queue = queue
        self._history: dict[int, list[dict]] = {}
        self._recent_replies: list[tuple[float, str]] = []

    def get_history(self, group_id: int) -> list[dict]:
        return list(self._history.get(group_id, []))

    def _record(self, group_id: int, text: str, is_bot: bool, reply_to: str, ts: float, message_id: str = "") -> None:
        h = self._history.setdefault(group_id, [])
        h.append(
            {
                "message_id": message_id or ("bot-" + reply_to),
                "user_id": self.config.bot_qq if is_bot else "",
                "text": text,
                "is_bot": is_bot,
                "reply_to": reply_to,
                "time": ts,
            }
        )
        if len(h) > HISTORY_LIMIT:
            h[:] = h[-HISTORY_LIMIT:]

    async def handle(self, env: MessageEnvelope) -> None:
        self._record(env.group_id, env.question, False, env.reply_message_id, env.event_time, env.message_id)

        if should_skip(env, self.config):
            self.auditor.log({**env.to_audit(), "action": "skip_prefilter"})
            return
        if gate(env, self.config) == "skip":
            return

        start = time.time()
        deadline = start + self.config.end_to_end_sec

        cls = await classify(env, self.llm, self.config)
        if cls.task == "skip":
            self.auditor.log({**env.to_audit(), "action": "skip_classify", "confidence": cls.confidence})
            return

        chunks: list[Chunk] | None = None
        if cls.task == "chat":
            gen = await generate_chat(env, self.llm, self.config)
            reply_type = "chat"
        else:  # knowledge
            chunks = await self.kb.retrieve(
                cls.knowledge_query,
                env.question,
                top_k=self.config.knowledge_top_k,
                threshold=self.config.knowledge_threshold,
                lexical_weight=self.config.lexical_weight,
                vector_weight=self.config.vector_weight,
            )
            if not chunks:
                gen = {"should_reply": True, "reply": UNCERTAIN_REPLY}
            else:
                gen = await generate_knowledge(env, chunks, self.llm, self.config)
            reply_type = "knowledge"

        if not gen.get("should_reply") or not gen.get("reply"):
            self.auditor.log({**env.to_audit(), "action": "no_reply", "task": cls.task})
            return

        ok, reason, reply = check(
            env,
            gen["reply"],
            reply_type=reply_type,
            chunks=chunks,
            config=self.config,
            recent_replies=self._recent_replies,
        )
        if not ok:
            self.auditor.log({**env.to_audit(), "action": "dropped", "reason": reason})
            return

        sent = await self.sender.send(env.group_id, reply, deadline=deadline)

        now = time.time()
        self._recent_replies.append((now, reply))
        self._recent_replies = [(t, r) for t, r in self._recent_replies if now - t <= self.config.dedup_window_sec]
        self._record(env.group_id, reply, True, env.message_id, now)
        self.store.add_chat_log(env.group_id, env.user_id, env.question, reply, reply_type)
        self.memory.ingest_turn(env, reply, reply_type)
        self.auditor.log(
            {
                **env.to_audit(),
                "action": "reply" if sent else "send_failed",
                "reply_type": reply_type,
                "reply": reply,
                "knowledge_query": cls.knowledge_query,
                "chunks": [c.ref for c in (chunks or [])],
                "latency_ms": int((now - start) * 1000),
            }
        )


def _item_to_env(item: dict) -> MessageEnvelope:
    kwargs = {k: v for k, v in item.items() if k != "_jid"}
    return MessageEnvelope(**kwargs)


async def run_worker(bot: Bot, queue: WorkQueue) -> None:
    while True:
        item = await queue.get()
        try:
            await bot.handle(_item_to_env(item))
        except Exception:  # noqa: BLE001 - never let one bad message kill the worker
            logger.exception("handle failed for item %s", item.get("_jid"))
        finally:
            queue.clear(item)
            queue.task_done()
