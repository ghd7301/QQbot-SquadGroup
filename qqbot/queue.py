"""In-memory work queue with SQLite-backed journal for restart recovery
(design doc §15 queue + persistence)."""
from __future__ import annotations

import asyncio
import logging

from .envelope import MessageEnvelope
from .store import Store

logger = logging.getLogger("qqbot.queue")

_PERSIST_FIELDS = (
    "group_id",
    "user_id",
    "message_id",
    "event_time",
    "question",
    "aggregated_from",
    "recent_context",
    "mentioned",
    "reply_target_user_id",
    "reply_message_id",
    "explicit_knowledge_command",
)


class WorkQueue:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._q: asyncio.Queue[dict] = asyncio.Queue()

    async def put(self, env: MessageEnvelope) -> None:
        item = {k: getattr(env, k) for k in _PERSIST_FIELDS}
        jid = self.store.push_queue(item)
        item["_jid"] = jid
        await self._q.put(item)

    async def get(self) -> dict:
        return await self._q.get()

    def task_done(self) -> None:
        self._q.task_done()

    async def load_pending(self) -> int:
        count = 0
        for _jid, item in self.store.pending_queue():
            item = dict(item)
            item["_jid"] = _jid
            await self._q.put(item)
            count += 1
        return count

    def clear(self, item: dict) -> None:
        jid = item.get("_jid")
        if jid is not None:
            self.store.clear_queue_item(jid)
