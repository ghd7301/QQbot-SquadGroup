"""Long-term memory (design doc §9).

V1 keeps it deliberately simple and asynchronous-friendly: explicit "remember this"
requests become memory immediately; everything else becomes a candidate that is
promoted only after enough evidence. Semantic extraction is deferred (per §9.2).
Recall is a lightweight lexical match used to seed context when needed.
"""
from __future__ import annotations

import logging
import re
import time

from .envelope import MessageEnvelope
from .store import Store

logger = logging.getLogger("qqbot.memory")

_EXPLICIT_RE = re.compile(r"(?:记住|记一下|以后叫我|叫我|记着)\s*(.+)", re.IGNORECASE)


class MemoryStore:
    def __init__(self, store: Store) -> None:
        self.store = store

    def ingest_turn(self, env: MessageEnvelope, reply: str, reply_type: str) -> None:
        m = _EXPLICIT_RE.search(env.question)
        if m:
            fact = m.group(1).strip().rstrip("。.!！")
            if fact:
                self.store.add_memory(fact, kind="explicit")
                logger.info("explicit memory stored: %s", fact)
                return
        # Otherwise a candidate for later promotion.
        self.store.add_memory_candidate(
            content=env.question, source_turn=env.message_id, explicit=False
        )

    def upgrade(self) -> int:
        return self.store.upgrade_candidates()

    def recall(self, query: str, k: int = 3) -> list[str]:
        memories = self.store.memory_contents()
        if not memories:
            return []
        q = set(query.lower())
        scored = sorted(
            memories,
            key=lambda m: len(set(m.lower()) & q),
            reverse=True,
        )
        return [m for m in scored[:k] if set(m.lower()) & q]

    def record_bot_reply(self, group_id: int, text: str) -> None:
        # Bot replies are not stored as user memory, but logged to chat log upstream.
        _ = group_id, text  # hook for future memory-from-reply extraction
