"""3-second aggregation window (design doc §4, §5).

Messages from the same group + same user arriving within the window are merged
into one MessageEnvelope before entering the pipeline, so a user firing three
quick lines is treated as a single turn.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from .config import Config
from .envelope import MessageEnvelope

logger = logging.getLogger("qqbot.aggregate")

HistoryFn = Callable[[int], list[dict]]
FlushFn = Callable[[MessageEnvelope], Awaitable[None]]


class Aggregator:
    def __init__(self, config: Config, on_flush: FlushFn, get_history: HistoryFn) -> None:
        self.config = config
        self.on_flush = on_flush
        self.get_history = get_history
        self._pending: dict[tuple[int, str], dict] = {}

    async def add(self, env: MessageEnvelope) -> None:
        key = (env.group_id, env.user_id)
        pend = self._pending.setdefault(key, {"items": [], "timer": None})
        pend["items"].append(env)
        if pend["timer"] is None:
            pend["timer"] = asyncio.create_task(self._schedule(key))

    async def _schedule(self, key: tuple[int, str]) -> None:
        await asyncio.sleep(self.config.agg_window_sec)
        await self._flush(key)

    async def _flush(self, key: tuple[int, str]) -> None:
        pend = self._pending.pop(key, None)
        if not pend:
            return
        items = sorted(pend["items"], key=lambda e: e.event_time)
        merged = items[-1]
        question = "\n".join(i.raw_text for i in items if i.raw_text).strip() or merged.question
        merged.question = question
        merged.aggregated_from = [i.message_id for i in items]
        hist = self.get_history(merged.group_id)
        merged.recent_context = hist
        merged.chat_sequence = len(hist)
        logger.debug("flush %s (%d msgs)", key, len(items))
        await self.on_flush(merged)
