"""Send + rate limit + per-group ordering (design doc §12).

Global per-minute rate cap, per-group serial send lock, end-to-end expiry drop,
and dry_run mode for development.
"""
from __future__ import annotations

import asyncio
import logging
import time

from .config import Config
from .ingress import OneBotIngress

logger = logging.getLogger("qqbot.send")


class Sender:
    def __init__(self, config: Config, ingress: OneBotIngress) -> None:
        self.config = config
        self.ingress = ingress
        self._rate_lock = asyncio.Lock()
        self._group_locks: dict[int, asyncio.Lock] = {}
        self._window_start = time.time()
        self._sent_in_window = 0

    async def send(self, group_id: int, text: str, deadline: float | None = None) -> bool:
        if deadline is not None and time.time() > deadline:
            logger.warning("message expired before send; dropped")
            return False

        async with self._rate_lock:
            now = time.time()
            if now - self._window_start >= 60:
                self._window_start = now
                self._sent_in_window = 0
            if self._sent_in_window >= self.config.rate_limit_per_min:
                logger.warning("rate limit reached; dropped")
                return False
            self._sent_in_window += 1

        async with self._group_locks.setdefault(group_id, asyncio.Lock()):
            if self.config.dry_run:
                logger.info("[dry_run] send to group %s: %s", group_id, text)
                return True
            result = await self.ingress.send_group_msg(group_id, text)
            return result is not None
