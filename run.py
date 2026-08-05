"""Entrypoint: wire every component and run (design doc §18 step 10).

Usage:
    python run.py            # serve (needs OneBot client to connect)
    DRY_RUN=1 python run.py  # log what would be sent instead of sending
"""
from __future__ import annotations

import asyncio
import logging

from qqbot.aggregate import Aggregator
from qqbot.audit import Auditor
from qqbot.config import Config
from qqbot.embedding import build_embedding
from qqbot.ingress import OneBotIngress
from qqbot.knowledge import KnowledgeBase
from qqbot.llm import LLMClient
from qqbot.memory import MemoryStore
from qqbot.queue import WorkQueue
from qqbot.send import Sender
from qqbot.server import Bot, run_worker
from qqbot.store import Store


async def main() -> None:
    config = Config.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("qqbot")

    llm = LLMClient(config)
    provider = build_embedding(config)
    kb = KnowledgeBase(provider)
    n_chunks = kb.load(config.knowledge_dir)
    await kb.embed_all()
    log.info("knowledge base ready: %d chunks from %s", n_chunks, config.knowledge_dir)

    store = Store(config)
    memory = MemoryStore(store)
    ingress = OneBotIngress(config, on_message=lambda _e: None)  # wired below
    sender = Sender(config, ingress)
    auditor = Auditor()
    queue = WorkQueue(store)

    bot = Bot(config, llm, kb, store, memory, sender, auditor, queue)
    aggregator = Aggregator(
        config,
        on_flush=lambda env: asyncio.ensure_future(queue.put(env)),
        get_history=bot.get_history,
    )
    ingress.on_message = lambda env: asyncio.ensure_future(aggregator.add(env))

    restored = await queue.load_pending()
    if restored:
        log.info("restored %d pending messages from journal", restored)

    worker_task = asyncio.ensure_future(run_worker(bot, queue))
    await ingress.start()
    log.info("Squad QQBot (simplified) online. dry_run=%s", config.dry_run)
    try:
        await worker_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await ingress.stop()
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
