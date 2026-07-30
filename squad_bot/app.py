from __future__ import annotations

import logging
import threading
from http.server import ThreadingHTTPServer

logger = logging.getLogger("squad_bot")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(deps) -> None:
    """Initialize runtime state, start worker threads and serve HTTP."""
    _configure_logging()
    memory_started = deps.initialize_chat_memory()
    restored = deps.restore_pending_messages()
    if restored:
        logger.info("Restored pending messages: %s", restored)
    loaded = deps.load_chat_history()
    if loaded:
        logger.info("Loaded chat history: %s entries", loaded)
        migrated = deps.migrate_loaded_chat_history_to_memory()
        if migrated:
            logger.info("Queued chat history migration: %s entries", migrated)
    threading.Thread(
        target=deps.worker,
        args=(deps.message_queue, "priority"),
        daemon=True,
    ).start()
    threading.Thread(
        target=deps.chat_history_save_worker,
        daemon=True,
        name="chat-history-saver",
    ).start()
    threading.Thread(
        target=deps.fragment_aggregation_worker,
        daemon=True,
        name="message-fragment-aggregator",
    ).start()
    threading.Thread(
        target=deps.worker,
        args=(deps.normal_message_queue, "normal"),
        daemon=True,
    ).start()
    threading.Thread(target=deps.chat_worker, daemon=True).start()
    handler = deps.Handler
    server = ThreadingHTTPServer((deps.settings.host, deps.settings.port), handler)
    logger.info(
        "Squad QQBot MVP listening on http://%s:%s",
        deps.settings.host, deps.settings.port,
    )
    logger.info("Knowledge chunks: %s", len(deps.kb.chunks))
    logger.info("Chat memory: %s", "enabled" if memory_started else "disabled")
    logger.info(
        "Allowed groups: %s, max replies/min: %s",
        ",".join(deps.settings.allowed_group_ids) or "all",
        deps.settings.max_replies_per_minute,
    )
    server.serve_forever()
