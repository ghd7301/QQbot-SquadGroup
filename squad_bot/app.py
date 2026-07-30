from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer


def main(deps) -> None:
    """Initialize runtime state, start worker threads and serve HTTP."""
    memory_started = deps.initialize_chat_memory()
    restored = deps.restore_pending_messages()
    if restored:
        print(f"Restored pending messages: {restored}")
    loaded = deps.load_chat_history()
    if loaded:
        print(f"Loaded chat history: {loaded} entries")
        migrated = deps.migrate_loaded_chat_history_to_memory()
        if migrated:
            print(f"Queued chat history migration: {migrated} entries")
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
    print(f"Squad QQBot MVP listening on http://{deps.settings.host}:{deps.settings.port}")
    print(f"Knowledge chunks: {len(deps.kb.chunks)}")
    print(f"Chat memory: {'enabled' if memory_started else 'disabled'}")
    print(
        f"Allowed groups: {','.join(deps.settings.allowed_group_ids) or 'all'}, "
        f"max replies/min: {deps.settings.max_replies_per_minute}"
    )
    server.serve_forever()
