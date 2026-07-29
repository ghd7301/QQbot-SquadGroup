from __future__ import annotations


def build_health_payload(deps) -> dict:
    scene_groups, scene_updating = deps.chat_scene_state.counts()
    memory_status = (
        deps.chat_memory_manager.status() if deps.chat_memory_manager else {}
    )
    pending_counts = deps.pending_status_counts()

    return {
        "ok": True,
        "chunks": len(deps.kb.chunks),
        "queued": (
            deps.message_queue.qsize()
            + deps.normal_message_queue.qsize()
            + deps.chat_queue.qsize()
        ),
        "priority_queued": deps.message_queue.qsize(),
        "normal_queued": deps.normal_message_queue.qsize(),
        "chat_queued": deps.chat_queue.qsize(),
        "fragment_buffered": deps.fragment_aggregator.buffered_count(),
        "pending": pending_counts.get("queued", 0)
        + pending_counts.get("retry", 0),
        "pending_retry": pending_counts.get("retry", 0),
        "pending_dispatching": pending_counts.get("dispatching", 0),
        "pending_dead_letter": pending_counts.get("dead_letter", 0),
        "pending_sent_unknown": pending_counts.get("sent_unknown", 0),
        "scene_groups": scene_groups,
        "scene_updating": scene_updating,
        "semantic_planner": deps.semantic_planner_health_snapshot(),
        "memory_messages": memory_status.get("messages", 0),
        "memory_chunks": memory_status.get("chunks", 0),
        "memory_topic_relations": memory_status.get("topic_relations", 0),
        "memory_queued": memory_status.get("queued", 0),
        "memory_paused": memory_status.get("paused", False),
    }
