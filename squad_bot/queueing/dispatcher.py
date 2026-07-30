from __future__ import annotations

import threading
import time


def next_sequence(deps) -> int:
    with deps.sequence_lock:
        deps.sequence_number += 1
        return deps.sequence_number


def enqueue_persistent_message(deps, priority: int, item: dict) -> int:
    # When enqueuing a mentioned=true item (priority 0), check for and
    # supersede any existing queued entry with the same message_ids that
    # was dispatched as mentioned=false. This prevents the same message
    # from being processed twice through different queue lanes.
    if priority == 0:
        group_id = int(item.get("group_id") or 0)
        ids = []
        for candidate in item.get("message_ids") or (item.get("message_id"),):
            value = str(candidate or "").strip()
            if value:
                ids.append(value)
        if group_id and ids:
            duplicate_ids = deps.find_queued_duplicates(group_id, ids)
            for dup_id in duplicate_ids:
                deps.mark_pending_superseded(int(dup_id))
                print("Superseded pending entry", dup_id, "for group", group_id)
    sequence = deps.next_sequence()
    pending_id = deps.persist_pending_message(priority, sequence, item)
    queued_item = dict(item)
    queued_item["_pending_id"] = pending_id
    queued_item["_queue_priority"] = priority
    queued_item["_queue_sequence"] = sequence
    target_queue = deps.message_queue if priority == 0 else deps.normal_message_queue
    target_queue.put((priority, sequence, queued_item))
    return pending_id


def _queue_pending_item(deps, item: dict, *, delay: float = 0.0) -> None:
    raw_priority = item.get("_queue_priority")
    priority = int(
        raw_priority
        if raw_priority is not None
        else 0 if item.get("mentioned") or item.get("explicit_knowledge_command") else 1
    )
    raw_sequence = item.get("_queue_sequence")
    sequence = int(raw_sequence if raw_sequence is not None else deps.next_sequence())
    queued_item = dict(item)
    queued_item["_queue_priority"] = priority
    queued_item["_queue_sequence"] = sequence
    target_queue = deps.message_queue if priority == 0 else deps.normal_message_queue

    def enqueue() -> None:
        target_queue.put((priority, sequence, queued_item))

    if delay <= 0:
        enqueue()
        return
    timer = threading.Timer(delay, enqueue)
    timer.daemon = True
    timer.start()


def handle_pending_worker_failure(deps, item: dict, error: str) -> str:
    pending_id = item.get("_pending_id")
    if pending_id is None:
        return "untracked"
    if item.get("_dispatch_completed"):
        return "delivered"
    if item.get("_dispatch_started"):
        deps.mark_pending_sent_unknown(int(pending_id), error)
        return "sent_unknown"
    result = deps.mark_pending_failure(int(pending_id), error)
    if result.status == "retry":
        deps._queue_pending_item(
            item, delay=max(0.0, result.next_attempt_at - time.time())
        )
    return result.status


def begin_pending_dispatch(deps, item: dict) -> None:
    pending_id = item.get("_pending_id")
    dispatch_id = f"{pending_id or 'untracked'}:{time.time_ns()}"
    if pending_id is not None:
        deps.mark_pending_dispatch_started(int(pending_id), dispatch_id)
    item["_pending_dispatch_id"] = dispatch_id
    item["_dispatch_started"] = True


def restore_pending_messages(deps) -> int:
    recovered = deps.recover_incomplete_pending_dispatches()
    if recovered:
        print("Marked interrupted message dispatches as sent_unknown", recovered)
    cleaned = 0
    try:
        cleaned = deps.cleanup_stale_pending_messages()
    except (AttributeError, TypeError):
        pass
    if cleaned:
        print("Cleaned stale pending entries:", cleaned)
    pending = deps.load_pending_messages(include_future=True)
    if not pending:
        return 0
    with deps.sequence_lock:
        deps.sequence_number = max(
            deps.sequence_number,
            max((sequence for (_priority, sequence, _item) in pending)),
        )
    for priority, sequence, item in pending:
        item["_queue_priority"] = priority
        item["_queue_sequence"] = sequence
        deps._queue_pending_item(
            item,
            delay=max(
                0.0, float(item.get("_pending_next_attempt_at") or 0) - time.time()
            ),
        )
    return len(pending)
