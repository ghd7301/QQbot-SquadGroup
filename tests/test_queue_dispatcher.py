import queue
import threading
import unittest
from types import SimpleNamespace

from squad_bot.queueing import dispatcher


class QueueDispatcherTests(unittest.TestCase):
    def test_enqueue_persists_sequence_and_routes_priority(self) -> None:
        persisted = []
        deps = SimpleNamespace(
            next_sequence=lambda: 7,
            persist_pending_message=lambda priority, sequence, item: persisted.append(
                (priority, sequence, item)
            )
            or 11,
            message_queue=queue.Queue(),
            normal_message_queue=queue.Queue(),
        )

        pending_id = dispatcher.enqueue_persistent_message(
            deps,
            0,
            {"question": "优先消息"},
        )
        priority, sequence, item = deps.message_queue.get_nowait()

        self.assertEqual(pending_id, 11)
        self.assertEqual(persisted, [(0, 7, {"question": "优先消息"})])
        self.assertEqual((priority, sequence), (0, 7))
        self.assertEqual(item["_pending_id"], 11)
        self.assertTrue(deps.normal_message_queue.empty())

    def test_restore_advances_sequence_and_requeues_items(self) -> None:
        queued = []
        item = {"_pending_next_attempt_at": 0}
        deps = SimpleNamespace(
            sequence_number=2,
            sequence_lock=threading.Lock(),
            recover_incomplete_pending_dispatches=lambda: 0,
            load_pending_messages=lambda **_kwargs: [(1, 9, item)],
            _queue_pending_item=lambda payload, **kwargs: queued.append(
                (dict(payload), kwargs["delay"])
            ),
        )

        restored = dispatcher.restore_pending_messages(deps)

        self.assertEqual(restored, 1)
        self.assertEqual(deps.sequence_number, 9)
        self.assertEqual(queued[0][0]["_queue_priority"], 1)
        self.assertEqual(queued[0][0]["_queue_sequence"], 9)
        self.assertEqual(queued[0][1], 0)


if __name__ == "__main__":
    unittest.main()
