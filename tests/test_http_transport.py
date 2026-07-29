from __future__ import annotations

import json
import io
import os
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from squad_bot.transport.http import create_handler


class SceneState:
    def counts(self):
        return (2, 1)


class FragmentState:
    def buffered_count(self):
        return 3


def invoke_json(
    deps,
    method: str,
    path: str,
    payload: dict | None = None,
) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    handler = object.__new__(create_handler(deps))
    handler.path = path
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: setattr(handler, "response_status", status)
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None

    getattr(handler, f"do_{method}")()

    return handler.response_status, json.loads(handler.wfile.getvalue().decode("utf-8"))


class HttpTransportTests(unittest.TestCase):
    def test_health_reports_runtime_state(self) -> None:
        deps = SimpleNamespace(
            chat_scene_state=SceneState(),
            chat_memory_manager=SimpleNamespace(
                status=lambda: {
                    "messages": 8,
                    "chunks": 5,
                    "topic_relations": 3,
                    "queued": 1,
                    "paused": False,
                }
            ),
            pending_status_counts=lambda: {
                "queued": 2,
                "retry": 1,
                "dispatching": 1,
                "dead_letter": 0,
                "sent_unknown": 0,
            },
            semantic_planner_health_snapshot=lambda: {"addressed": {"attempts": 4}},
            kb=SimpleNamespace(chunks=[1, 2, 3]),
            message_queue=queue.Queue(),
            normal_message_queue=queue.Queue(),
            chat_queue=queue.Queue(),
            fragment_aggregator=FragmentState(),
        )
        deps.message_queue.put(object())

        status, payload = invoke_json(deps, "GET", "/health")

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["chunks"], 3)
        self.assertEqual(payload["queued"], 1)
        self.assertEqual(payload["pending"], 3)
        self.assertEqual(payload["scene_groups"], 2)
        self.assertEqual(payload["memory_messages"], 8)

    def test_ask_uses_live_dependency(self) -> None:
        deps = SimpleNamespace(answer_question=lambda question: f"回答：{question}")

        status, payload = invoke_json(
            deps,
            "POST",
            "/ask",
            {"question": "医疗兵怎么玩"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "answer": "回答：医疗兵怎么玩"})

    def test_onebot_delegates_event_processing(self) -> None:
        events = []
        deps = SimpleNamespace(
            handle_onebot_event=lambda event: events.append(event)
            or (202, {"ok": True, "accepted": True})
        )

        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                status, payload = invoke_json(
                    deps,
                    "POST",
                    "/onebot",
                    {"post_type": "meta_event"},
                )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(status, 202)
        self.assertEqual(payload, {"ok": True, "accepted": True})
        self.assertEqual(events, [{"post_type": "meta_event"}])


if __name__ == "__main__":
    unittest.main()
