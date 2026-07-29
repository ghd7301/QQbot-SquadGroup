from __future__ import annotations
import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs


def create_handler(deps):

    class Handler(BaseHTTPRequestHandler):

        def _read_request_body(self) -> bytes:
            content_length = self.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                length = int(content_length)
                if length > 0:
                    return self.rfile.read(length)
            transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
            if "chunked" not in transfer_encoding:
                return b""
            body = bytearray()
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    size = int(line.split(b";", 1)[0], 16)
                except ValueError:
                    break
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if not trailer or trailer in {b"\r\n", b"\n", b""}:
                            break
                    break
                body.extend(self.rfile.read(size))
                self.rfile.read(2)
            return bytes(body)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                (scene_groups, scene_updating) = deps.chat_scene_state.counts()
                memory_status = (
                    deps.chat_memory_manager.status()
                    if deps.chat_memory_manager
                    else {}
                )
                pending_counts = deps.pending_status_counts()
                planner_health = deps.semantic_planner_health_snapshot()
                self._json(
                    200,
                    {
                        "ok": True,
                        "chunks": len(deps.kb.chunks),
                        "queued": deps.message_queue.qsize()
                        + deps.normal_message_queue.qsize()
                        + deps.chat_queue.qsize(),
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
                        "semantic_planner": planner_health,
                        "memory_messages": memory_status.get("messages", 0),
                        "memory_chunks": memory_status.get("chunks", 0),
                        "memory_topic_relations": memory_status.get(
                            "topic_relations", 0
                        ),
                        "memory_queued": memory_status.get("queued", 0),
                        "memory_paused": memory_status.get("paused", False),
                    },
                )
                return
            self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            raw_body = self._read_request_body()
            length = len(raw_body)
            try:
                text_body = raw_body.decode("utf-8")
                event = json.loads(text_body)
            except json.JSONDecodeError:
                try:
                    parsed = parse_qs(text_body)
                    event = {
                        key: values[0] if len(values) == 1 else values
                        for (key, values) in parsed.items()
                    }
                except Exception:
                    event = {}
                if not event:
                    debug_path = Path("work/last_bad_onebot_payload.txt")
                    debug_path.parent.mkdir(exist_ok=True)
                    debug_path.write_text(
                        f"content-type: {self.headers.get('Content-Type', '')}\ntransfer-encoding: {self.headers.get('Transfer-Encoding', '')}\ncontent-length: {length}\n\n{raw_body[:4000]!r}\n",
                        encoding="utf-8",
                    )
                    self._json(200, {"ok": True, "ignored": "bad payload"})
                    return
            except UnicodeDecodeError:
                debug_path = Path("work/last_bad_onebot_payload.txt")
                debug_path.parent.mkdir(exist_ok=True)
                debug_path.write_bytes(raw_body[:4000])
                self._json(200, {"ok": True, "ignored": "bad encoding"})
                return
            if self.path == "/onebot":
                Path("work/last_onebot_raw.json").parent.mkdir(exist_ok=True)
                Path("work/last_onebot_raw.json").write_text(
                    json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if self.path == "/ask":
                question = str(event.get("question", "")).strip()
                self._json(200, {"ok": True, "answer": deps.answer_question(question)})
                return
            if self.path != "/onebot":
                self._json(404, {"ok": False, "error": "not found"})
                return
            (status, payload) = deps.handle_onebot_event(event)
            self._json(status, payload)

    return Handler
