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
            if (
                event.get("post_type") == "notice"
                and event.get("notice_type") == "group_recall"
            ):
                group_id = event.get("group_id")
                if (
                    not deps.settings.allowed_group_ids
                    or str(group_id) in deps.settings.allowed_group_ids
                ):
                    try:
                        deps.recall_group_chat_message(
                            int(group_id), str(event.get("message_id") or "")
                        )
                        deps.schedule_chat_history_save()
                    except (TypeError, ValueError):
                        pass
                self._json(200, {"ok": True, "recalled": True})
                return
            if event.get("message_type") != "group":
                print(
                    "Ignored event: not group",
                    event.get("post_type"),
                    event.get("message_type"),
                )
                self._json(200, {"ok": True, "ignored": "not group message"})
                return
            group_id = event.get("group_id")
            if (
                deps.settings.allowed_group_ids
                and str(group_id) not in deps.settings.allowed_group_ids
            ):
                print("Ignored event: group not allowed", group_id)
                (question, mentioned) = deps.extract_event_question(event)
                if mentioned:
                    deps.write_message_audit(
                        decision="ignored",
                        reason="group not allowed",
                        group_id=group_id,
                        user_id=event.get("user_id"),
                        question=question,
                        mentioned=mentioned,
                        event_time=event.get("time"),
                    )
                self._json(200, {"ok": True, "ignored": "group not allowed"})
                return
            raw_message = event.get("message", "")
            received_time = time.time()
            text = deps.extract_plain_text(raw_message)
            context_text = deps.extract_context_text(raw_message)
            content_segments = deps.extract_content_segments(raw_message)
            mentioned = deps.is_mentioned(deps.settings.bot_qq, raw_message)
            mentioned_user_ids = deps.extract_mentioned_user_ids(
                deps.settings.bot_qq, raw_message
            )
            user_id = event.get("user_id")
            if deps.settings.bot_qq and str(user_id) == deps.settings.bot_qq:
                self._json(200, {"ok": True, "ignored": "bot's own message"})
                return
            numeric_group_id = int(group_id)
            reply_message_id = deps.extract_reply_message_id(raw_message)
            reply_target_user_id = ""
            reply_text = ""
            if reply_message_id:
                (reply_target_user_id, reply_text) = deps.resolve_reply_message_context(
                    numeric_group_id, reply_message_id
                )
            chat_sequence = deps.record_group_chat_message(
                numeric_group_id,
                user_id,
                context_text,
                event.get("time"),
                message_id=str(event.get("message_id") or ""),
                reply_message_id=reply_message_id,
                reply_target_user_id=reply_target_user_id,
                reply_text=reply_text,
                mentioned_bot=mentioned,
                mentioned_user_ids=mentioned_user_ids,
                display_name=event.get("sender", {}).get("card")
                or event.get("sender", {}).get("nickname")
                or "",
                received_time=received_time,
                content_segments=content_segments,
            )
            deps.schedule_chat_history_save()
            deps.schedule_chat_scene_update(numeric_group_id, chat_sequence)
            try:
                deps.flush_fragment_buffer_for_new_speaker(
                    numeric_group_id, user_id, defer_dispatch=True
                )
            except Exception as exc:
                print("Fragment queue write failed:", repr(exc))
                self._json(503, {"ok": False, "error": "queue unavailable"})
                return
            if deps.is_event_too_old(event):
                deps.write_message_audit(
                    decision="ignored",
                    reason="message too old",
                    group_id=group_id,
                    user_id=user_id,
                    question=text,
                    mentioned=mentioned,
                    event_time=event.get("time"),
                )
                self._json(200, {"ok": True, "ignored": "message too old"})
                return
            (continue_processing, mentioned, reply_reason) = deps.classify_reply_target(
                reply_message_id, reply_target_user_id, mentioned, deps.settings.bot_qq
            )
            if not continue_processing:
                deps.flush_group_fragment_buffer(numeric_group_id, defer_dispatch=True)
                deps.write_message_audit(
                    decision="ignored",
                    reason=reply_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=text,
                    mentioned=mentioned,
                    reply_message_id=reply_message_id,
                    reply_target_user_id=reply_target_user_id,
                    event_time=event.get("time"),
                )
                self._json(200, {"ok": True, "ignored": reply_reason})
                return
            try:
                context_now = float(event.get("time"))
            except (TypeError, ValueError):
                context_now = time.time()
            chat_context = deps.recent_group_chat_context(
                numeric_group_id, now=context_now, focus_sequence=chat_sequence
            )
            explicit_knowledge_command = bool(
                deps.settings.command_prefix
                and text.strip().startswith(deps.settings.command_prefix)
            )
            (ok, question) = deps.should_respond(
                text, deps.settings.command_prefix, deps.settings.bot_qq, raw_message
            )
            print(
                "Group event",
                group_id,
                "mentioned",
                mentioned,
                "queued",
                ok,
                "question",
                question,
            )
            Path("work/last_onebot_event.json").parent.mkdir(exist_ok=True)
            Path("work/last_onebot_event.json").write_text(
                json.dumps(
                    {
                        "group_id": group_id,
                        "raw_message": raw_message,
                        "text": text,
                        "mentioned": mentioned,
                        "reply_message_id": reply_message_id,
                        "reply_target_user_id": reply_target_user_id,
                        "reply_reason": reply_reason,
                        "queued": ok,
                        "question": question,
                        "mentioned_user_ids": list(mentioned_user_ids),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not ok or not question:
                deps.write_message_audit(
                    decision="ignored",
                    reason="no trigger",
                    group_id=group_id,
                    user_id=event.get("user_id"),
                    question=question,
                    mentioned=mentioned,
                    event_time=event.get("time"),
                )
                self._json(200, {"ok": True, "ignored": "no trigger"})
                return
            item = {
                "group_id": group_id,
                "question": question,
                "mentioned": mentioned,
                "time": event.get("time"),
                "user_id": user_id,
                "sender_role": event.get("sender", {}).get("role", ""),
                "chat_context": list(chat_context),
                "mentions_other": bool(mentioned_user_ids),
                "mentioned_user_ids": list(mentioned_user_ids),
                "reply_message_id": reply_message_id,
                "reply_target_user_id": reply_target_user_id,
                "reply_text": reply_text,
                "message_id": str(event.get("message_id") or ""),
                "chat_sequence": chat_sequence,
                "received_time": received_time,
                "content_segments": list(content_segments),
                "message_status": "active",
                "explicit_knowledge_command": explicit_knowledge_command,
            }
            fragment_audience = deps.classify_fragment_audience(item)
            try:
                pending_ids = deps.submit_message_fragment(item, defer_dispatch=True)
            except Exception as exc:
                print("Pending queue write failed:", repr(exc))
                deps.write_message_audit(
                    decision="error",
                    reason=f"pending queue write failed: {exc!r}",
                    group_id=group_id,
                    user_id=event.get("user_id"),
                    question=question,
                    mentioned=mentioned,
                    event_time=event.get("time"),
                )
                self._json(503, {"ok": False, "error": "queue unavailable"})
                return
            if fragment_audience == "human":
                deps.write_message_audit(
                    decision="ignored",
                    reason="fragment directed at another member",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=False,
                    event_time=event.get("time"),
                )
                self._json(200, {"ok": True, "ignored": "directed at another member"})
                return
            self._json(
                200,
                {
                    "ok": True,
                    "buffered": not pending_ids,
                    "queued": bool(pending_ids),
                    "mentioned": mentioned,
                },
            )

    return Handler
