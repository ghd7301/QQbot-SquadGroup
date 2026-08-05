"""OneBot v11 ingress over reverse WebSocket (design doc §4, §5).

NapCat connects to us as a WebSocket client. We receive `message` events, build a
MessageEnvelope, and forward it downstream. Outbound sends (group messages) go
back over the same connection via OneBot actions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable

import aiohttp
from aiohttp import web

from .config import Config
from .envelope import MessageEnvelope

logger = logging.getLogger("qqbot.ingress")

EnvelopeFn = Callable[[MessageEnvelope], Awaitable[None]]

_AT_RE = re.compile(r"@(\d+)")


def _extract(message, bot_qq: str):
    """Return (text, mentioned, mentions_other, reply_message_id, segments)."""
    text_parts: list[str] = []
    mentioned = False
    mentions_other = False
    reply_message_id = ""
    segments: list[dict] = []

    def consume(seg):
        nonlocal mentioned, mentions_other, reply_message_id
        t = seg.get("type")
        d = seg.get("data", {})
        if t == "text":
            text_parts.append(d.get("text", ""))
        elif t == "at":
            qq = str(d.get("qq", ""))
            if qq == bot_qq:
                mentioned = True
                # 剥掉对 bot 自身的 @，避免把 "@3119065126" 带进 question
            else:
                mentions_other = True
                text_parts.append(f"@{qq}")
        elif t == "reply":
            reply_message_id = str(d.get("id", ""))
        elif t == "image":
            text_parts.append("[图片]")
        else:
            text_parts.append("")

    if isinstance(message, str):
        segments = [{"type": "text", "data": {"text": message}}]
        for m in _AT_RE.finditer(message):
            if m.group(1) == bot_qq:
                mentioned = True
            else:
                mentions_other = True
        text = message
    else:
        for seg in message:
            consume(seg)
        text = "".join(text_parts)

    return text.strip(), mentioned, mentions_other, reply_message_id, segments


class OneBotIngress:
    def __init__(self, config: Config, on_message: EnvelopeFn) -> None:
        self.config = config
        self.on_message = on_message
        self._app = web.Application()
        self._app.router.add_get("/onebot", self._ws_handler)
        self._app.router.add_get("/onebot/", self._ws_handler)
        self._runner: web.AppRunner | None = None
        self._connections: set[web.WebSocketResponse] = set()
        self._echo_futures: dict[int, asyncio.Future] = {}
        self._echo_seq = 0

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._connections.add(ws)
        logger.info("OneBot client connected")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._dispatch(msg.data)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    break
        finally:
            self._connections.discard(ws)
            logger.info("OneBot client disconnected")
        return ws

    async def _dispatch(self, raw: str) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return
        if event.get("post_type") != "message" or event.get("message_type") != "group":
            # Still capture action responses if present.
            if "echo" in event and event["echo"] in self._echo_futures:
                fut = self._echo_futures.pop(event["echo"])
                if not fut.done():
                    fut.set_result(event)
            return
        env = self._to_envelope(event)
        if env is not None:
            await self.on_message(env)

    def _to_envelope(self, event: dict) -> MessageEnvelope | None:
        group_id = event.get("group_id")
        user_id = str(event.get("user_id", ""))
        message_id = str(event.get("message_id", ""))
        if not group_id or not user_id:
            return None
        text, mentioned, mentions_other, reply_message_id, segments = _extract(
            event.get("message", ""), self.config.bot_qq
        )
        explicit_cmd = False
        if text.startswith(("/知识", "/kb", "/knowledge")):
            explicit_cmd = True
            text = text.split(" ", 1)[1] if " " in text else ""

        return MessageEnvelope(
            group_id=group_id,
            user_id=user_id,
            message_id=message_id,
            event_time=int(event.get("time", 0)),
            sender_role=event.get("sender", {}).get("role", ""),
            mentioned=mentioned,
            mentions_other=mentions_other,
            explicit_knowledge_command=explicit_cmd,
            reply_message_id=reply_message_id,
            content_segments=segments,
            raw_text=text,
            question=text,
        )

    async def send_group_msg(self, group_id: int, text: str) -> dict | None:
        if not self._connections:
            logger.warning("no OneBot connection; cannot send")
            return None
        ws = next(iter(self._connections))
        self._echo_seq += 1
        echo = self._echo_seq
        payload = {
            "action": "send_group_msg",
            "params": {"group_id": group_id, "message": text},
            "echo": echo,
        }
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._echo_futures[echo] = fut
        await ws.send_json(payload)
        try:
            return await asyncio.wait_for(fut, timeout=10)
        except asyncio.TimeoutError:
            self._echo_futures.pop(echo, None)
            return None

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.onebot_ws_host, self.config.onebot_ws_port)
        await site.start()
        logger.info("OneBot ingress listening on %s:%d/onebot/", self.config.onebot_ws_host, self.config.onebot_ws_port)

    async def stop(self) -> None:
        for ws in list(self._connections):
            await ws.close()
        if self._runner:
            await self._runner.cleanup()
