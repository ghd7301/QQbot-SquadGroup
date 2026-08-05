"""Program-level safety checks (design doc §11).

No second LLM review. Deterministic rules only:
  - numeric grounding: knowledge replies must not invent numbers absent from chunks
  - length cap
  - identity protection (no leaking user ids)
  - de-duplication against recent replies
"""
from __future__ import annotations

import logging
import re
import time

from .config import Config
from .envelope import MessageEnvelope
from .knowledge import Chunk

logger = logging.getLogger("qqbot.safety")

UNCERTAIN_REPLY = "这个我不太确定，你可以问问群里的老人"
_NUMBER_RE = re.compile(r"\d{2,}(?:\.\d+)?")
_QQ_RE = re.compile(r"\b\d{5,}\b")


def _extract_numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER_RE.finditer(text)]


def check(
    env: MessageEnvelope,
    reply: str,
    *,
    reply_type: str,
    chunks: list[Chunk] | None = None,
    config: Config,
    recent_replies: list[tuple[float, str]] | None = None,
) -> tuple[bool, str | None, str]:
    """Return (ok, reason, reply). ok=False means drop; reply may be replaced."""
    if not reply:
        return False, "empty", ""

    # Length cap
    if len(reply) > config.max_reply_chars:
        reply = reply[: config.max_reply_chars].rstrip() + "…"

    # Identity protection
    if env.user_id and env.user_id in reply:
        reply = reply.replace(env.user_id, "你")
    for m in _QQ_RE.findall(reply):
        reply = reply.replace(m, "某人")

    # De-duplication
    now = time.time()
    for ts, prev in recent_replies or []:
        if now - ts <= config.dedup_window_sec and prev == reply:
            return False, "duplicate", ""

    # Numeric grounding (knowledge only, when we have chunks)
    if reply_type == "knowledge" and chunks:
        corpus = "\n".join(c.text for c in chunks)
        for num in _extract_numbers(reply):
            if num not in corpus:
                logger.info("numeric grounding failed for %s; replying uncertain", num)
                return True, "numeric_grounding", UNCERTAIN_REPLY

    return True, None, reply
