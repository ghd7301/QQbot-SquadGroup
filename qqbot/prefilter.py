"""Minimal pre-filter (design doc §10).

Cheap, deterministic gates that avoid wasting an LLM call. Deliberately does NOT
try to detect "is this a question" or "is this a follow-up" — those are the LLM's
job downstream.
"""
from __future__ import annotations

import time

from .config import Config
from .envelope import MessageEnvelope

MESSAGE_EXPIRY_SEC = 300


def _is_expired(event_time: int, now: int | None = None) -> bool:
    now = time.time() if now is None else now
    return (now - event_time) > MESSAGE_EXPIRY_SEC


def should_skip(env: MessageEnvelope, config: Config, now: int | None = None) -> bool:
    if env.group_id not in config.whitelisted_groups:
        return True
    if env.user_id == config.bot_qq:
        return True
    if _is_expired(env.event_time, now):
        return True
    if env.is_recalled:
        return True
    if len(env.question.strip()) < 3 and not env.mentioned:
        return True
    return False
