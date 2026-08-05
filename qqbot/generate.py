"""Reply generation (design doc §7.2, §7.3).

Two generators share one LLM client: chat (no knowledge) and knowledge (chunks
injected into the system prompt). Both may veto with should_reply=false.
"""
from __future__ import annotations

import logging

from .context import build_chat_context, build_knowledge_context, format_chunks
from .envelope import MessageEnvelope
from .knowledge import Chunk
from .llm import LLMClient
from .prompts import CHAT_SYSTEM, KNOWLEDGE_SYSTEM

logger = logging.getLogger("qqbot.generate")


def _parse(data: dict | None) -> dict:
    data = data or {}
    return {
        "should_reply": bool(data.get("should_reply", False)),
        "reply": str(data.get("reply", "")).strip(),
    }


async def generate_chat(env: MessageEnvelope, llm: LLMClient, config) -> dict:
    ctx = build_chat_context(env, config)
    resp = await llm.chat(CHAT_SYSTEM, ctx, json_mode=True)
    return _parse(resp.data)


async def generate_knowledge(
    env: MessageEnvelope, chunks: list[Chunk], llm: LLMClient, config
) -> dict:
    system = KNOWLEDGE_SYSTEM.replace("{chunks}", format_chunks(chunks))
    user = build_knowledge_context(env, config)
    resp = await llm.chat(system, user, json_mode=True)
    return _parse(resp.data)
