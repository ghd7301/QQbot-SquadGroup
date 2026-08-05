"""Single semantic classifier (design doc §3.2, §7.1).

One responsibility-only call: decide task (skip / chat / knowledge) and, for
knowledge, a normalised retrieval query. It does NOT generate the reply.
"""
from __future__ import annotations

import logging

from .context import build_classification_context
from .envelope import MessageEnvelope
from .llm import LLMClient
from .prompts import CLASSIFIER_SYSTEM

logger = logging.getLogger("qqbot.classifier")


class ClassificationResult:
    def __init__(self, task: str, knowledge_query: str = "", confidence: float = 0.5) -> None:
        self.task = task if task in ("skip", "chat", "knowledge") else "skip"
        self.knowledge_query = knowledge_query.strip()
        self.confidence = float(confidence)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Classification(task={self.task!r}, kq={self.knowledge_query!r}, conf={self.confidence})"


async def classify(env: MessageEnvelope, llm: LLMClient, config) -> ClassificationResult:
    ctx = build_classification_context(env, config)
    try:
        resp = await llm.chat(CLASSIFIER_SYSTEM, ctx, json_mode=True)
        data = resp.data or {}
    except Exception as e:  # noqa: BLE001 - classifier failure => skip, never crash
        logger.warning("classify failed: %s", e)
        return ClassificationResult("skip", confidence=0.0)
    return ClassificationResult(
        task=data.get("task", "skip"),
        knowledge_query=data.get("knowledge_query", ""),
        confidence=data.get("confidence", 0.5),
    )
