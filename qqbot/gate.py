"""Low-cost candidate gating (design doc §4).

Goal: ordinary chatter that clearly needs no bot involvement never reaches the
LLM classifier. Explicit triggers (being @-mentioned, being replied to, explicit
knowledge command) always classify. Everything else classifies only when there is
a question or an active conversational hook.
"""
from __future__ import annotations

from .config import Config
from .envelope import MessageEnvelope

_QUESTION_CHARS = set("?？吗怎么如何为何啥多少几哪些哪咋为啥是否呢么")


def _looks_like_question(text: str) -> bool:
    return any(ch in text for ch in _QUESTION_CHARS)


def gate(env: MessageEnvelope, config: Config) -> str:
    """Return 'classify' or 'skip'."""
    # Always classify when the bot is explicitly addressed or commanded.
    if env.mentioned:
        return "classify"
    if env.reply_target_user_id and env.reply_target_user_id == config.bot_qq:
        return "classify"
    if env.explicit_knowledge_command:
        return "classify"
    # Otherwise only classify when there is a question or an active conversational hook.
    if _looks_like_question(env.question):
        return "classify"
    if env.reply_message_id:
        return "classify"
    if env.mentions_other:
        return "classify"
    return "skip"
