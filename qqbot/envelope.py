"""MessageEnvelope — unified message object built during preprocessing (design doc §5).

The envelope is the single read source for every downstream component. It is
populated in stages: ingress sets the identity/structure fields, the 3s window
sets the aggregation fields, and context building reads the short-term fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageEnvelope:
    # --- identity (from ingress) ---
    group_id: int
    user_id: str
    message_id: str
    event_time: int
    sender_role: str = ""
    mentioned: bool = False               # @ the bot
    mentions_other: bool = False          # @ someone else
    explicit_knowledge_command: bool = False

    # --- QQ structure (from ingress) ---
    reply_message_id: str = ""
    reply_target_user_id: str = ""
    reply_text: str = ""
    content_segments: list[Any] = field(default_factory=list)

    # --- aggregation result (from 3s window) ---
    question: str = ""                    # merged full text
    aggregated_from: list[str] = field(default_factory=list)

    # --- short-term context (from 3s window) ---
    recent_context: list[dict] = field(default_factory=list)
    chat_sequence: int = 0

    # --- runtime helpers (not part of the design contract) ---
    is_recalled: bool = False
    raw_text: str = ""

    def to_audit(self) -> dict:
        return {
            "group_id": self.group_id,
            "user_id": self.user_id,
            "message_id": self.message_id,
            "mentioned": self.mentioned,
            "question": self.question,
        }
