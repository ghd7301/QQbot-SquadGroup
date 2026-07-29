from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .chat_memory import MemoryHit
    from .knowledge import ContextResult
    from .llm import SemanticTopicCandidate, SubjectCandidate


@dataclass
class ProcessingDecision:
    should_reply: bool
    reason: str
    has_context: bool = False
    sources: tuple[str, ...] = ()
    effective_question: str = ""
    followup_of: str = ""
    followup_scope: str = ""
    reply_mode: str = ""
    chat_context: tuple[str, ...] = ()
    retrieval_score: float = 0.0
    retrieval_coverage: float = 0.0
    knowledge_query: str = ""
    knowledge_result: ContextResult | None = None
    semantic_intent: str = ""
    semantic_topic: str = ""
    implicit_meaning: str = ""
    capability: str = "none"
    draft_reply: str = ""
    semantic_confidence: float = 0.0
    memory_context: tuple[str, ...] = ()
    memory_query: str = ""
    memory_needed: bool = False
    memory_hit_count: int = 0
    memory_retrieval_attempted: bool = False
    memory_retrieval_mode: str = ""
    memory_candidate_count: int = 0
    memory_rejection_reason: str = ""
    recent_context_candidate_count: int = 0
    risk_flags: tuple[str, ...] = ()
    topic_candidates: tuple[SemanticTopicCandidate, ...] = ()
    recent_context_selected_count: int = 0
    recent_context_chars: int = 0
    memory_context_chars: int = 0
    context_deduplicated_count: int = 0
    recent_context_selected_ids: tuple[str, ...] = ()
    memory_selected_chunk_ids: tuple[int, ...] = ()
    memory_selected_by_planner: bool = False
    self_history_context: tuple[str, ...] = ()
    self_history_candidate_count: int = 0
    self_history_selected_count: int = 0
    self_history_chars: int = 0
    self_history_selected_message_ids: tuple[str, ...] = ()
    self_history_reasons: tuple[str, ...] = ()
    reply_regenerated: bool = False
    subject_candidates: tuple[SubjectCandidate, ...] = ()
    subject_ambiguity: str = "unknown"
    bot_involvement: str = "uncertain"
    reply_perspective: str = "neutral"
    semantic_audience: str = "unclear"
    participation_role: str = "uncertain"
    plan_context_revision: int = 0
    plan_scene_version: int = 0
    related_message_ids: tuple[str, ...] = ()
    semantic_replan_count: int = 0
    semantic_replan_reason: str = ""
    planner_status: str = "not_run"
    planner_latency_ms: int = 0


@dataclass(frozen=True)
class PendingFailureResult:
    status: str
    attempts: int
    next_attempt_at: float = 0.0


@dataclass(frozen=True)
class MemoryProbeResult:
    query: str = ""
    hits: tuple[MemoryHit, ...] = ()
    context: tuple[str, ...] = ()
    attempted: bool = False
    rejection_reason: str = ""


@dataclass
class ConversationState:
    last_question: str
    sources: tuple[str, ...]
    timestamp: float
    user_id: str = ""
    last_answer: str = ""
    reply_mode: str = "knowledge"
    bot_message_id: str = ""
    user_message_id: str = ""
    trigger_message_ids: tuple[str, ...] = ()
    turn_id: str = ""
    semantic_intent: str = ""
    semantic_topic: str = ""


@dataclass
class FollowupMatch:
    state: ConversationState
    scope: str


@dataclass
class GroupChatMessage:
    text: str
    user_id: str
    timestamp: float
    sequence: int = 0
    message_id: str = ""
    reply_message_id: str = ""
    reply_target_user_id: str = ""
    reply_text: str = ""
    mentioned_bot: bool = False
    mentioned_user_ids: tuple[str, ...] = ()
    display_name: str = ""
    generated_for_message_ids: tuple[str, ...] = ()
    turn_id: str = ""
    reply_mode: str = ""
    semantic_topic: str = ""
    received_time: float = 0.0
    content_segments: tuple[dict[str, str], ...] = ()
    message_status: str = "active"


@dataclass
class GroupChatScene:
    summary: str
    updated_at: float
    sequence: int


@dataclass
class MessageFragmentBuffer:
    group_id: int
    user_id: str
    audience: str
    item: dict
    parts: list[str]
    fragments: list[dict]
    started_at: float
    deadline: float
