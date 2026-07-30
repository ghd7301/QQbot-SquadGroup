from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
from typing import Sequence

from . import app as app_service
from . import pending_store
from . import chat_history as chat_history_service
from . import chat_scene as chat_scene_service
from . import conversation as conversation_service
from . import context_pipeline as context_pipeline_service
from . import answering as answering_service
from .message_fragments import (
    classify_audience,
    items_compatible,
    message_ids,
)
from . import admin as admin_service
from . import message_router, semantic_routing, worker_handlers
from .workers.priority import process_worker_item as _process_worker_item_impl
from .workers.chat import process_chat_item as _process_chat_item_impl
from .delivery import policies as delivery_policies
from .delivery import replies as reply_delivery
from .delivery import review as delivery_review
from .ingress import events as ingress_events
from .ingress import fragments as fragment_ingress
from .observability import audit as audit_observability
from .queueing import dispatcher as queue_dispatcher
from .queueing import store as queue_store
from .transport import http as http_transport
from .config import settings
from .chat_memory import ChatMemoryManager, ChatMemoryStore, MemoryHit, MemoryMessage, redact_for_model
from .embedding import build_embedding_provider
from .fact_guard import precise_fact_tokens, unsupported_fallback_precise_facts
from .knowledge import ContextResult
from .knowledge_routing import KnowledgeRoutingService, attach_result
from .runtime import BotRuntime
from .runtime_dependencies import RuntimeDependencies
from .worker_runtime import (
    PendingItemLifecycle,
    normal_lane_should_yield,
    run_chat_worker,
    run_worker,
)
from .models import (
    ConversationState,
    FollowupMatch,
    GroupChatMessage,
    GroupChatScene,
    MemoryProbeResult,
    MessageFragmentBuffer,
    PendingFailureResult,
    ProcessingDecision,
)
from .llm import (
    MessagePlan,
    SemanticTopicCandidate,
    SubjectCandidate,
    analyze_chat_scene,
    answer_chat,
    ask_fallback_llm,
    ask_llm,
    classify_bot_fragment_prefix,
    is_chat_no_reply,
    is_provider_refusal_text,
    normalize_model_answer,
    plan_group_message,
    review_candidate_reply,
    rewrite_contextual_question,
    verify_bot_capability,
)
from .onebot import (
    extract_content_segments,
    extract_mentioned_user_ids,
    extract_context_text,
    extract_plain_text,
    extract_reply_message_id,
    get_message_info,
    is_mentioned,
    send_group_msg,
    should_respond,
)


bot_runtime = BotRuntime(settings.knowledge_dir)
kb = bot_runtime.knowledge.base
message_queue = bot_runtime.queues.priority
normal_message_queue = bot_runtime.queues.normal
chat_queue = bot_runtime.queues.chat
reply_timestamps = bot_runtime.queues.reply_timestamps
rate_limit_lock = bot_runtime.queues.rate_limit_lock
kb_lock = bot_runtime.knowledge.lock
sequence_lock = bot_runtime.queues.sequence_lock
sequence_number = 0
audit_lock = bot_runtime.audit_lock
auto_reply_enabled = settings.auto_reply_enabled
topic_cooldown_lock = bot_runtime.conversation.topic_cooldown_lock
recent_reply_topics = bot_runtime.conversation.recent_reply_topics
chat_history_state = bot_runtime.conversation.history
chat_history_lock = chat_history_state.lock
group_chat_history = chat_history_state.messages
chat_message_sequence = 0
chat_reply_lock = bot_runtime.conversation.chat_reply_lock
group_send_locks_lock = bot_runtime.conversation.group_send_locks_lock
group_send_locks = bot_runtime.conversation.group_send_locks
chat_scene_state = bot_runtime.conversation.scene
chat_scene_lock = chat_scene_state.lock
group_chat_scenes = chat_scene_state.scenes
chat_scene_requested_sequence = chat_scene_state.requested_sequences
chat_scene_pending_messages = chat_scene_state.pending_messages
chat_scene_running = chat_scene_state.running
hostile_reply_lock = bot_runtime.conversation.hostile_reply_lock
hostile_reply_history = bot_runtime.conversation.hostile_reply_history
memory_clear_confirmations = bot_runtime.conversation.memory_clear_confirmations
fragment_aggregator = bot_runtime.conversation.fragments
fragment_condition = fragment_aggregator.condition
group_fragment_buffers = fragment_aggregator.buffers
ready_fragment_buffers = fragment_aggregator.ready
chat_memory_manager: ChatMemoryManager | None = None
chat_history_save_event = bot_runtime.conversation.history_save_event
knowledge_gap_lock = bot_runtime.knowledge.gap_lock
recent_knowledge_gap_queries = bot_runtime.knowledge.recent_gap_queries
semantic_planner_health = bot_runtime.planner_health
runtime_dependencies = RuntimeDependencies(globals())

# Constants migrated to conversation_service
from .conversation import (
    IDENTITY_KEYWORDS,
    SELF_REFERENCE_KEYWORDS,
    AUTO_REPLY_KEYWORDS,
    QUESTION_CUES,
    HELP_CUES,
    ASSIGNMENT_CUES,
    NON_BOT_TARGET_CUES,
    BROAD_CONTENT_CUES,
    META_CONTENT_CUES,
    BOT_META_CUES,
    CHAT_BLOCK_CUES,
    BARE_CHAT_REACTIONS,
    BIRTHDAY_CELEBRATION_CUES,
    BIRTHDAY_DISCUSSION_CUES,
    GRADUATION_CELEBRATION_CUES,
    GRADUATION_DISCUSSION_CUES,
    SEMANTIC_CHAT_INTENTS,
    LOCAL_REPLY_MODES,
)



COMMAND_ALIASES = {
    "重载知识库": "reload",
    "reload": "reload",
    "健康状态": "health",
    "查看健康状态": "health",
    "状态": "health",
    "health": "health",
    "最近跳过": "recent_skips",
    "最近跳过消息": "recent_skips",
    "skipped": "recent_skips",
    "skips": "recent_skips",
    "自动回复开": "auto_reply_on",
    "开启自动回复": "auto_reply_on",
    "打开自动回复": "auto_reply_on",
    "auto on": "auto_reply_on",
    "自动回复关": "auto_reply_off",
    "关闭自动回复": "auto_reply_off",
    "关掉自动回复": "auto_reply_off",
    "auto off": "auto_reply_off",
    "聊天记忆状态": "memory_status",
    "记忆状态": "memory_status",
    "memory status": "memory_status",
    "暂停聊天记忆": "memory_pause",
    "memory pause": "memory_pause",
    "恢复聊天记忆": "memory_resume",
    "memory resume": "memory_resume",
    "重建聊天记忆": "memory_rebuild",
    "memory rebuild": "memory_rebuild",
    "清空本群聊天记忆": "memory_clear_request",
    "清空本群聊天记忆 确认": "memory_clear_confirm",
    "最近知识未命中": "knowledge_gaps",
    "知识未命中": "knowledge_gaps",
    "knowledge gaps": "knowledge_gaps",
}


def _rotate_log_if_needed(
    log_path: Path, max_bytes: int = 5 * 1024 * 1024, keep: int = 5
) -> None:
    return audit_observability._rotate_log_if_needed(log_path, max_bytes, keep)



def write_message_audit(
    *,
    decision: str,
    reason: str,
    group_id=None,
    user_id=None,
    question: str = "",
    mentioned: bool = False,
    has_context: bool = False,
    sources: Sequence[str] = (),
    followup_of: str = "",
    followup_scope: str = "",
    reply_mode: str = "",
    retrieval_score: float = 0.0,
    retrieval_coverage: float = 0.0,
    model_latency_ms: int = 0,
    model_name: str = "",
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    chat_context: Sequence[str] = (),
    scene_context: str = "",
    answer: str = "",
    mention_user_id: str = "",
    semantic_intent: str = "",
    semantic_topic: str = "",
    implicit_meaning: str = "",
    capability: str = "none",
    semantic_confidence: float = 0.0,
    topic_candidates: Sequence[SemanticTopicCandidate] = (),
    subject_candidates: Sequence[SubjectCandidate] = (),
    subject_ambiguity: str = "unknown",
    bot_involvement: str = "uncertain",
    reply_perspective: str = "neutral",
    semantic_audience: str = "unclear",
    participation_role: str = "uncertain",
    plan_context_revision: int = 0,
    plan_scene_version: int = 0,
    related_message_ids: Sequence[str] = (),
    semantic_replan_count: int = 0,
    semantic_replan_reason: str = "",
    planner_status: str = "not_run",
    planner_latency_ms: int = 0,
    memory_query: str = "",
    memory_hit_count: int = 0,
    memory_retrieval_attempted: bool = False,
    memory_retrieval_mode: str = "",
    memory_candidate_count: int = 0,
    memory_rejection_reason: str = "",
    recent_context_candidate_count: int = 0,
    recent_context_selected_count: int = 0,
    recent_context_chars: int = 0,
    memory_context_chars: int = 0,
    context_deduplicated_count: int = 0,
    recent_context_selected_ids: Sequence[str] = (),
    memory_selected_chunk_ids: Sequence[int] = (),
    memory_selected_by_planner: bool = False,
    self_history_candidate_count: int = 0,
    self_history_selected_count: int = 0,
    self_history_chars: int = 0,
    self_history_selected_message_ids: Sequence[str] = (),
    self_history_reasons: Sequence[str] = (),
    bot_message_id: str = "",
    generated_for_message_ids: Sequence[str] = (),
    turn_id: str = "",
    event_time=None
) -> None:
    return audit_observability.write_message_audit(
        runtime_dependencies,
        decision=decision,
        reason=reason,
        group_id=group_id,
        user_id=user_id,
        question=question,
        mentioned=mentioned,
        has_context=has_context,
        sources=sources,
        followup_of=followup_of,
        followup_scope=followup_scope,
        reply_mode=reply_mode,
        retrieval_score=retrieval_score,
        retrieval_coverage=retrieval_coverage,
        model_latency_ms=model_latency_ms,
        model_name=model_name,
        reply_message_id=reply_message_id,
        reply_target_user_id=reply_target_user_id,
        chat_context=chat_context,
        scene_context=scene_context,
        answer=answer,
        mention_user_id=mention_user_id,
        semantic_intent=semantic_intent,
        semantic_topic=semantic_topic,
        implicit_meaning=implicit_meaning,
        capability=capability,
        semantic_confidence=semantic_confidence,
        topic_candidates=topic_candidates,
        subject_candidates=subject_candidates,
        subject_ambiguity=subject_ambiguity,
        bot_involvement=bot_involvement,
        reply_perspective=reply_perspective,
        semantic_audience=semantic_audience,
        participation_role=participation_role,
        plan_context_revision=plan_context_revision,
        plan_scene_version=plan_scene_version,
        related_message_ids=related_message_ids,
        semantic_replan_count=semantic_replan_count,
        semantic_replan_reason=semantic_replan_reason,
        planner_status=planner_status,
        planner_latency_ms=planner_latency_ms,
        memory_query=memory_query,
        memory_hit_count=memory_hit_count,
        memory_retrieval_attempted=memory_retrieval_attempted,
        memory_retrieval_mode=memory_retrieval_mode,
        memory_candidate_count=memory_candidate_count,
        memory_rejection_reason=memory_rejection_reason,
        recent_context_candidate_count=recent_context_candidate_count,
        recent_context_selected_count=recent_context_selected_count,
        recent_context_chars=recent_context_chars,
        memory_context_chars=memory_context_chars,
        context_deduplicated_count=context_deduplicated_count,
        recent_context_selected_ids=recent_context_selected_ids,
        memory_selected_chunk_ids=memory_selected_chunk_ids,
        memory_selected_by_planner=memory_selected_by_planner,
        self_history_candidate_count=self_history_candidate_count,
        self_history_selected_count=self_history_selected_count,
        self_history_chars=self_history_chars,
        self_history_selected_message_ids=self_history_selected_message_ids,
        self_history_reasons=self_history_reasons,
        bot_message_id=bot_message_id,
        generated_for_message_ids=generated_for_message_ids,
        turn_id=turn_id,
        event_time=event_time,
    )



def extract_event_question(event: dict) -> tuple[str, bool]:
    raw_message = event.get("message", "")
    text = extract_plain_text(raw_message)
    mentioned = is_mentioned(settings.bot_qq, raw_message)
    _ok, question = should_respond(text, settings.command_prefix, settings.bot_qq, raw_message)
    return question, mentioned


def classify_reply_target(
    reply_message_id: str,
    reply_target_user_id: str,
    mentioned: bool,
    bot_qq: str,
) -> tuple[bool, bool, str]:
    if not reply_message_id:
        return True, mentioned, ""
    if mentioned:
        return True, True, "explicit mention"
    if reply_target_user_id and reply_target_user_id == bot_qq:
        return True, True, "reply to bot"
    if reply_target_user_id:
        return False, False, "reply directed at another member"
    return False, False, "reply target unknown"


def normalize_command_text(message: str) -> str:
    return admin_service.normalize_command_text(runtime_dependencies, message)



def get_admin_command(message: str) -> str:
    return admin_service.get_admin_command(runtime_dependencies, message)



def is_restored_admin_command(item: dict) -> bool:
    return admin_service.is_restored_admin_command(runtime_dependencies, item)



def is_admin_user(user_id, sender_role: str = "") -> bool:
    return admin_service.is_admin_user(runtime_dependencies, user_id, sender_role)



def recent_audit_entries(limit: int = 5) -> list[dict]:
    return audit_observability.recent_audit_entries(runtime_dependencies, limit)



def answer_admin_command(command: str, *, group_id: int = 0, user_id: str = "") -> str:
    return admin_service.answer_admin_command(
        runtime_dependencies, command, group_id=group_id, user_id=user_id
    )



def topic_key(question: str, decision: ProcessingDecision) -> str:
    return conversation_service.topic_key(runtime_dependencies, question, decision)


def is_topic_on_cooldown(group_id: int, key: str) -> bool:
    return conversation_service.is_topic_on_cooldown(runtime_dependencies, group_id, key)


def mark_topic_replied(group_id: int, key: str) -> None:
    return conversation_service.mark_topic_replied(runtime_dependencies, group_id, key)


def check_and_mark_topic_replied(group_id: int, key: str) -> bool:
    return conversation_service.check_and_mark_topic_replied(
        runtime_dependencies, group_id, key,
    )


def followup_context_for(
    group_id: int,
    user_id,
    question: str,
    mentioned: bool,
    *,
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    reply_text: str = "",
    bot_qq: str = "",
    db_path: str | Path | None = None,
) -> FollowupMatch | None:
    return conversation_service.followup_context_for(
        runtime_dependencies,
        group_id,
        user_id,
        question,
        mentioned,
        reply_message_id=reply_message_id,
        reply_target_user_id=reply_target_user_id,
        reply_text=reply_text,
        bot_qq=bot_qq,
        db_path=db_path,
    )


def build_effective_question(question: str, followup_match: FollowupMatch | None) -> str:
    return conversation_service.build_effective_question(question, followup_match)


def build_generation_question(question: str, followup_match: FollowupMatch | None) -> str:
    return conversation_service.build_generation_question(question, followup_match)


def remember_conversation(
    group_id: int,
    user_id,
    question: str,
    decision: ProcessingDecision,
    *,
    answer: str = "",
    bot_message_id: str = "",
    user_message_id: str = "",
    trigger_message_ids: Sequence[str] = (),
    turn_id: str = "",
    db_path: str | Path | None = None,
) -> None:
    return conversation_service.remember_conversation(
        runtime_dependencies,
        group_id,
        user_id,
        question,
        decision,
        answer=answer,
        bot_message_id=bot_message_id,
        user_message_id=user_message_id,
        trigger_message_ids=trigger_message_ids,
        turn_id=turn_id,
        db_path=db_path,
    )


def record_group_chat_message(
    group_id: int,
    user_id,
    text: str,
    event_time=None,
    *,
    message_id: str = "",
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    reply_text: str = "",
    mentioned_bot: bool = False,
    mentioned_user_ids: Sequence[str] = (),
    display_name: str = "",
    generated_for_message_ids: Sequence[str] = (),
    turn_id: str = "",
    reply_mode: str = "",
    semantic_topic: str = "",
    received_time=None,
    content_segments: Sequence[dict[str, str]] = (),
    message_status: str = "active",
) -> int:
    return chat_history_service.record_group_chat_message(
        runtime_dependencies,
        group_id,
        user_id,
        text,
        event_time,
        message_id=message_id,
        reply_message_id=reply_message_id,
        reply_target_user_id=reply_target_user_id,
        reply_text=reply_text,
        mentioned_bot=mentioned_bot,
        mentioned_user_ids=mentioned_user_ids,
        display_name=display_name,
        generated_for_message_ids=generated_for_message_ids,
        turn_id=turn_id,
        reply_mode=reply_mode,
        semantic_topic=semantic_topic,
        received_time=received_time,
        content_segments=content_segments,
        message_status=message_status,
    )


def find_group_chat_message(group_id: int, message_id: str) -> GroupChatMessage | None:
    return chat_history_service.find_group_chat_message(
        runtime_dependencies, group_id, message_id
    )


def resolve_reply_message_context(
    group_id: int,
    message_id: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    return chat_history_service.resolve_reply_message_context(
        runtime_dependencies, group_id, message_id, db_path=db_path
    )


def stable_member_id(group_id: int, user_id: str) -> str:
    return chat_history_service.stable_member_id(
        runtime_dependencies, group_id, user_id
    )


def _context_message_payload(
    group_id: int,
    item: GroupChatMessage,
    *,
    current: bool,
) -> dict:
    return chat_history_service.context_message_payload(
        runtime_dependencies, group_id, item, current=current
    )


def recent_group_chat_context(
    group_id: int,
    *,
    now: float | None = None,
    context_seconds: int | None = None,
    max_messages: int | None = None,
    focus_sequence: int = 0,
    through_sequence: int = 0,
) -> tuple[str, ...]:
    return chat_history_service.recent_group_chat_context(
        runtime_dependencies,
        group_id,
        now=now,
        context_seconds=context_seconds,
        max_messages=max_messages,
        focus_sequence=focus_sequence,
        through_sequence=through_sequence,
    )


def current_group_chat_scene(
    group_id: int,
    *,
    focus_sequence: int = 0,
    now: float | None = None,
    stale_seconds: int | None = None,
) -> str:
    return chat_scene_service.current_group_chat_scene(
        runtime_dependencies,
        group_id,
        focus_sequence=focus_sequence,
        now=now,
        stale_seconds=stale_seconds,
    )


def chat_scene_enabled_for_group(group_id: int) -> bool:
    return chat_scene_service.chat_scene_enabled_for_group(
        runtime_dependencies, group_id
    )


def _finish_chat_scene_update(group_id: int) -> None:
    return chat_scene_service.finish_chat_scene_update(
        runtime_dependencies, group_id
    )


def _chat_scene_update_loop(group_id: int) -> None:
    return chat_scene_service.chat_scene_update_loop(
        runtime_dependencies, group_id
    )


def schedule_chat_scene_update(group_id: int, sequence: int) -> bool:
    return chat_scene_service.schedule_chat_scene_update(
        runtime_dependencies, group_id, sequence
    )


def has_substantive_chat_context(chat_context: Sequence[str]) -> bool:
    return conversation_service.has_substantive_chat_context(runtime_dependencies, chat_context)


def group_chat_has_newer_user_message(group_id: int, sequence: int) -> bool:
    return conversation_service.group_chat_has_newer_user_message(
        runtime_dependencies, group_id, sequence,
    )


def latest_group_user_sequence(group_id: int) -> int:
    return conversation_service.latest_group_user_sequence(runtime_dependencies, group_id)


def locked_send_context_change(
    group_id: int,
    reviewed_revision: int,
    item: dict,
) -> tuple[bool, tuple[str, ...], str]:
    return delivery_review.locked_send_context_change(
        runtime_dependencies, group_id, reviewed_revision, item
    )


def validate_locked_send(
    group_id: int,
    item: dict,
    reviewed_revision: int,
    *,
    check_context: bool = True,
) -> tuple[bool, str, str]:
    return delivery_review.validate_locked_send(
        runtime_dependencies,
        group_id,
        item,
        reviewed_revision,
        check_context=check_context,
    )


def unsafe_or_repeated_reply(group_id: int, answer: str, *, limit: int = 10) -> str:
    return delivery_review.unsafe_or_repeated_reply(
        runtime_dependencies, group_id, answer, limit=limit
    )


def is_recent_duplicate_group_message(
    group_id: int,
    text: str,
    *,
    focus_sequence: int,
    event_time=None,
    window_seconds: int = 60,
) -> bool:
    return delivery_review.is_recent_duplicate_group_message(
        runtime_dependencies,
        group_id,
        text,
        focus_sequence=focus_sequence,
        event_time=event_time,
        window_seconds=window_seconds,
    )


def reply_deadline(event_time, mentioned: bool) -> float:
    return delivery_review.reply_deadline(
        runtime_dependencies, event_time, mentioned
    )


def remaining_reply_timeout(
    deadline: float,
    *,
    cap: int,
    reserve: int = 0,
) -> int:
    return delivery_review.remaining_reply_timeout(
        deadline, cap=cap, reserve=reserve
    )


def review_and_refresh_answer(
    *,
    question: str,
    answer: str,
    decision: ProcessingDecision,
    group_id: int,
    mentioned: bool,
    admin: bool,
    deadline: float,
    baseline_revision: int | None = None,
    target_item: dict | None = None,
) -> tuple[str, str, int]:
    return delivery_review.review_and_refresh_answer(
        runtime_dependencies,
        question=question,
        answer=answer,
        decision=decision,
        group_id=group_id,
        mentioned=mentioned,
        admin=admin,
        deadline=deadline,
        baseline_revision=baseline_revision,
        target_item=target_item,
    )


def refresh_answer_for_late_context(
    *,
    question: str,
    answer: str,
    decision: ProcessingDecision,
    group_id: int,
    mentioned: bool,
    admin: bool,
    deadline: float,
    reviewed_revision: int,
    target_item: dict | None = None,
) -> tuple[str, str, int]:
    return delivery_review.refresh_answer_for_late_context(
        runtime_dependencies,
        question=question,
        answer=answer,
        decision=decision,
        group_id=group_id,
        mentioned=mentioned,
        admin=admin,
        deadline=deadline,
        reviewed_revision=reviewed_revision,
        target_item=target_item,
    )


def clear_chat_state() -> None:
    return conversation_service.clear_chat_state(runtime_dependencies)


def group_send_lock(group_id: int) -> threading.Lock:
    return conversation_service.group_send_lock(runtime_dependencies, group_id)




def allow_hostile_reply(group_id: int, user_id: str, *, now: float | None = None) -> bool:
    return conversation_service.allow_hostile_reply(runtime_dependencies, group_id, user_id, now=now)


def save_chat_history(path: str | Path | None = None) -> None:
    return chat_history_service.save_chat_history(runtime_dependencies, path)


def schedule_chat_history_save() -> None:
    return chat_history_service.schedule_chat_history_save(runtime_dependencies)


def chat_history_save_worker() -> None:
    return chat_history_service.chat_history_save_worker(runtime_dependencies)


def initialize_chat_memory() -> bool:
    return chat_history_service.initialize_chat_memory(runtime_dependencies)


def recall_group_chat_message(group_id: int, message_id: str) -> None:
    return chat_history_service.recall_group_chat_message(
        runtime_dependencies, group_id, message_id
    )


def load_chat_history(path: str | Path | None = None) -> int:
    return chat_history_service.load_chat_history(runtime_dependencies, path)


def migrate_loaded_chat_history_to_memory() -> int:
    return chat_history_service.migrate_loaded_chat_history_to_memory(
        runtime_dependencies
    )


def chat_reply_quota_reason(
    group_id: int,
    *,
    now: float | None = None,
    cooldown_seconds: int | None = None,
    max_per_hour: int | None = None,
    db_path: str | Path | None = None,
) -> str:
    return conversation_service.chat_reply_quota_reason(
        runtime_dependencies,
        group_id,
        now=now,
        cooldown_seconds=cooldown_seconds,
        max_per_hour=max_per_hour,
        db_path=db_path,
    )


def mark_chat_replied(
    group_id: int,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    return conversation_service.mark_chat_replied(
        runtime_dependencies, group_id, now=now, db_path=db_path,
    )


def check_and_mark_chat_reply_quota(
    group_id: int,
    *,
    now: float | None = None,
    cooldown_seconds: int | None = None,
    max_per_hour: int | None = None,
    db_path: str | Path | None = None,
) -> str:
    return conversation_service.check_and_mark_chat_reply_quota(
        runtime_dependencies, group_id,
        now=now, cooldown_seconds=cooldown_seconds,
        max_per_hour=max_per_hour, db_path=db_path,
    )


def celebration_was_replied(
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    now: float | None = None,
    window_seconds: int = 86400,
    db_path: str | Path | None = None,
) -> bool:
    return conversation_service.celebration_was_replied(
        runtime_dependencies, group_id, target_key, event_kind,
        now=now, window_seconds=window_seconds, db_path=db_path,
    )


def mark_celebration_replied(
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    return conversation_service.mark_celebration_replied(
        runtime_dependencies, group_id, target_key, event_kind,
        now=now, db_path=db_path,
    )


def has_auto_reply_keyword(message: str) -> bool:
    return conversation_service.has_auto_reply_keyword(runtime_dependencies, message)


def looks_like_direct_question(message: str) -> bool:
    return conversation_service.looks_like_direct_question(runtime_dependencies, message)


def looks_like_birthday_celebration(message: str) -> bool:
    return conversation_service.looks_like_birthday_celebration(runtime_dependencies, message)


def celebration_kind(message: str) -> str:
    return conversation_service.celebration_kind(runtime_dependencies, message)


def is_self_celebration(message: str, kind: str) -> bool:
    return conversation_service.is_self_celebration(runtime_dependencies, message, kind)


def response_mention_user_id(
    *,
    mentioned: bool,
    user_id,
    reply_mode: str,
    question: str,
    mentioned_user_ids: Sequence[str] = (),
) -> str:
    return conversation_service.response_mention_user_id(
        runtime_dependencies,
        mentioned=mentioned,
        user_id=user_id,
        reply_mode=reply_mode,
        question=question,
        mentioned_user_ids=mentioned_user_ids,
    )


def looks_like_assignment_to_humans(message: str) -> bool:
    return conversation_service.looks_like_assignment_to_humans(runtime_dependencies, message)


def is_identity_question(question: str) -> bool:
    return conversation_service.is_identity_question(runtime_dependencies, question)


def record_knowledge_gap(query: str, result) -> bool:
    return audit_observability.record_knowledge_gap(runtime_dependencies, query, result)



def recent_knowledge_gap_entries(limit: int = 5) -> list[dict]:
    return audit_observability.recent_knowledge_gap_entries(runtime_dependencies, limit)



def retrieve_knowledge(query: str, max_chars: int):
    with kb_lock:
        return kb.build_context_with_metrics(query, max_chars)


def attach_knowledge_result(
    decision: ProcessingDecision,
    query: str,
    result: ContextResult,
) -> ProcessingDecision:
    return attach_result(decision, query, result)


def is_strong_knowledge_match(top_score: float, query_coverage: float) -> bool:
    return semantic_routing.is_strong_knowledge_match(
        top_score,
        query_coverage,
        min_score=settings.knowledge_strong_min_score,
        min_coverage=settings.knowledge_strong_min_coverage,
    )


def contextual_retrieval_question(
    question: str,
    chat_context: Sequence[str],
) -> tuple[str, float] | None:
    if not getattr(settings, "contextual_query_enabled", False) or not chat_context:
        return None
    return rewrite_contextual_question(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=getattr(settings, "contextual_query_model", settings.llm_model),
        question=question,
        context=tuple(chat_context[-8:]),
        timeout=getattr(settings, "contextual_query_timeout_seconds", 8),
    )


def semantic_plan_for_message(
    question: str,
    chat_context: Sequence[str],
    *,
    scene_context: str = "",
    memory_candidates: Sequence[str] = (),
    mentioned: bool,
    mentions_other: bool,
    reply_target_user_id: str = "",
    newer_message_ids: Sequence[str] = (),
    timeout: int | None = None,
) -> MessagePlan | None:
    return context_pipeline_service.semantic_plan_for_message(
        runtime_dependencies, question, chat_context,
        scene_context=scene_context, memory_candidates=memory_candidates,
        mentioned=mentioned, mentions_other=mentions_other,
        reply_target_user_id=reply_target_user_id,
        newer_message_ids=newer_message_ids, timeout=timeout,
    )


def semantic_planner_timeout_cap(
    *,
    mentioned: bool,
    reply_target_user_id: str = "",
    explicit_knowledge_command: bool = False,
) -> int:
    return context_pipeline_service.semantic_planner_timeout_cap(
        runtime_dependencies,
        mentioned=mentioned, reply_target_user_id=reply_target_user_id,
        explicit_knowledge_command=explicit_knowledge_command,
    )


def semantic_planner_circuit_is_open(
    *, lane: str = "unsolicited", now: float | None = None,
) -> bool:
    return context_pipeline_service.semantic_planner_circuit_is_open(
        runtime_dependencies, lane=lane, now=now,
    )


def reserve_semantic_planner_request(
    *, lane: str = "unsolicited", now: float | None = None,
) -> bool:
    return context_pipeline_service.reserve_semantic_planner_request(
        runtime_dependencies, lane=lane, now=now,
    )


def semantic_planner_health_snapshot(*, now: float | None = None) -> dict[str, dict]:
    return context_pipeline_service.semantic_planner_health_snapshot(
        runtime_dependencies, now=now,
    )


def record_semantic_planner_availability(
    available: bool, *, lane: str = "unsolicited", now: float | None = None,
) -> None:
    return context_pipeline_service.record_semantic_planner_availability(
        runtime_dependencies, available, lane=lane, now=now,
    )


def semantic_plan_is_usable(plan: MessagePlan | None) -> bool:
    return context_pipeline_service.semantic_plan_is_usable(runtime_dependencies, plan)


def compact_scene_context(scene_context: str) -> str:
    return context_pipeline_service.compact_scene_context(runtime_dependencies, scene_context)


def context_selected_by_plan(
    chat_context: Sequence[str], plan: MessagePlan | None,
) -> tuple[str, ...]:
    return context_pipeline_service.context_selected_by_plan(
        runtime_dependencies, chat_context, plan,
    )


def context_line_payload(line: str) -> dict:
    return context_pipeline_service.context_line_payload(runtime_dependencies, line)


def context_line_message_id(line: str) -> str:
    return context_pipeline_service.context_line_message_id(runtime_dependencies, line)


def context_revision(context: Sequence[str]) -> int:
    return context_pipeline_service.context_revision(runtime_dependencies, context)


def context_message_ids_after_revision(
    context: Sequence[str], revision: int,
) -> tuple[str, ...]:
    return context_pipeline_service.context_message_ids_after_revision(
        runtime_dependencies, context, revision,
    )


def budget_recent_context(
    context: Sequence[str],
    *,
    max_messages: int,
    max_chars: int,
    required_message_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    return context_pipeline_service.budget_recent_context(
        runtime_dependencies, context,
        max_messages=max_messages, max_chars=max_chars,
        required_message_ids=required_message_ids,
    )


def semantic_context_for_decision(decision: ProcessingDecision) -> str:
    return context_pipeline_service.semantic_context_for_decision(
        runtime_dependencies, decision,
    )


def semantic_relation_audit_fields(decision: ProcessingDecision) -> dict:
    return context_pipeline_service.semantic_relation_audit_fields(
        runtime_dependencies, decision,
    )


def derive_bot_reply_perspective(
    plan: MessagePlan, selected_context: Sequence[str],
) -> tuple[str, str]:
    return context_pipeline_service.derive_bot_reply_perspective(
        runtime_dependencies, plan, selected_context,
    )


def derive_participation_role(
    plan: MessagePlan, selected_context: Sequence[str],
    *, explicitly_addressed: bool,
) -> str:
    return context_pipeline_service.derive_participation_role(
        runtime_dependencies, plan, selected_context,
        explicitly_addressed=explicitly_addressed,
    )


def apply_semantic_plan_metadata(
    decision: ProcessingDecision,
    plan: MessagePlan,
    *,
    explicitly_addressed: bool = False,
    context_revision: int = 0,
    scene_context: str = "",
    target_message_ids: Sequence[str] = (),
) -> ProcessingDecision:
    return context_pipeline_service.apply_semantic_plan_metadata(
        runtime_dependencies, decision, plan,
        explicitly_addressed=explicitly_addressed,
        context_revision=context_revision,
        scene_context=scene_context,
        target_message_ids=target_message_ids,
    )


def refresh_semantic_decision_for_late_context(
    decision: ProcessingDecision,
    item: dict,
    latest_context: Sequence[str],
    *,
    scene_context: str,
    deadline: float,
) -> tuple[ProcessingDecision | None, str]:
    return context_pipeline_service.refresh_semantic_decision_for_late_context(
        runtime_dependencies, decision, item, latest_context,
        scene_context=scene_context, deadline=deadline,
    )


def chat_memory_enabled_for_group(group_id: int) -> bool:
    return context_pipeline_service.chat_memory_enabled_for_group(
        runtime_dependencies, group_id,
    )


def probe_chat_memory(item: dict, query: str) -> MemoryProbeResult:
    return context_pipeline_service.probe_chat_memory(runtime_dependencies, item, query)


def deduplicate_memory_context(
    memory_context: Sequence[str], recent_context: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    return context_pipeline_service.deduplicate_memory_context(
        runtime_dependencies, memory_context, recent_context,
    )


def budget_memory_context(
    context: Sequence[str], *, max_hits: int, max_chars: int,
) -> tuple[str, ...]:
    return context_pipeline_service.budget_memory_context(
        runtime_dependencies, context, max_hits=max_hits, max_chars=max_chars,
    )


def apply_context_budget(decision: ProcessingDecision) -> None:
    return context_pipeline_service.apply_context_budget(runtime_dependencies, decision)


def enrich_decision_with_chat_memory(
    decision: ProcessingDecision,
    item: dict,
    plan: MessagePlan | None,
    probe: MemoryProbeResult | None = None,
) -> ProcessingDecision:
    return context_pipeline_service.enrich_decision_with_chat_memory(
        runtime_dependencies, decision, item, plan, probe,
    )



def answer_bot_meta(capability: str, *, admin: bool) -> str:
    return answering_service.answer_bot_meta(runtime_dependencies, capability, admin=admin)


def finalize_model_answer(answer: str, *, unsolicited: bool = False) -> str:
    return answering_service.finalize_model_answer(
        runtime_dependencies, answer, unsolicited=unsolicited,
    )


def answer_question(
    question: str,
    effective_question: str | None = None,
    *,
    retrieval_question: str | None = None,
    allow_fallback: bool = True,
    chat_context: Sequence[str] = (),
    memory_context: Sequence[str] = (),
    self_history_context: Sequence[str] = (),
    semantic_context: str = "",
    knowledge_result: ContextResult | None = None,
    timeout: int | None = None,
) -> str:
    return answering_service.answer_question(
        runtime_dependencies, question, effective_question,
        retrieval_question=retrieval_question, allow_fallback=allow_fallback,
        chat_context=chat_context, memory_context=memory_context,
        self_history_context=self_history_context, semantic_context=semantic_context,
        knowledge_result=knowledge_result, timeout=timeout,
    )


def fallback_grounding_issue(decision: ProcessingDecision, answer: str) -> str:
    return answering_service.fallback_grounding_issue(
        runtime_dependencies, decision, answer,
    )


def deterministic_review_failure_answer(
    decision: ProcessingDecision,
    candidate: str,
    *,
    mentioned: bool,
    context_changed: bool = False,
) -> str:
    return answering_service.deterministic_review_failure_answer(
        runtime_dependencies, decision, candidate,
        mentioned=mentioned, context_changed=context_changed,
    )


def answer_for_decision(
    question: str,
    decision: ProcessingDecision,
    generation_question: str,
    *,
    admin: bool = False,
    timeout: int | None = None,
) -> str:
    return answering_service.answer_for_decision(
        runtime_dependencies, question, decision, generation_question,
        admin=admin, timeout=timeout,
    )


def is_model_error_answer(answer: str) -> bool:
    return answering_service.is_model_error_answer(answer)


def next_sequence() -> int:
    return queue_dispatcher.next_sequence(runtime_dependencies)



def _pending_db_path(db_path: str | Path | None = None) -> str | Path:
    return queue_store._pending_db_path(runtime_dependencies, db_path)



def open_pending_queue_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    return queue_store.open_pending_queue_db(runtime_dependencies, db_path)



def persist_conversation_turn(
    group_id: int, state: ConversationState, *, db_path: str | Path | None = None
) -> int:
    return queue_store.persist_conversation_turn(
        runtime_dependencies, group_id, state, db_path=db_path
    )



def _conversation_state_from_row(row) -> ConversationState:
    return queue_store._conversation_state_from_row(runtime_dependencies, row)



def load_conversation_turn_by_bot_message_id(
    group_id: int, bot_message_id: str, *, db_path: str | Path | None = None
) -> ConversationState | None:
    return queue_store.load_conversation_turn_by_bot_message_id(
        runtime_dependencies, group_id, bot_message_id, db_path=db_path
    )



def persist_pending_message(
    priority: int, sequence: int, item: dict, db_path: str | Path | None = None
) -> int:
    return queue_store.persist_pending_message(
        runtime_dependencies, priority, sequence, item, db_path
    )



def load_pending_messages(
    db_path: str | Path | None = None, *, include_future: bool = False
) -> list[tuple[int, int, dict]]:
    return queue_store.load_pending_messages(
        runtime_dependencies, db_path, include_future=include_future
    )



def mark_pending_failure(
    pending_id: int,
    error: str,
    *,
    db_path: str | Path | None = None,
    now: float | None = None,
    max_attempts: int | None = None
) -> PendingFailureResult:
    return queue_store.mark_pending_failure(
        runtime_dependencies,
        pending_id,
        error,
        db_path=db_path,
        now=now,
        max_attempts=max_attempts,
    )



def mark_pending_sent_unknown(
    pending_id: int, error: str, *, db_path: str | Path | None = None
) -> None:
    return queue_store.mark_pending_sent_unknown(
        runtime_dependencies, pending_id, error, db_path=db_path
    )



def mark_pending_dispatch_started(
    pending_id: int, dispatch_id: str, *, db_path: str | Path | None = None
) -> None:
    return queue_store.mark_pending_dispatch_started(
        runtime_dependencies, pending_id, dispatch_id, db_path=db_path
    )



def recover_incomplete_pending_dispatches(db_path: str | Path | None = None) -> int:
    return queue_store.recover_incomplete_pending_dispatches(
        runtime_dependencies, db_path
    )



def delete_pending_message(pending_id: int, db_path: str | Path | None = None) -> None:
    return queue_store.delete_pending_message(runtime_dependencies, pending_id, db_path)



def pending_message_count(db_path: str | Path | None = None) -> int:
    return queue_store.pending_message_count(runtime_dependencies, db_path)



def pending_status_counts(db_path: str | Path | None = None) -> dict[str, int]:
    return queue_store.pending_status_counts(runtime_dependencies, db_path)



def enqueue_persistent_message(priority: int, item: dict) -> int:
    return queue_dispatcher.enqueue_persistent_message(
        runtime_dependencies, priority, item
    )



def _queue_pending_item(item: dict, *, delay: float = 0.0) -> None:
    return queue_dispatcher._queue_pending_item(runtime_dependencies, item, delay=delay)



def handle_pending_worker_failure(item: dict, error: str) -> str:
    return queue_dispatcher.handle_pending_worker_failure(
        runtime_dependencies, item, error
    )



def begin_pending_dispatch(item: dict) -> None:
    return queue_dispatcher.begin_pending_dispatch(runtime_dependencies, item)



def classify_fragment_audience(item: dict) -> str:
    return fragment_ingress.classify_fragment_audience(runtime_dependencies, item)



def fragment_items_compatible(
    buffer: MessageFragmentBuffer, item: dict, audience: str
) -> bool:
    return fragment_ingress.fragment_items_compatible(
        runtime_dependencies, buffer, item, audience
    )



def _message_ids(item: dict) -> list[str]:
    return reply_delivery.message_ids(runtime_dependencies, item)


def bot_turn_metadata(item: dict, bot_message_id) -> tuple[tuple[str, ...], str]:
    return reply_delivery.bot_turn_metadata(
        runtime_dependencies, item, bot_message_id
    )


def send_and_record_bot_turn(
    *,
    group_id: int,
    item: dict,
    answer: str,
    reply_mode: str,
    semantic_topic: str = "",
    mention_user_id: str = "",
    reply_to_trigger: bool = False,
) -> tuple[object, tuple[str, ...], str]:
    return reply_delivery.send_and_record_bot_turn(
        runtime_dependencies,
        group_id=group_id,
        item=item,
        answer=answer,
        reply_mode=reply_mode,
        semantic_topic=semantic_topic,
        mention_user_id=mention_user_id,
        reply_to_trigger=reply_to_trigger,
    )


def message_already_covered_by_bot(group_id: int, message_id) -> bool:
    return reply_delivery.message_already_covered_by_bot(
        runtime_dependencies, group_id, message_id
    )


def _context_message_speakers(context: Sequence[str]) -> dict[str, str]:
    return reply_delivery.context_message_speakers(context)


def merge_review_message_ids(item: dict | None, review, latest_context: Sequence[str]) -> None:
    return reply_delivery.merge_review_message_ids(
        runtime_dependencies, item, review, latest_context
    )


def _new_fragment_buffer(
    item: dict, audience: str, now: float
) -> MessageFragmentBuffer:
    return fragment_ingress._new_fragment_buffer(
        runtime_dependencies, item, audience, now
    )



def _merge_fragment(
    buffer: MessageFragmentBuffer, item: dict, audience: str, now: float
) -> None:
    return fragment_ingress._merge_fragment(
        runtime_dependencies, buffer, item, audience, now
    )



def _fragment_prefix_item(buffer: MessageFragmentBuffer, count: int) -> dict:
    return fragment_ingress._fragment_prefix_item(runtime_dependencies, buffer, count)



def semantic_bot_fragment_count(buffer: MessageFragmentBuffer) -> int:
    return fragment_ingress.semantic_bot_fragment_count(runtime_dependencies, buffer)



def _dispatch_fragment_buffer(buffer: MessageFragmentBuffer) -> int:
    return fragment_ingress._dispatch_fragment_buffer(runtime_dependencies, buffer)



def _defer_fragment_buffers(buffers: Sequence[MessageFragmentBuffer]) -> None:
    return fragment_ingress._defer_fragment_buffers(runtime_dependencies, buffers)



def clear_fragment_state() -> None:
    return fragment_ingress.clear_fragment_state(runtime_dependencies)



def flush_group_fragment_buffer(
    group_id: int, *, defer_dispatch: bool = False
) -> int | None:
    return fragment_ingress.flush_group_fragment_buffer(
        runtime_dependencies, group_id, defer_dispatch=defer_dispatch
    )



def flush_fragment_buffer_for_new_speaker(
    group_id: int, user_id, *, defer_dispatch: bool = False
) -> int | None:
    return fragment_ingress.flush_fragment_buffer_for_new_speaker(
        runtime_dependencies, group_id, user_id, defer_dispatch=defer_dispatch
    )



def submit_message_fragment(
    item: dict, *, now: float | None = None, defer_dispatch: bool = False
) -> list[int]:
    return fragment_ingress.submit_message_fragment(
        runtime_dependencies, item, now=now, defer_dispatch=defer_dispatch
    )



def fragment_aggregation_worker() -> None:
    return fragment_ingress.fragment_aggregation_worker(runtime_dependencies)



def restore_pending_messages() -> int:
    return queue_dispatcher.restore_pending_messages(runtime_dependencies)



def message_max_age_seconds(mentioned: bool) -> int:
    return delivery_policies.message_max_age_seconds(runtime_dependencies, mentioned)


def is_message_too_old(
    event_time,
    mentioned: bool,
    *,
    fallback_time=None,
    now: float | None = None,
) -> bool:
    return delivery_policies.is_message_too_old(
        runtime_dependencies,
        event_time,
        mentioned,
        fallback_time=fallback_time,
        now=now,
    )


def is_event_too_old(event: dict) -> bool:
    return delivery_policies.is_event_too_old(runtime_dependencies, event)


def acquire_reply_slot(
    *,
    block: bool = True,
    reserve_slots: int = 0,
    deadline: float | None = None,
) -> bool:
    return delivery_policies.acquire_reply_slot(
        runtime_dependencies,
        block=block,
        reserve_slots=reserve_slots,
        deadline=deadline,
    )


def wait_for_rate_limit(deadline: float | None = None) -> bool:
    return delivery_policies.wait_for_rate_limit(runtime_dependencies, deadline)


def consider_chat_reply(
    normalized: str,
    *,
    group_id: int,
    chat_context: Sequence[str],
    mentions_other: bool,
    has_context: bool = False,
    sources: Sequence[str] = (),
    query_text: str = "",
    followup_of: str = "",
    followup_scope: str = "",
) -> ProcessingDecision:
    return conversation_service.consider_chat_reply(
        runtime_dependencies,
        normalized,
        group_id=group_id,
        chat_context=chat_context,
        mentions_other=mentions_other,
        has_context=has_context,
        sources=sources,
        query_text=query_text,
        followup_of=followup_of,
        followup_scope=followup_scope,
    )


def should_process_message(
    question: str,
    mentioned: bool,
    effective_question: str | None = None,
    followup_of: str = "",
    followup_scope: str = "",
    group_id: int = 0,
    chat_context: Sequence[str] = (),
    scene_context: str = "",
    memory_candidates: Sequence[str] = (),
    mentions_other: bool = False,
    reply_target_user_id: str = "",
    user_id: str = "",
    sender_role: str = "",
    planner_timeout: int | None = None,
    plan_out: list[MessagePlan] | None = None,
    explicit_knowledge_command: bool = False,
) -> ProcessingDecision:
    return message_router.should_process_message(
        runtime_dependencies,
        question,
        mentioned,
        effective_question,
        followup_of,
        followup_scope,
        group_id,
        chat_context,
        scene_context,
        memory_candidates,
        mentions_other,
        reply_target_user_id,
        user_id,
        sender_role,
        planner_timeout,
        plan_out,
        explicit_knowledge_command,
    )


def process_worker_item(
    item: dict,
    lane: str,
    lifecycle: PendingItemLifecycle | None = None,
) -> PendingItemLifecycle:
    return _process_worker_item_impl(
        runtime_dependencies,
        item,
        lane,
        lifecycle,
    )



def worker(work_queue: queue.PriorityQueue, lane: str) -> None:
    return run_worker(runtime_dependencies, work_queue, lane)


def process_chat_item(
    item: dict,
    decision: ProcessingDecision,
    lifecycle: PendingItemLifecycle | None = None,
) -> PendingItemLifecycle:
    return _process_chat_item_impl(
        runtime_dependencies,
        item,
        decision,
        lifecycle,
    )



def chat_worker() -> None:
    return run_chat_worker(runtime_dependencies)


def handle_onebot_event(event: dict) -> tuple[int, dict]:
    return ingress_events.handle_onebot_event(runtime_dependencies, event)


Handler = http_transport.create_handler(runtime_dependencies)



def main() -> None:
    app_service.main(runtime_dependencies)


if __name__ == "__main__":
    main()
