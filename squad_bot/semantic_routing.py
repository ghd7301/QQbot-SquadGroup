from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from .knowledge import ContextResult
from .llm import MessagePlan
from .models import ProcessingDecision


def compact_scene_context(scene_context: str) -> str:
    if not scene_context:
        return ""
    try:
        payload = json.loads(scene_context)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    topics = []
    for topic in payload.get("topics") or ():
        if not isinstance(topic, dict) or len(topics) >= 2:
            continue
        topics.append({
            "id": str(topic.get("id") or "")[:30],
            "summary": str(topic.get("summary") or "")[:120],
            "participants": list(topic.get("participants") or ())[:6],
            "anchor_message_ids": list(topic.get("anchor_message_ids") or ())[:4],
            "confidence": topic.get("confidence", 0),
        })
    compact = {
        "version": int(payload.get("version") or 0),
        "updated_through_sequence": int(
            payload.get("updated_through_sequence") or 0
        ),
        "active_topic_id": str(payload.get("active_topic_id") or "")[:30],
        "topics": topics,
    }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def plan_message(
    question: str,
    chat_context: Sequence[str],
    *,
    planner: Callable[..., MessagePlan | None],
    memory_budget: Callable[..., tuple[str, ...]],
    base_url: str,
    api_key: str,
    model: str,
    scene_context: str,
    memory_candidates: Sequence[str],
    mentioned: bool,
    mentions_other: bool,
    reply_target_user_id: str,
    bot_user_id: str,
    newer_message_ids: Sequence[str],
    timeout: int,
    context_messages: int,
    context_max_chars: int,
    memory_max_chars: int,
) -> MessagePlan | None:
    if reply_target_user_id == bot_user_id:
        reply_target = "bot"
    elif reply_target_user_id:
        reply_target = "member"
    else:
        reply_target = "none"
    planner_context = budget_recent_context(
        chat_context,
        max_messages=max(1, context_messages),
        max_chars=max(800, context_max_chars),
    )
    planner_memory = memory_budget(
        memory_candidates,
        max_hits=3,
        max_chars=max(200, memory_max_chars),
    )
    return planner(
        base_url=base_url,
        api_key=api_key,
        model=model,
        message=question,
        context=planner_context,
        scene_context=compact_scene_context(scene_context),
        memory_candidates=planner_memory,
        newer_message_ids=tuple(newer_message_ids),
        mentioned=mentioned,
        mentions_other=mentions_other,
        reply_target=reply_target,
        timeout=timeout,
    )


def is_strong_knowledge_match(
    top_score: float,
    query_coverage: float,
    *,
    min_score: float,
    min_coverage: float,
) -> bool:
    return top_score >= min_score and query_coverage >= min_coverage


def semantic_plan_is_usable(
    plan: MessagePlan | None,
    *,
    min_confidence: float,
) -> bool:
    return bool(plan and plan.confidence >= min_confidence)


def planner_timeout_cap(
    *,
    explicitly_addressed: bool,
    default_timeout: int,
    addressed_timeout: int,
) -> int:
    return addressed_timeout if explicitly_addressed else default_timeout


def addressed_planner_fallback(
    *,
    result: ContextResult,
    query_text: str,
    followup_of: str,
    followup_scope: str,
    chat_context: Sequence[str],
    explicit_knowledge_command: bool,
    strong_match: bool,
    fallback_allowed: bool,
    low_confidence: bool,
) -> ProcessingDecision | None:
    planner_status = "low_confidence" if low_confidence else "unavailable"
    if explicit_knowledge_command and strong_match:
        decision = ProcessingDecision(
            True,
            "explicit knowledge command with strong context",
            True,
            tuple(result.sources),
            query_text,
            followup_of,
            followup_scope,
            "knowledge",
            tuple(chat_context),
            result.top_score,
            result.query_coverage,
            semantic_intent="knowledge",
            semantic_audience="bot",
            participation_role="addressed",
            planner_status=planner_status,
        )
        decision.knowledge_query = query_text
        decision.knowledge_result = result
        return decision
    if not fallback_allowed:
        return None
    decision = ProcessingDecision(
        should_reply=True,
        reason="explicit address with unverified semantic fallback",
        has_context=bool(result.context),
        sources=tuple(result.sources),
        effective_question=query_text,
        followup_of=followup_of,
        followup_scope=followup_scope,
        reply_mode="fallback",
        chat_context=(),
        retrieval_score=result.top_score,
        retrieval_coverage=result.query_coverage,
        semantic_intent="unclear",
        semantic_audience="bot",
        participation_role="addressed",
        risk_flags=("intent_unverified",),
        planner_status=planner_status,
    )
    if strong_match:
        decision.knowledge_query = query_text
        decision.knowledge_result = result
    return decision


def unavailable_unsolicited_decision(
    query_text: str,
    *,
    circuit_open: bool,
    low_confidence: bool,
) -> ProcessingDecision:
    return ProcessingDecision(
        False,
        (
            "semantic planner circuit open; unsolicited reply fails closed"
            if circuit_open
            else "semantic planner unavailable; unsolicited reply fails closed"
        ),
        effective_question=query_text,
        planner_status=(
            "circuit_open"
            if circuit_open
            else ("low_confidence" if low_confidence else "unavailable")
        ),
    )


def context_line_payload(line: str) -> dict:
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def context_line_message_id(line: str) -> str:
    return str(context_line_payload(line).get("message_id") or "")


def context_revision(context: Sequence[str]) -> int:
    revision = 0
    for line in context:
        payload = context_line_payload(line)
        speaker = payload.get("speaker") or {}
        if speaker.get("role") == "bot":
            continue
        try:
            revision = max(revision, int(payload.get("sequence") or 0))
        except (TypeError, ValueError):
            continue
    return revision


def context_message_ids_after_revision(
    context: Sequence[str],
    revision: int,
) -> tuple[str, ...]:
    result = []
    for line in context:
        payload = context_line_payload(line)
        speaker = payload.get("speaker") or {}
        if speaker.get("role") == "bot":
            continue
        try:
            sequence = int(payload.get("sequence") or 0)
        except (TypeError, ValueError):
            continue
        message_id = str(payload.get("message_id") or "").strip()
        if sequence > revision and message_id and message_id not in result:
            result.append(message_id)
    return tuple(result)


def context_selected_by_plan(
    chat_context: Sequence[str],
    plan: MessagePlan | None,
) -> tuple[str, ...]:
    if not plan:
        return tuple(chat_context)
    current_lines = tuple(
        line
        for line in chat_context
        if context_line_payload(line).get("current") or "【当前消息】" in line
    )
    required_ids = {
        context_line_message_id(line)
        for line in current_lines
        if context_line_message_id(line)
    }
    reply_ids = {
        reply_id
        for line in current_lines
        if (
            reply_id := str(
                (context_line_payload(line).get("reply_to") or {}).get("message_id")
                or ""
            )
        )
    }
    required_ids.update(reply_ids)
    for candidate in plan.topic_candidates:
        required_ids.update(candidate.anchor_message_ids)
    for candidate in plan.subject_candidates:
        required_ids.update(candidate.evidence_message_ids)
    bot_is_possible_subject = any(
        candidate.entity_type == "bot" and candidate.confidence >= 0.5
        for candidate in plan.subject_candidates
    )
    if bot_is_possible_subject:
        required_ids.update(
            message_id
            for line in chat_context
            if (context_line_payload(line).get("speaker") or {}).get("role") == "bot"
            if (message_id := context_line_message_id(line))
        )
    changed = True
    while changed:
        changed = False
        for line in chat_context:
            message_id = context_line_message_id(line)
            if message_id not in required_ids:
                continue
            reply_id = str(
                (context_line_payload(line).get("reply_to") or {}).get("message_id")
                or ""
            )
            if reply_id and reply_id not in required_ids:
                required_ids.add(reply_id)
                changed = True
    hard_context = tuple(
        line
        for line in chat_context
        if line in current_lines or context_line_message_id(line) in required_ids
    )
    selected: tuple[str, ...] = ()
    if plan.relevant_context_message_ids:
        wanted = set(plan.relevant_context_message_ids) | required_ids
        selected = tuple(
            line for line in chat_context if context_line_message_id(line) in wanted
        )
    elif plan.relevant_context_indices:
        selected = tuple(
            chat_context[index - 1]
            for index in plan.relevant_context_indices
            if 1 <= index <= len(chat_context)
        )
    if not selected:
        return hard_context or tuple(chat_context[-1:])
    return tuple(dict.fromkeys((*selected, *hard_context)))


def budget_recent_context(
    context: Sequence[str],
    *,
    max_messages: int,
    max_chars: int,
    required_message_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    if not context:
        return ()
    required_ids: set[str] = {
        str(value or "").strip()
        for value in required_message_ids
        if str(value or "").strip()
    }
    for line in context:
        payload = context_line_payload(line)
        if payload.get("current"):
            message_id = str(payload.get("message_id") or "")
            if message_id:
                required_ids.add(message_id)
            reply_to = payload.get("reply_to") or {}
            replied_id = str(reply_to.get("message_id") or "")
            if replied_id:
                required_ids.add(replied_id)
    selected: list[str] = []
    used = 0
    for line in reversed(tuple(context)):
        message_id = context_line_message_id(line)
        required = message_id in required_ids
        if not required and len(selected) >= max(1, max_messages):
            continue
        if not required and selected and used + len(line) > max(200, max_chars):
            continue
        selected.append(line)
        used += len(line)
    selected.reverse()
    return tuple(selected)


def semantic_context_for_decision(decision: ProcessingDecision) -> str:
    parts = []
    if decision.planner_status in {"unavailable", "circuit_open", "low_confidence"}:
        parts.append("语义规划未确认：只能保守理解当前消息，不得因知识检索命中而假定这是事实问题")
    if decision.semantic_topic:
        parts.append(f"相关话题：{decision.semantic_topic}")
    if decision.implicit_meaning:
        parts.append(f"可能的非字面含义：{decision.implicit_meaning}")
    if decision.topic_candidates:
        candidates = "；".join(
            f"{candidate.label}（{candidate.basis}，置信度 {candidate.confidence:.2f}）"
            for candidate in decision.topic_candidates
        )
        parts.append(f"候选话题：{candidates}")
    if decision.semantic_intent or decision.subject_candidates:
        parts.append(f"语义受众：{decision.semantic_audience}")
        parts.append(f"机器人参与资格：{decision.participation_role}")
        parts.append(f"讨论对象状态：{decision.subject_ambiguity}")
        parts.append(f"机器人参与关系：{decision.bot_involvement}")
        parts.append(f"要求回复视角：{decision.reply_perspective}")
    if decision.subject_candidates:
        subjects = "；".join(
            (
                f"{candidate.label}（类型 {candidate.entity_type}，"
                f"置信度 {candidate.confidence:.2f}）"
            )
            for candidate in decision.subject_candidates
        )
        parts.append(f"讨论对象候选：{subjects}")
    return "\n".join(parts)


def semantic_relation_audit_fields(decision: ProcessingDecision) -> dict:
    return {
        "topic_candidates": decision.topic_candidates,
        "subject_candidates": decision.subject_candidates,
        "subject_ambiguity": decision.subject_ambiguity,
        "bot_involvement": decision.bot_involvement,
        "reply_perspective": decision.reply_perspective,
        "semantic_audience": decision.semantic_audience,
        "participation_role": decision.participation_role,
        "plan_context_revision": decision.plan_context_revision,
        "plan_scene_version": decision.plan_scene_version,
        "related_message_ids": decision.related_message_ids,
        "semantic_replan_count": decision.semantic_replan_count,
        "semantic_replan_reason": decision.semantic_replan_reason,
        "planner_status": decision.planner_status,
        "planner_latency_ms": decision.planner_latency_ms,
    }


def derive_bot_reply_perspective(
    plan: MessagePlan,
    selected_context: Sequence[str],
) -> tuple[str, str]:
    candidates = sorted(
        plan.subject_candidates,
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )
    top = candidates[0] if candidates else None
    bot_candidate = next(
        (candidate for candidate in candidates if candidate.entity_type == "bot"),
        None,
    )
    if (
        bot_candidate
        and bot_candidate.confidence >= 0.72
        and plan.subject_ambiguity == "clear"
        and (top is bot_candidate or bot_candidate.confidence >= top.confidence)
        and (bool(bot_candidate.evidence_message_ids) or plan.audience == "bot")
    ):
        return "subject", "first_person"
    if bot_candidate and bot_candidate.confidence >= 0.5:
        return "uncertain", "neutral"

    bot_participated = any(
        (context_line_payload(line).get("speaker") or {}).get("role") == "bot"
        for line in selected_context
    )
    if plan.subject_ambiguity != "clear" or not top or top.confidence < 0.72:
        return "uncertain", "neutral"
    if bot_participated:
        return "participant", "observer"
    return "observer", "observer"


def derive_participation_role(
    plan: MessagePlan,
    selected_context: Sequence[str],
    *,
    explicitly_addressed: bool,
) -> str:
    if explicitly_addressed:
        return "addressed"
    bot_subject = next(
        (
            candidate
            for candidate in plan.subject_candidates
            if candidate.entity_type == "bot" and candidate.confidence >= 0.72
        ),
        None,
    )
    if bot_subject and plan.subject_ambiguity == "clear":
        return "subject"

    role = plan.participation_role
    if role == "participant":
        bot_participated = any(
            (context_line_payload(line).get("speaker") or {}).get("role") == "bot"
            for line in selected_context
        )
        return "participant" if bot_participated else "uncertain"
    if role == "group_open":
        return "group_open" if plan.audience == "group" else "uncertain"
    if role in {"bystander", "uncertain"}:
        return role
    return "uncertain"


def apply_semantic_plan_metadata(
    decision: ProcessingDecision,
    plan: MessagePlan,
    *,
    explicitly_addressed: bool = False,
    context_revision: int = 0,
    scene_context: str = "",
    target_message_ids: Sequence[str] = (),
) -> ProcessingDecision:
    decision.risk_flags = plan.risk_flags
    decision.topic_candidates = plan.topic_candidates
    decision.subject_candidates = plan.subject_candidates
    decision.subject_ambiguity = plan.subject_ambiguity
    decision.semantic_intent = plan.intent
    decision.semantic_topic = plan.topic_summary
    decision.implicit_meaning = plan.implicit_meaning
    decision.semantic_confidence = plan.confidence
    decision.effective_question = plan.standalone_question
    decision.capability = plan.capability if plan.intent == "bot_meta" else "none"
    decision.draft_reply = ""
    decision.semantic_audience = plan.audience
    decision.participation_role = derive_participation_role(
        plan,
        decision.chat_context,
        explicitly_addressed=explicitly_addressed,
    )
    decision.plan_context_revision = max(0, int(context_revision or 0))
    try:
        scene_payload = json.loads(scene_context) if scene_context else {}
    except (json.JSONDecodeError, TypeError):
        scene_payload = {}
    decision.plan_scene_version = (
        int(scene_payload.get("version") or 0)
        if isinstance(scene_payload, dict)
        else 0
    )
    decision.related_message_ids = tuple(dict.fromkeys(
        str(value or "").strip()
        for value in (*target_message_ids, *plan.relevant_context_message_ids)
        if str(value or "").strip()
    ))
    (
        decision.bot_involvement,
        decision.reply_perspective,
    ) = derive_bot_reply_perspective(plan, decision.chat_context)
    if decision.bot_involvement in {"subject", "uncertain"}:
        decision.risk_flags = tuple(dict.fromkeys(
            (*decision.risk_flags, "self_identity")
        ))
    if decision.reply_perspective == "neutral":
        decision.draft_reply = ""
    return decision
