from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from . import semantic_routing
from .chat_memory import MemoryHit
from .llm import MessagePlan
from .models import (
    MemoryProbeResult,
    ProcessingDecision,
)


# ---------------------------------------------------------------------------
# Semantic planner request and configuration
# ---------------------------------------------------------------------------


def semantic_plan_for_message(
    deps,
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
    if not getattr(deps.settings, "semantic_planner_enabled", False):
        return None
    return semantic_routing.plan_message(
        question,
        chat_context,
        planner=deps.plan_group_message,
        memory_budget=lambda ctx, **kw: budget_memory_context(deps, ctx, **kw),
        base_url=deps.settings.llm_base_url,
        api_key=deps.settings.llm_api_key,
        model=getattr(deps.settings, "semantic_planner_model", deps.settings.llm_model),
        scene_context=scene_context,
        memory_candidates=memory_candidates,
        mentioned=mentioned,
        mentions_other=mentions_other,
        reply_target_user_id=reply_target_user_id,
        bot_user_id=deps.settings.bot_qq,
        newer_message_ids=newer_message_ids,
        timeout=timeout or getattr(deps.settings, "semantic_planner_timeout_seconds", 4),
        context_messages=getattr(deps.settings, "semantic_planner_context_messages", 10),
        context_max_chars=getattr(
            deps.settings,
            "semantic_planner_context_max_chars",
            3200,
        ),
        memory_max_chars=getattr(deps.settings, "semantic_planner_memory_max_chars", 800),
    )


def semantic_planner_timeout_cap(
    deps,
    *,
    mentioned: bool,
    reply_target_user_id: str = "",
    explicit_knowledge_command: bool = False,
) -> int:
    explicitly_addressed = bool(
        mentioned
        or explicit_knowledge_command
        or reply_target_user_id == deps.settings.bot_qq
    )
    default_timeout = getattr(deps.settings, "semantic_planner_timeout_seconds", 3)
    return semantic_routing.planner_timeout_cap(
        explicitly_addressed=explicitly_addressed,
        default_timeout=default_timeout,
        addressed_timeout=getattr(
            deps.settings,
            "semantic_planner_addressed_timeout_seconds",
            default_timeout,
        ),
    )


# ---------------------------------------------------------------------------
# Planner health and circuit breaker
# ---------------------------------------------------------------------------


def semantic_planner_circuit_is_open(
    deps,
    *,
    lane: str = "unsolicited",
    now: float | None = None,
) -> bool:
    return deps.semantic_planner_health.circuit_is_open(lane, now=now)


def reserve_semantic_planner_request(
    deps,
    *,
    lane: str = "unsolicited",
    now: float | None = None,
) -> bool:
    return deps.semantic_planner_health.reserve_request(
        lane,
        failure_threshold=getattr(deps.settings, "semantic_planner_circuit_failures", 5),
        now=now,
    )


def semantic_planner_health_snapshot(deps, *, now: float | None = None) -> dict[str, dict]:
    return deps.semantic_planner_health.snapshot(now=now)


def record_semantic_planner_availability(
    deps,
    available: bool,
    *,
    lane: str = "unsolicited",
    now: float | None = None,
) -> None:
    deps.semantic_planner_health.record(
        available,
        lane,
        failure_threshold=getattr(deps.settings, "semantic_planner_circuit_failures", 5),
        circuit_seconds=getattr(deps.settings, "semantic_planner_circuit_seconds", 30),
        now=now,
    )


def semantic_plan_is_usable(deps, plan: MessagePlan | None) -> bool:
    return semantic_routing.semantic_plan_is_usable(
        plan,
        min_confidence=getattr(deps.settings, "semantic_planner_min_confidence", 0.68),
    )


# ---------------------------------------------------------------------------
# Context utilities (thin delegates to semantic_routing)
# ---------------------------------------------------------------------------


def compact_scene_context(deps, scene_context: str) -> str:
    return semantic_routing.compact_scene_context(scene_context)


def context_selected_by_plan(
    deps,
    chat_context: Sequence[str],
    plan: MessagePlan | None,
) -> tuple[str, ...]:
    return semantic_routing.context_selected_by_plan(chat_context, plan)


def context_line_payload(deps, line: str) -> dict:
    return semantic_routing.context_line_payload(line)


def context_line_message_id(deps, line: str) -> str:
    return semantic_routing.context_line_message_id(line)


def context_revision(deps, context: Sequence[str]) -> int:
    return semantic_routing.context_revision(context)


def context_message_ids_after_revision(
    deps,
    context: Sequence[str],
    revision: int,
) -> tuple[str, ...]:
    return semantic_routing.context_message_ids_after_revision(context, revision)


def budget_recent_context(
    deps,
    context: Sequence[str],
    *,
    max_messages: int,
    max_chars: int,
    required_message_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    return semantic_routing.budget_recent_context(
        context,
        max_messages=max_messages,
        max_chars=max_chars,
        required_message_ids=required_message_ids,
    )


def semantic_context_for_decision(deps, decision: ProcessingDecision) -> str:
    return semantic_routing.semantic_context_for_decision(decision)


def semantic_relation_audit_fields(deps, decision: ProcessingDecision) -> dict:
    return semantic_routing.semantic_relation_audit_fields(decision)


def derive_bot_reply_perspective(
    deps,
    plan: MessagePlan,
    selected_context: Sequence[str],
) -> tuple[str, str]:
    return semantic_routing.derive_bot_reply_perspective(plan, selected_context)


def derive_participation_role(
    deps,
    plan: MessagePlan,
    selected_context: Sequence[str],
    *,
    explicitly_addressed: bool,
) -> str:
    return semantic_routing.derive_participation_role(
        plan,
        selected_context,
        explicitly_addressed=explicitly_addressed,
    )


def apply_semantic_plan_metadata(
    deps,
    decision: ProcessingDecision,
    plan: MessagePlan,
    *,
    explicitly_addressed: bool = False,
    context_revision: int = 0,
    scene_context: str = "",
    target_message_ids: Sequence[str] = (),
) -> ProcessingDecision:
    return semantic_routing.apply_semantic_plan_metadata(
        decision,
        plan,
        explicitly_addressed=explicitly_addressed,
        context_revision=context_revision,
        scene_context=scene_context,
        target_message_ids=target_message_ids,
    )


# ---------------------------------------------------------------------------
# Late-context semantic replan
# ---------------------------------------------------------------------------


def refresh_semantic_decision_for_late_context(
    deps,
    decision: ProcessingDecision,
    item: dict,
    latest_context: Sequence[str],
    *,
    scene_context: str,
    deadline: float,
) -> tuple[ProcessingDecision | None, str]:
    latest_context = tuple(latest_context)
    latest_revision = deps.context_revision(latest_context)
    newer_ids = deps.context_message_ids_after_revision(
        latest_context,
        decision.plan_context_revision,
    )
    if not newer_ids:
        return decision, "semantic context unchanged"
    if (
        not getattr(deps.settings, "semantic_replan_enabled", True)
        or decision.semantic_replan_count >= 1
    ):
        return decision, "semantic replan already used or disabled"

    timeout = deps.remaining_reply_timeout(
        deadline,
        cap=deps.semantic_planner_timeout_cap(
            mentioned=bool(item.get("mentioned")),
            reply_target_user_id=str(item.get("reply_target_user_id") or ""),
            explicit_knowledge_command=bool(item.get("explicit_knowledge_command")),
        ),
        reserve=2,
    )
    if not timeout:
        return None, "reply deadline exhausted before semantic replan"
    plan = deps.semantic_plan_for_message(
        str(item.get("question") or ""),
        latest_context,
        scene_context=scene_context,
        mentioned=bool(item.get("mentioned")),
        mentions_other=bool(item.get("mentions_other")),
        reply_target_user_id=str(item.get("reply_target_user_id") or ""),
        newer_message_ids=newer_ids,
        timeout=timeout,
    )
    if not deps.semantic_plan_is_usable(plan):
        if item.get("mentioned"):
            decision.plan_context_revision = latest_revision
            return decision, "semantic replan unavailable; explicit address preserved"
        return None, "semantic replan unavailable; unsolicited reply fails closed"

    related_ids = set(plan.relevant_context_message_ids)
    related_ids.update(
        message_id
        for candidate in plan.topic_candidates
        for message_id in candidate.anchor_message_ids
    )
    related_ids.update(
        message_id
        for candidate in plan.subject_candidates
        for message_id in candidate.evidence_message_ids
    )
    related_new_ids = tuple(
        message_id for message_id in newer_ids if message_id in related_ids
    )
    if not related_new_ids:
        decision.plan_context_revision = latest_revision
        decision.semantic_replan_count += 1
        decision.semantic_replan_reason = "new messages were unrelated parallel context"
        return decision, decision.semantic_replan_reason

    decision.chat_context = deps.context_selected_by_plan(latest_context, plan)
    deps.apply_semantic_plan_metadata(
        decision,
        plan,
        explicitly_addressed=(
            bool(item.get("mentioned"))
            or str(item.get("reply_target_user_id") or "") == deps.settings.bot_qq
        ),
        context_revision=latest_revision,
        scene_context=scene_context,
        target_message_ids=(*deps._message_ids(item), *related_new_ids),
    )
    decision.semantic_replan_count += 1
    decision.semantic_replan_reason = (
        "related late context replaced semantic decision: "
        + ",".join(related_new_ids)
    )
    if (
        not item.get("mentioned")
        and decision.participation_role in {"bystander", "uncertain"}
    ):
        return None, (
            "semantic replan removed bot participation: "
            f"{decision.participation_role}"
        )
    if decision.semantic_audience == "member" and not item.get("mentioned"):
        return None, "semantic replan redirected message to another member"
    if decision.reply_mode == "chat" and (
        plan.intent not in deps.SEMANTIC_CHAT_INTENTS or not plan.reply_worthy
    ):
        return None, "semantic replan found no natural chat entry"
    return decision, decision.semantic_replan_reason


# ---------------------------------------------------------------------------
# Long-term memory probe, dedup and budget
# ---------------------------------------------------------------------------


def chat_memory_enabled_for_group(deps, group_id: int) -> bool:
    if not deps.chat_memory_manager or not getattr(deps.settings, "chat_memory_enabled", True):
        return False
    allowed = getattr(deps.settings, "chat_memory_allowed_group_ids", ())
    return not allowed or str(group_id) in allowed


def probe_chat_memory(deps, item: dict, query: str) -> MemoryProbeResult:
    group_id = int(item.get("group_id") or 0)
    if not chat_memory_enabled_for_group(deps, group_id):
        return MemoryProbeResult(query=query, rejection_reason="memory disabled for group")
    normalized = "".join(re.findall(r"[a-z0-9㐀-鿿]", str(query or "").lower()))
    if len(normalized) < 4:
        return MemoryProbeResult(query=query, rejection_reason="query too short for automatic probe")
    try:
        hits = deps.chat_memory_manager.store.lexical_probe(
            group_id=group_id,
            query=query,
            exclude_message_id=str(item.get("message_id") or ""),
            limit=max(1, getattr(deps.settings, "chat_memory_probe_max_hits", 8)),
            max_chars=max(200, getattr(deps.settings, "chat_memory_probe_max_chars", 1600)),
        )
        context = deps.chat_memory_manager.store.format_hits(hits)
        return MemoryProbeResult(
            query=query,
            hits=hits,
            context=context,
            attempted=True,
            rejection_reason="" if context else "no relevant memory candidate",
        )
    except Exception as exc:
        print("Chat memory probe failed:", type(exc).__name__, repr(exc))
        return MemoryProbeResult(
            query=query,
            attempted=True,
            rejection_reason=f"probe error: {type(exc).__name__}",
        )


def deduplicate_memory_context(
    deps,
    memory_context: Sequence[str],
    recent_context: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    recent_message_ids = {
        context_line_message_id(deps, line) for line in recent_context if context_line_message_id(deps, line)
    }
    result: list[str] = []
    dropped = 0
    for line in memory_context:
        payload = context_line_payload(deps, line)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            result.append(line)
            continue
        kept = []
        for message in messages:
            message_id = str(message.get("message_id") or "") if isinstance(message, dict) else ""
            if message_id and message_id in recent_message_ids:
                dropped += 1
                continue
            kept.append(message)
        if not kept:
            continue
        payload["messages"] = kept
        result.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return tuple(result), dropped


def budget_memory_context(
    deps,
    context: Sequence[str],
    *,
    max_hits: int,
    max_chars: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    used = 0
    for line in context:
        if not selected and len(line) > max_chars:
            continue
        if selected and used + len(line) > max(200, max_chars):
            continue
        selected.append(line)
        used += len(line)
        if len(selected) >= max(1, max_hits) or used >= max_chars:
            break
    return tuple(selected)


# ---------------------------------------------------------------------------
# Context budget enforcement
# ---------------------------------------------------------------------------


def apply_context_budget(deps, decision: ProcessingDecision) -> None:
    recent_candidate_count = (
        decision.recent_context_candidate_count or len(decision.chat_context)
    )
    if decision.reply_mode == "chat":
        recent_limit = max(1, getattr(deps.settings, "planned_chat_context_messages", 10))
        recent_chars = max(400, getattr(deps.settings, "planned_chat_context_max_chars", 1600))
        memory_hits, memory_chars = 3, 1000
    elif decision.reply_mode == "knowledge":
        recent_limit, recent_chars = 8, 1200
        memory_hits, memory_chars = 2, 800
    else:
        recent_limit, recent_chars = 8, 1200
        memory_hits, memory_chars = 2, 800
    topic_anchor_ids = tuple(dict.fromkeys(
        message_id
        for candidate in decision.topic_candidates
        for message_id in candidate.anchor_message_ids
    ))
    decision.chat_context = budget_recent_context(
        deps,
        decision.chat_context,
        max_messages=recent_limit,
        max_chars=recent_chars,
        required_message_ids=topic_anchor_ids,
    )
    decision.memory_context = budget_memory_context(
        deps,
        decision.memory_context,
        max_hits=memory_hits,
        max_chars=memory_chars,
    )
    decision.recent_context_candidate_count = recent_candidate_count
    decision.recent_context_selected_count = len(decision.chat_context)
    decision.recent_context_chars = sum(len(line) for line in decision.chat_context)
    decision.memory_context_chars = sum(len(line) for line in decision.memory_context)
    decision.memory_hit_count = len(decision.memory_context)
    decision.recent_context_selected_ids = tuple(
        message_id
        for line in decision.chat_context
        if (message_id := context_line_message_id(deps, line))
    )
    selected_chunk_ids: list[int] = []
    for line in decision.memory_context:
        try:
            chunk_id = int(context_line_payload(deps, line).get("chunk_id"))
        except (TypeError, ValueError):
            continue
        if chunk_id not in selected_chunk_ids:
            selected_chunk_ids.append(chunk_id)
    decision.memory_selected_chunk_ids = tuple(selected_chunk_ids)


# ---------------------------------------------------------------------------
# Memory enrichment
# ---------------------------------------------------------------------------


def enrich_decision_with_chat_memory(
    deps,
    decision: ProcessingDecision,
    item: dict,
    plan: MessagePlan | None,
    probe: MemoryProbeResult | None = None,
) -> ProcessingDecision:
    group_id = int(item.get("group_id") or 0)
    reply_message_id = str(item.get("reply_message_id") or "")
    requested_by_plan = bool(plan and plan.memory_needed)
    hard_reply_retrieval = bool(reply_message_id and not requested_by_plan)
    query = (
        (plan.memory_query if plan else "")
        or decision.effective_question
        or str(item.get("question") or "")
    )
    probe = probe or probe_chat_memory(deps, item, query)
    decision.memory_query = query
    decision.memory_retrieval_attempted = probe.attempted
    decision.memory_candidate_count = len(probe.context)
    decision.memory_rejection_reason = probe.rejection_reason
    decision.memory_selected_by_planner = bool(
        plan and (plan.selected_memory_chunk_ids or plan.memory_needed)
    )
    hits: Sequence[MemoryHit] = ()
    try:
        if not decision.should_reply:
            decision.memory_retrieval_mode = "probe_only"
            decision.memory_rejection_reason = "router decided no reply"
        elif not chat_memory_enabled_for_group(deps, group_id):
            decision.memory_retrieval_mode = "disabled"
            decision.memory_rejection_reason = "memory disabled for group"
        elif requested_by_plan or hard_reply_retrieval:
            decision.memory_retrieval_mode = "planned" if requested_by_plan else "reply_chain"
            decision.memory_retrieval_attempted = True
            hits = deps.chat_memory_manager.store.retrieve(
                group_id=group_id,
                query=query,
                speaker_id=deps.stable_member_id(group_id, str(item.get("user_id") or "")),
                reply_message_id=reply_message_id,
                exclude_message_id=str(item.get("message_id") or ""),
                participant_scope=plan.participant_scope if requested_by_plan else "reply_chain",
                time_scope=plan.time_scope if plan else "",
                limit=max(1, getattr(deps.settings, "chat_memory_max_hits", 6)),
                max_chars=max(200, getattr(deps.settings, "chat_memory_max_chars", 2400)),
            )
        else:
            decision.memory_retrieval_mode = "lexical_probe"
            if plan:
                selected_ids = set(plan.selected_memory_chunk_ids)
                hits = tuple(hit for hit in probe.hits if hit.chunk_id in selected_ids)
                if probe.context and not selected_ids:
                    decision.memory_rejection_reason = "planner did not select memory candidate"
            else:
                hits = probe.hits
        formatted = deps.chat_memory_manager.store.format_hits(hits) if hits else ()
        formatted, deduplicated = deduplicate_memory_context(deps, formatted, decision.chat_context)
        decision.context_deduplicated_count = deduplicated
        decision.memory_hit_count = len(formatted)
        decision.memory_needed = bool(formatted)
        if not formatted:
            decision.memory_rejection_reason = decision.memory_rejection_reason or "no selected memory candidate"
        if getattr(deps.settings, "chat_memory_shadow_mode", False):
            print("Chat memory shadow", group_id, len(formatted), query[:80])
            deps.write_message_audit(
                decision="memory_shadow",
                reason="long-term chat retrieval shadow result",
                group_id=group_id,
                user_id=item.get("user_id"),
                question=str(item.get("question") or ""),
                memory_query=query,
                memory_hit_count=len(formatted),
                memory_retrieval_attempted=True,
                memory_retrieval_mode=decision.memory_retrieval_mode,
                memory_candidate_count=decision.memory_candidate_count,
                memory_rejection_reason=decision.memory_rejection_reason,
                event_time=item.get("time"),
            )
        else:
            decision.memory_context = formatted
            if formatted:
                decision.draft_reply = ""

        if decision.should_reply and chat_memory_enabled_for_group(deps, group_id):
            related_message_ids = list(deps._message_ids(item))
            selected_recent_ids = tuple(
                message_id
                for line in decision.chat_context
                if (message_id := context_line_message_id(deps, line))
            )
            for message_id in (
                reply_message_id,
                *selected_recent_ids,
                *(message_id for hit in hits for message_id in hit.message_ids),
            ):
                value = str(message_id or "").strip()
                if value and value not in related_message_ids:
                    related_message_ids.append(value)
            topic_ids_list = list(dict.fromkeys(hit.topic_id for hit in hits if hit.topic_id))
            for message_id in selected_recent_ids:
                for assignment in deps.chat_memory_manager.store.topic_assignments_for_message(
                    group_id,
                    message_id,
                ):
                    if assignment.topic_id not in topic_ids_list:
                        topic_ids_list.append(assignment.topic_id)
            topic_ids = tuple(topic_ids_list)
            self_history = deps.chat_memory_manager.store.format_self_history(
                group_id=group_id,
                related_message_ids=related_message_ids,
                topic_ids=topic_ids,
                limit=max(1, getattr(deps.settings, "bot_self_history_max_turns", 3)),
                max_chars=max(200, getattr(deps.settings, "bot_self_history_max_chars", 1200)),
            )
            decision.self_history_context = self_history
            decision.self_history_candidate_count = len(self_history)
            decision.self_history_selected_count = len(self_history)
            decision.self_history_chars = sum(len(line) for line in self_history)
            selected_bot_ids: list[str] = []
            for line in self_history:
                try:
                    message_id = str(
                        context_line_payload(deps, line).get("bot_message", {}).get("message_id") or ""
                    )
                except (AttributeError, TypeError):
                    message_id = ""
                if message_id and message_id not in selected_bot_ids:
                    selected_bot_ids.append(message_id)
            decision.self_history_selected_message_ids = tuple(selected_bot_ids)
            if self_history:
                # Planner drafts are created before exact prior bot turns are loaded.
                decision.draft_reply = ""
            reasons: list[str] = []
            if reply_message_id:
                reasons.append("qq_reply")
            if selected_recent_ids:
                reasons.append("selected_recent_context")
            if hits:
                reasons.append("retrieved_topic")
            decision.self_history_reasons = tuple(reasons)
    except Exception as exc:
        decision.memory_rejection_reason = f"retrieval error: {type(exc).__name__}"
        print("Chat memory retrieval failed:", type(exc).__name__, repr(exc))
    apply_context_budget(deps, decision)
    if not getattr(deps.settings, "chat_memory_shadow_mode", False):
        deps.write_message_audit(
            decision="memory_retrieval",
            reason=decision.memory_rejection_reason or "memory context injected",
            group_id=group_id,
            user_id=item.get("user_id"),
            question=str(item.get("question") or ""),
            memory_query=query,
            memory_hit_count=decision.memory_hit_count,
            memory_retrieval_attempted=decision.memory_retrieval_attempted,
            memory_retrieval_mode=decision.memory_retrieval_mode,
            memory_candidate_count=decision.memory_candidate_count,
            memory_rejection_reason=decision.memory_rejection_reason,
            recent_context_candidate_count=decision.recent_context_candidate_count,
            recent_context_selected_count=decision.recent_context_selected_count,
            recent_context_chars=decision.recent_context_chars,
            memory_context_chars=decision.memory_context_chars,
            context_deduplicated_count=decision.context_deduplicated_count,
            recent_context_selected_ids=decision.recent_context_selected_ids,
            memory_selected_chunk_ids=decision.memory_selected_chunk_ids,
            memory_selected_by_planner=decision.memory_selected_by_planner,
            self_history_candidate_count=decision.self_history_candidate_count,
            self_history_selected_count=decision.self_history_selected_count,
            self_history_chars=decision.self_history_chars,
            self_history_selected_message_ids=decision.self_history_selected_message_ids,
            self_history_reasons=decision.self_history_reasons,
            topic_candidates=decision.topic_candidates,
            subject_candidates=decision.subject_candidates,
            subject_ambiguity=decision.subject_ambiguity,
            bot_involvement=decision.bot_involvement,
            reply_perspective=decision.reply_perspective,
            semantic_audience=decision.semantic_audience,
            participation_role=decision.participation_role,
            plan_context_revision=decision.plan_context_revision,
            plan_scene_version=decision.plan_scene_version,
            related_message_ids=decision.related_message_ids,
            event_time=item.get("time"),
        )
    return decision
