from __future__ import annotations

import re
import time

from ..models import ProcessingDecision


def locked_send_context_change(
    deps,
    group_id: int,
    reviewed_revision: int,
    item: dict,
) -> tuple[bool, tuple[str, ...], str]:
    """Classify messages that arrived after review without model calls under the send lock."""
    bot_qq = str(deps.settings.bot_qq or "")
    original_user_id = str(item.get("user_id") or "")
    trigger_message_ids = set(deps._message_ids(item))
    direct_to_bot = bool(
        item.get("mentioned")
        or (bot_qq and str(item.get("reply_target_user_id") or "") == bot_qq)
    )
    with deps.chat_history_lock:
        delta = tuple(
            message
            for message in deps.group_chat_history.get(int(group_id), ())
            if message.sequence > reviewed_revision
            and message.user_id != bot_qq
            and message.message_status == "active"
        )
    if not delta:
        return False, (), "unchanged"

    delta_ids = tuple(
        message.message_id or f"sequence:{message.sequence}" for message in delta
    )
    human_thread_by_sender: dict[str, tuple[str, float]] = {}
    continuation_window = max(
        0.0,
        float(getattr(deps.settings, "message_fragment_max_wait_seconds", 8)),
    )

    for message in delta:
        reply_target = str(message.reply_target_user_id or "")
        if message.reply_message_id:
            if not reply_target:
                return True, delta_ids, "new reply target could not be resolved"
            if reply_target == bot_qq or message.reply_message_id in trigger_message_ids:
                return True, delta_ids, "new message directly extends or answers the bot turn"
            human_thread_by_sender[message.user_id] = (
                reply_target,
                message.received_time,
            )
            continue
        if message.mentioned_bot:
            if message.user_id == original_user_id:
                return True, delta_ids, "original sender sent another message to the bot"
            continue
        if message.mentioned_user_ids:
            human_thread_by_sender[message.user_id] = (
                str(message.mentioned_user_ids[0]),
                message.received_time,
            )
            continue
        if message.user_id == original_user_id:
            human_thread = human_thread_by_sender.get(message.user_id)
            if (
                human_thread
                and human_thread[0] != bot_qq
                and 0
                <= message.received_time - human_thread[1]
                <= continuation_window
            ):
                human_thread_by_sender[message.user_id] = (
                    human_thread[0],
                    message.received_time,
                )
                continue
            return True, delta_ids, "original sender added an ambiguous follow-up"
        if not direct_to_bot:
            return True, delta_ids, "unsolicited reply has a newer ambiguous group message"

    return False, delta_ids, "only unrelated directed messages arrived"


def validate_locked_send(
    deps,
    group_id: int,
    item: dict,
    reviewed_revision: int,
    *,
    check_context: bool = True,
) -> tuple[bool, str, str]:
    """Return whether a candidate may send, a blocking reason, and an audit note."""
    if deps.message_already_covered_by_bot(group_id, item.get("message_id")):
        return False, "message covered while waiting to send", ""
    if not check_context:
        return True, "", ""
    invalidated, delta_message_ids, relation = deps.locked_send_context_change(
        group_id,
        reviewed_revision,
        item,
    )
    delta_text = ",".join(delta_message_ids)
    if invalidated:
        return (
            False,
            "context invalidated while waiting for group send lock: "
            f"{relation}; delta={delta_text}",
            "",
        )
    note = (
        f"send-lock context preserved: {relation}; delta={delta_text}"
        if delta_message_ids
        else ""
    )
    return True, "", note


def unsafe_or_repeated_reply(deps, group_id: int, answer: str, *, limit: int = 10) -> str:
    if re.search(r"\bmember_[0-9a-f]{6,}\b", answer, flags=re.I):
        return "internal member id leaked"
    normalized = re.sub(r"[\W_]+", "", answer.lower())
    if len(normalized) < 6:
        return ""
    with deps.chat_history_lock:
        recent_bot_answers = [
            item.text
            for item in deps.group_chat_history.get(group_id, ())
            if item.user_id == deps.settings.bot_qq
        ][-limit:]
    for previous in recent_bot_answers:
        previous_normalized = re.sub(r"[\W_]+", "", previous.lower())
        if normalized == previous_normalized:
            return "duplicate recent bot reply"
    return ""


def is_recent_duplicate_group_message(
    deps,
    group_id: int,
    text: str,
    *,
    focus_sequence: int,
    event_time=None,
    window_seconds: int = 60,
) -> bool:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    if len(normalized) < 4 or focus_sequence <= 0:
        return False
    try:
        current_time = float(event_time)
    except (TypeError, ValueError):
        current_time = time.time()
    with deps.chat_history_lock:
        for item in reversed(deps.group_chat_history.get(group_id, ())):
            if item.sequence >= focus_sequence:
                continue
            if current_time - item.timestamp > max(0, window_seconds):
                break
            previous = re.sub(r"[\W_]+", "", item.text.lower())
            if previous == normalized:
                return True
    return False


def reply_deadline(deps, event_time, mentioned: bool) -> float:
    total = (
        getattr(deps.settings, "mentioned_reply_total_timeout_seconds", 15)
        if mentioned
        else getattr(deps.settings, "normal_reply_total_timeout_seconds", 10)
    )
    try:
        elapsed = max(0.0, time.time() - float(event_time))
    except (TypeError, ValueError):
        elapsed = 0.0
    return time.monotonic() + max(0.0, float(total) - elapsed)


def remaining_reply_timeout(
    deadline: float,
    *,
    cap: int,
    reserve: int = 0,
) -> int:
    remaining = deadline - time.monotonic() - max(0, reserve)
    if remaining < 1:
        return 0
    return max(1, min(int(cap), int(remaining)))


def review_and_refresh_answer(
    deps,
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
    original_context = tuple(decision.chat_context)
    candidate = answer
    regenerated = decision.reply_regenerated

    while True:
        latest_revision = deps.latest_group_user_sequence(group_id)
        review_mode = getattr(deps.settings, "final_reply_review_mode", "adaptive")
        risky_intents = {
            "banter_at_bot",
            "control_attempt",
            "third_party_attack",
            "genuine_criticism",
            "hostile_abuse",
            "action",
            "unclear",
        }
        context_changed = baseline_revision is None or latest_revision > baseline_revision
        context_change_note = ""
        if (
            context_changed
            and baseline_revision is not None
            and target_item
            and target_item.get("user_id")
            and target_item.get("chat_sequence")
        ):
            context_changed, delta_ids, relation = deps.locked_send_context_change(
                group_id, baseline_revision, target_item
            )
            if not context_changed:
                rendered_ids = ",".join(delta_ids)
                context_change_note = (
                    f"adaptive final review skipped: {relation}"
                    + (f"; delta={rendered_ids}" if rendered_ids else "")
                )
        requires_model_review = (
            review_mode == "always"
            or context_changed
            or regenerated
            or bool(decision.risk_flags)
            or decision.semantic_intent in risky_intents
            or (
                bool(decision.self_history_context)
                and decision.bot_involvement in {"subject", "participant", "uncertain"}
            )
            or (
                decision.reply_mode == "fallback"
                and (
                    decision.semantic_audience == "unclear"
                    or decision.participation_role == "uncertain"
                    or decision.semantic_confidence < 0.8
                    or (
                        decision.semantic_intent == "knowledge"
                        and not deps.is_strong_knowledge_match(
                            decision.retrieval_score,
                            decision.retrieval_coverage,
                        )
                    )
                )
            )
            or (
                decision.reply_mode == "chat"
                and decision.reply_perspective in {"first_person", "neutral"}
            )
            or (
                decision.reply_mode == "chat"
                and (
                    decision.semantic_intent != "normal_chat"
                    or decision.semantic_confidence < 0.8
                )
            )
        )
        if not requires_model_review:
            unsafe_reason = deps.unsafe_or_repeated_reply(group_id, candidate)
            if unsafe_reason:
                return "", unsafe_reason, latest_revision
            return (
                candidate,
                context_change_note or "adaptive final review skipped",
                latest_revision,
            )
        latest_context = deps.recent_group_chat_context(
            group_id,
            now=time.time(),
            focus_sequence=int((target_item or {}).get("chat_sequence") or 0),
        )
        review_timeout = deps.remaining_reply_timeout(
            deadline,
            cap=getattr(deps.settings, "final_reply_review_timeout_seconds", 4),
        )
        if not review_timeout:
            if mentioned or decision.reply_mode in {"knowledge", "fallback"}:
                degraded = deps.deterministic_review_failure_answer(
                    decision, candidate, mentioned=mentioned,
                    context_changed=context_changed,
                )
                if degraded:
                    return degraded, "reply deadline exhausted; deterministic fallback", latest_revision
                if candidate:
                    return candidate, "reply deadline exhausted; raw candidate sent", latest_revision
            return "", "reply deadline exhausted before final review", latest_revision
        review = deps.review_candidate_reply(
            base_url=deps.settings.llm_base_url,
            api_key=deps.settings.llm_api_key,
            model=getattr(
                deps.settings, "final_reply_review_model", deps.settings.chat_model
            ),
            original_message=question,
            candidate_reply=candidate,
            original_context=original_context,
            latest_context=latest_context,
            self_history_context=decision.self_history_context,
            reply_mode=decision.reply_mode,
            mentioned=mentioned,
            topic_summary=decision.semantic_topic,
            semantic_context=deps.semantic_context_for_decision(decision),
            original_message_ids=deps._message_ids(target_item or {}),
            knowledge_sources=decision.sources,
            candidate_knowledge_context=(
                decision.knowledge_result.context
                if decision.knowledge_result is not None
                else ""
            ),
            retrieval_score=decision.retrieval_score,
            retrieval_coverage=decision.retrieval_coverage,
            allow_regenerate=not regenerated,
            timeout=review_timeout,
        )
        if not review or review.confidence < 0.6:
            failure_reason = (
                "final review unavailable"
                if review is None
                else "final review low confidence"
            )
            degraded = deps.deterministic_review_failure_answer(
                decision,
                candidate,
                mentioned=mentioned,
                context_changed=context_changed,
            )
            if degraded:
                return (
                    degraded,
                    f"{failure_reason}; deterministic addressed degradation",
                    latest_revision,
                )
            if mentioned or decision.reply_mode in {"knowledge", "fallback"}:
                if candidate:
                    return candidate, f"{failure_reason}; raw candidate sent", latest_revision
            return "", failure_reason, latest_revision
        deps.merge_review_message_ids(target_item, review, latest_context)
        if review.action == "drop":
            if mentioned or decision.reply_mode in {"knowledge", "fallback"}:
                degraded = deps.deterministic_review_failure_answer(
                    decision, candidate, mentioned=mentioned,
                    context_changed=context_changed,
                )
                if degraded:
                    return degraded, f"review dropped [{review.context_relation}]; deterministic fallback", latest_revision
                if candidate:
                    return candidate, f"review dropped [{review.context_relation}]; raw candidate sent", latest_revision
            return (
                "",
                f"final review dropped [{review.context_relation}]: {review.reason}",
                latest_revision,
            )
        if review.action == "revise":
            candidate = deps.finalize_model_answer(
                review.revised_reply,
                unsolicited=decision.reply_mode == "chat",
            )
            if not candidate:
                return "", "final review produced empty revision", latest_revision
            return (
                candidate,
                f"final review revised [{review.context_relation}]: {review.reason}",
                latest_revision,
            )
        if review.action == "send":
            return (
                candidate,
                f"final review accepted [{review.context_relation}]: {review.reason}",
                latest_revision,
            )

        regenerated = True
        decision.reply_regenerated = True
        semantic_replan_reason = ""
        if (
            target_item
            and decision.semantic_audience != "unclear"
            and decision.semantic_replan_count < 1
        ):
            refreshed_decision, semantic_replan_reason = (
                deps.refresh_semantic_decision_for_late_context(
                    decision,
                    target_item,
                    latest_context,
                    scene_context=deps.current_group_chat_scene(
                        group_id,
                        focus_sequence=int(target_item.get("chat_sequence") or 0),
                    ),
                    deadline=deadline,
                )
            )
            if refreshed_decision is None:
                if mentioned or decision.reply_mode in {"knowledge", "fallback"} and candidate:
                    return candidate, f"{semantic_replan_reason}; raw candidate sent", latest_revision
                return "", semantic_replan_reason, latest_revision
            decision = refreshed_decision
        updated_question = (
            decision.effective_question
            if semantic_replan_reason.startswith("related late context")
            else review.updated_question
        )
        if not semantic_replan_reason.startswith("related late context"):
            decision.chat_context = tuple(latest_context)
        decision.effective_question = updated_question
        decision.draft_reply = ""
        reserve = getattr(deps.settings, "final_reply_review_timeout_seconds", 4)
        generation_cap = (
            getattr(deps.settings, "chat_generation_timeout_seconds", 7)
            if decision.reply_mode == "chat"
            else getattr(deps.settings, "knowledge_generation_timeout_seconds", 10)
        )
        generation_timeout = deps.remaining_reply_timeout(
            deadline,
            cap=generation_cap,
            reserve=reserve,
        )
        if not generation_timeout:
            if mentioned or decision.reply_mode in {"knowledge", "fallback"} and candidate:
                return candidate, "reply deadline exhausted before regeneration; raw candidate sent", latest_revision
            return "", "reply deadline exhausted before regeneration", latest_revision
        candidate = deps.answer_for_decision(
            updated_question,
            decision,
            updated_question,
            admin=admin,
            timeout=generation_timeout,
        )
        if not candidate:
            return "", "regeneration produced no answer", latest_revision
        baseline_revision = latest_revision


def refresh_answer_for_late_context(
    deps,
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
    """Re-review only when messages arrived after the previous review."""
    latest_revision = deps.latest_group_user_sequence(group_id)
    if latest_revision <= reviewed_revision:
        return answer, "context unchanged after final review", reviewed_revision
    refreshed_answer, reason, revision = deps.review_and_refresh_answer(
        question=question,
        answer=answer,
        decision=decision,
        group_id=group_id,
        mentioned=mentioned,
        admin=admin,
        deadline=deadline,
        baseline_revision=reviewed_revision,
        target_item=target_item,
    )
    if refreshed_answer:
        unsafe_reason = deps.unsafe_or_repeated_reply(group_id, refreshed_answer)
        if unsafe_reason:
            return "", unsafe_reason, revision
        grounding_issue = deps.fallback_grounding_issue(decision, refreshed_answer)
        if grounding_issue:
            return "", grounding_issue, revision
    return refreshed_answer, reason, revision
