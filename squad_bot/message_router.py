from __future__ import annotations

import logging

# Quick-skip signals for unsolicited messages that rarely need a bot reply.

logger = logging.getLogger(__name__)
_SHORT_SKIP_MAX_LENGTH = 2
_QUESTION_SIGNALS = ("?", "？", "怎么", "如何", "什么", "啥", "为什么", "为啥",
                     "能不能", "可以吗", "是不是", "怎么办", "求助", "请问")


def should_process_message(
    deps,
    question: str,
    mentioned: bool,
    effective_question: str | None = None,
    followup_of: str = "",
    followup_scope: str = "",
    group_id: int = 0,
    chat_context: deps.Sequence[str] = (),
    scene_context: str = "",
    memory_candidates: deps.Sequence[str] = (),
    mentions_other: bool = False,
    reply_target_user_id: str = "",
    user_id: str = "",
    sender_role: str = "",
    planner_timeout: int | None = None,
    plan_out: list[deps.MessagePlan] | None = None,
    explicit_knowledge_command: bool = False,
) -> deps.ProcessingDecision:
    if mentioned and deps.is_identity_question(question):
        return deps.ProcessingDecision(
            True, "mentioned identity request", reply_mode="identity"
        )
    normalized = question.strip()
    query_text = effective_question or normalized

    # Fast pre-filter: skip unsolicited messages that are clearly casual noise
    # before expensive knowledge retrieval and LLM planner calls.
    explicitly_addressed = bool(
        mentioned
        or explicit_knowledge_command
        or reply_target_user_id == deps.settings.bot_qq
    )
    if not explicitly_addressed and not followup_of:
        lowered = normalized.lower()
        is_short = len(normalized) <= _SHORT_SKIP_MAX_LENGTH
        has_question = any(cue in lowered for cue in _QUESTION_SIGNALS)
        has_scene = bool(scene_context)
        if is_short and not has_question and not has_scene:
            return deps.ProcessingDecision(
                False, "unsolicited pre-filter: short message without signal",
            )

    knowledge_router = deps.KnowledgeRoutingService(
        lambda query: deps.retrieve_knowledge(query, deps.settings.max_context_chars),
        deps.is_strong_knowledge_match,
    )
    planner_enabled = bool(getattr(deps.settings, "semantic_planner_enabled", False))
    planner_request_allowed = bool(
        not planner_enabled
        or explicitly_addressed
        or deps.reserve_semantic_planner_request(lane="unsolicited")
    )
    circuit_open = planner_enabled and (not planner_request_allowed)
    plan = None
    if planner_request_allowed:
        try:
            plan = deps.semantic_plan_for_message(
                normalized,
                chat_context,
                scene_context=scene_context,
                memory_candidates=memory_candidates,
                mentioned=mentioned,
                mentions_other=mentions_other,
                reply_target_user_id=reply_target_user_id,
                timeout=planner_timeout,
            )
        finally:
            if planner_enabled:
                deps.record_semantic_planner_availability(
                    plan is not None,
                    lane="addressed" if explicitly_addressed else "unsolicited",
                )
    usable_plan = plan if deps.semantic_plan_is_usable(plan) else None
    if usable_plan is not None and plan_out is not None:
        plan_out.append(usable_plan)
    if usable_plan:
        chat_context = deps.context_selected_by_plan(chat_context, usable_plan)
        query_text = usable_plan.standalone_question or query_text
        participation_role = deps.derive_participation_role(
            usable_plan, chat_context, explicitly_addressed=explicitly_addressed
        )
        requested_capability = (
            usable_plan.capability if usable_plan.intent == "bot_meta" else "none"
        )
        if (
            explicitly_addressed
            and usable_plan.intent == "bot_meta"
            and ("self_identity" in usable_plan.risk_flags)
        ):
            return deps.ProcessingDecision(
                True,
                "semantic plan: addressed identity discussion",
                effective_question=query_text,
                reply_mode="identity",
                chat_context=tuple(chat_context),
                semantic_intent=usable_plan.intent,
                semantic_topic=usable_plan.topic_summary,
                implicit_meaning=usable_plan.implicit_meaning,
                semantic_confidence=usable_plan.confidence,
                risk_flags=usable_plan.risk_flags,
            )
        validated_capability = "none"
        if (
            usable_plan.intent == "bot_meta"
            and requested_capability != "none"
            and explicitly_addressed
            and (usable_plan.audience == "bot")
        ):
            verified_capability = deps.verify_bot_capability(
                base_url=deps.settings.llm_base_url,
                api_key=deps.settings.llm_api_key,
                model=getattr(
                    deps.settings, "semantic_planner_model", deps.settings.llm_model
                ),
                message=question,
                planned_capability=requested_capability,
                topic_summary=usable_plan.topic_summary,
                implicit_meaning=usable_plan.implicit_meaning,
                context=tuple(chat_context),
                timeout=max(1, min(3, planner_timeout or 3)),
            )
            if verified_capability == requested_capability:
                validated_capability = verified_capability
        if usable_plan.audience == "member" and (not mentioned):
            return deps.ProcessingDecision(
                False,
                "semantic plan: directed at another member",
                chat_context=tuple(chat_context),
                semantic_intent=usable_plan.intent,
                semantic_topic=usable_plan.topic_summary,
                implicit_meaning=usable_plan.implicit_meaning,
                capability=validated_capability,
                semantic_confidence=usable_plan.confidence,
            )
        if usable_plan.intent == "bot_meta" and requested_capability != "none":
            if explicitly_addressed and usable_plan.audience == "bot":
                if validated_capability == "none":
                    return deps.ProcessingDecision(
                        True,
                        "semantic plan: unverified bot capability fallback",
                        effective_question=query_text,
                        reply_mode="fallback",
                        chat_context=tuple(chat_context),
                        semantic_intent=usable_plan.intent,
                        semantic_topic=usable_plan.topic_summary,
                        implicit_meaning=usable_plan.implicit_meaning,
                        semantic_confidence=usable_plan.confidence,
                        risk_flags=usable_plan.risk_flags,
                    )
                if not deps.is_admin_user(user_id, sender_role):
                    return deps.ProcessingDecision(
                        True,
                        "semantic plan: bot capability access denied",
                        effective_question=query_text,
                        reply_mode="bot_meta",
                        chat_context=tuple(chat_context),
                        semantic_intent=usable_plan.intent,
                        semantic_topic=usable_plan.topic_summary,
                        implicit_meaning=usable_plan.implicit_meaning,
                        capability=validated_capability,
                        semantic_confidence=usable_plan.confidence,
                        risk_flags=usable_plan.risk_flags,
                    )
                return deps.ProcessingDecision(
                    True,
                    "semantic plan: explicit bot capability query",
                    effective_question=query_text,
                    reply_mode="bot_meta",
                    chat_context=tuple(chat_context),
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    capability=validated_capability,
                    semantic_confidence=usable_plan.confidence,
                )
            return deps.ProcessingDecision(
                False, "semantic plan: bot capability requires explicit bot address"
            )
        if usable_plan.intent == "bot_meta":
            if explicitly_addressed and usable_plan.audience == "bot":
                return deps.ProcessingDecision(
                    True,
                    "semantic plan: bot meta access boundary",
                    effective_question=query_text,
                    reply_mode="bot_meta",
                    chat_context=tuple(chat_context),
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    capability="none",
                    semantic_confidence=usable_plan.confidence,
                    risk_flags=usable_plan.risk_flags,
                )
            return deps.ProcessingDecision(
                False, "semantic plan: bot meta requires explicit bot address"
            )
        if usable_plan.intent == "admin":
            return deps.ProcessingDecision(
                False,
                "semantic plan: maintenance action requires explicit admin command",
                semantic_intent=usable_plan.intent,
                semantic_confidence=usable_plan.confidence,
            )
        if usable_plan.intent == "control_attempt":
            if explicitly_addressed:
                return deps.ProcessingDecision(
                    True,
                    "semantic plan: addressed control boundary",
                    effective_question=query_text,
                    reply_mode="control_boundary",
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    semantic_confidence=usable_plan.confidence,
                    semantic_audience="bot",
                    participation_role="addressed",
                    risk_flags=tuple(
                        dict.fromkeys((*usable_plan.risk_flags, "control"))
                    ),
                )
            return deps.ProcessingDecision(
                False,
                "semantic plan: unsolicited control attempt ignored",
                semantic_intent=usable_plan.intent,
                semantic_topic=usable_plan.topic_summary,
                semantic_confidence=usable_plan.confidence,
            )
        if usable_plan.intent in deps.SEMANTIC_CHAT_INTENTS:
            if usable_plan.intent == "hostile_abuse" and (
                not deps.allow_hostile_reply(group_id, user_id)
            ):
                return deps.ProcessingDecision(
                    False,
                    "semantic plan: repeated hostile abuse ignored",
                    reply_mode="chat",
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    semantic_confidence=usable_plan.confidence,
                )
            if explicitly_addressed:
                return deps.ProcessingDecision(
                    True,
                    f"semantic plan: addressed {usable_plan.intent}",
                    effective_question=query_text,
                    reply_mode="fallback",
                    chat_context=tuple(chat_context),
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    draft_reply=usable_plan.draft_reply,
                    semantic_confidence=usable_plan.confidence,
                )
            if participation_role in {"bystander", "uncertain"}:
                return deps.ProcessingDecision(
                    False,
                    f"semantic plan: no bot participation ({participation_role})",
                    reply_mode="chat",
                    chat_context=tuple(chat_context),
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    draft_reply=usable_plan.draft_reply,
                    semantic_confidence=usable_plan.confidence,
                )
            quota_reason = deps.chat_reply_quota_reason(group_id)
            chat_allowed = (
                deps.auto_reply_enabled
                and deps.settings.chat_reply_enabled
                and (
                    not deps.settings.chat_allowed_group_ids
                    or str(group_id) in deps.settings.chat_allowed_group_ids
                )
            )
            if usable_plan.reply_worthy and chat_allowed and (not quota_reason):
                return deps.ProcessingDecision(
                    True,
                    "semantic plan: chat candidate",
                    effective_question=query_text,
                    reply_mode="chat",
                    chat_context=tuple(chat_context),
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    draft_reply=usable_plan.draft_reply,
                    semantic_confidence=usable_plan.confidence,
                )
            return deps.ProcessingDecision(
                False,
                quota_reason or "semantic plan: no natural chat entry",
                reply_mode="chat",
                chat_context=tuple(chat_context),
                semantic_intent=usable_plan.intent,
                semantic_topic=usable_plan.topic_summary,
                implicit_meaning=usable_plan.implicit_meaning,
                draft_reply=usable_plan.draft_reply,
                semantic_confidence=usable_plan.confidence,
            )
        if usable_plan.intent == "action" and (not mentioned):
            return deps.ProcessingDecision(
                False,
                "semantic plan: requires real-world participation",
                semantic_intent=usable_plan.intent,
                semantic_topic=usable_plan.topic_summary,
                semantic_confidence=usable_plan.confidence,
            )
        if usable_plan and usable_plan.intent != "knowledge":
            if explicitly_addressed:
                return deps.ProcessingDecision(
                    True,
                    f"semantic plan: addressed {usable_plan.intent}",
                    effective_question=query_text,
                    reply_mode="fallback",
                    chat_context=tuple(chat_context),
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    semantic_confidence=usable_plan.confidence,
                )
            return deps.ProcessingDecision(
                False,
                f"semantic plan: unsupported unsolicited intent {usable_plan.intent}",
                chat_context=tuple(chat_context),
                semantic_intent=usable_plan.intent,
                semantic_topic=usable_plan.topic_summary,
                semantic_confidence=usable_plan.confidence,
            )
    if planner_enabled and (not usable_plan) and explicitly_addressed:
        initial_selection = knowledge_router.lookup(query_text)
        initial_result = initial_selection.result
        initial_strong_match = initial_selection.strong_match
        decision = deps.semantic_routing.addressed_planner_fallback(
            result=initial_result,
            query_text=query_text,
            followup_of=followup_of,
            followup_scope=followup_scope,
            chat_context=chat_context,
            explicit_knowledge_command=explicit_knowledge_command,
            strong_match=initial_strong_match,
            fallback_allowed=deps.settings.llm_fallback_enabled
            and (
                explicitly_addressed or not deps.settings.fallback_only_when_mentioned
            ),
            low_confidence=plan is not None,
        )
        if decision is not None:
            return decision
    if planner_enabled and (not usable_plan) and (not explicitly_addressed):
        return deps.semantic_routing.unavailable_unsolicited_decision(
            query_text, circuit_open=circuit_open, low_confidence=plan is not None
        )
    selection = knowledge_router.lookup(query_text)
    result = selection.result
    context = result.context
    sources = result.sources
    strong_match = selection.strong_match
    if (
        not usable_plan
        and (not strong_match)
        and chat_context
        and (mentioned or deps.looks_like_direct_question(normalized))
    ):
        rewritten = deps.contextual_retrieval_question(normalized, chat_context)
        selection = knowledge_router.contextual_candidate(
            selection,
            rewritten,
            min_confidence=getattr(
                deps.settings, "contextual_query_min_confidence", 0.75
            ),
        )
        query_text = selection.query
        result = selection.result
        context = result.context
        sources = result.sources
        strong_match = selection.strong_match
    if (
        not strong_match
        and usable_plan is not None
        and (usable_plan.intent == "knowledge")
        and getattr(deps.settings, "knowledge_gap_log_enabled", False)
    ):
        deps.record_knowledge_gap(query_text, result)
    if not context or not strong_match:
        # High-confidence knowledge intent override: when the planner is very
        # confident this is a knowledge question and we have SOME context
        # (just below strong_match threshold), go directly to the knowledge
        # path instead of the slower fallback LLM. This avoids the fallback
        # path's extra overhead and prevents deterministic_review_failure
        # from returning a generic refusal.
        if (
            context
            and usable_plan is not None
            and usable_plan.intent == "knowledge"
            and usable_plan.confidence >= 0.9
            and usable_plan.capability == "none"
            and (explicitly_addressed or getattr(deps, "auto_reply_enabled", False))
        ):
            return deps.attach_knowledge_result(
                deps.ProcessingDecision(
                    True,
                    "high-confidence knowledge override (weak match)",
                    True,
                    tuple(sources),
                    query_text,
                    followup_of,
                    followup_scope,
                    "knowledge",
                    tuple(chat_context),
                    result.top_score,
                    result.query_coverage,
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    semantic_confidence=usable_plan.confidence,
                ),
                query_text,
                result,
            )
        fallback_allowed = deps.settings.llm_fallback_enabled and (
            explicitly_addressed or not deps.settings.fallback_only_when_mentioned
        )
        if fallback_allowed:
            fallback_reason = "weak-context llm fallback" if context else "llm fallback"
            if mentioned:
                fallback_reason = "mentioned " + fallback_reason
            logger.info("Fallback reply: %s %s", fallback_reason, normalized)
            return deps.ProcessingDecision(
                should_reply=True,
                reason=fallback_reason,
                has_context=bool(context),
                sources=tuple(sources),
                effective_question=query_text,
                followup_of=followup_of,
                followup_scope=followup_scope,
                reply_mode="fallback",
                retrieval_score=result.top_score,
                retrieval_coverage=result.query_coverage,
                chat_context=tuple(chat_context) if usable_plan else (),
                semantic_intent=usable_plan.intent if usable_plan else "",
                semantic_topic=usable_plan.topic_summary if usable_plan else "",
                implicit_meaning=usable_plan.implicit_meaning if usable_plan else "",
                capability=(
                    usable_plan.capability
                    if usable_plan and usable_plan.intent == "bot_meta"
                    else "none"
                ),
                semantic_confidence=usable_plan.confidence if usable_plan else 0.0,
            )
        if mentioned:
            logger.warning("Skip mentioned message: fallback disabled %s", normalized)
            return deps.ProcessingDecision(
                should_reply=False,
                reason="no strong knowledge context and fallback disabled",
                has_context=bool(context),
                sources=tuple(sources),
                effective_question=query_text,
                followup_of=followup_of,
                followup_scope=followup_scope,
                retrieval_score=result.top_score,
                retrieval_coverage=result.query_coverage,
            )
        decision = deps.consider_chat_reply(
            normalized,
            group_id=group_id,
            chat_context=chat_context,
            mentions_other=mentions_other,
            has_context=bool(context),
            sources=sources,
            query_text=query_text,
            followup_of=followup_of,
            followup_scope=followup_scope,
        )
        decision.retrieval_score = result.top_score
        decision.retrieval_coverage = result.query_coverage
        return decision
    if usable_plan and usable_plan.intent == "knowledge":
        if explicitly_addressed or (
            deps.auto_reply_enabled and usable_plan.reply_worthy
        ):
            return deps.attach_knowledge_result(
                deps.ProcessingDecision(
                    True,
                    "semantic plan: knowledge question",
                    True,
                    tuple(sources),
                    query_text,
                    followup_of,
                    followup_scope,
                    "knowledge",
                    tuple(chat_context),
                    result.top_score,
                    result.query_coverage,
                    semantic_intent=usable_plan.intent,
                    semantic_topic=usable_plan.topic_summary,
                    implicit_meaning=usable_plan.implicit_meaning,
                    capability=(
                        usable_plan.capability
                        if usable_plan.intent == "bot_meta"
                        else "none"
                    ),
                    semantic_confidence=usable_plan.confidence,
                ),
                query_text,
                result,
            )
        return deps.ProcessingDecision(
            False,
            "semantic plan: knowledge statement not requesting reply",
            has_context=True,
            sources=tuple(sources),
            semantic_intent=usable_plan.intent,
            semantic_confidence=usable_plan.confidence,
        )
    if (
        len(normalized) < 5
        and (not deps.has_auto_reply_keyword(query_text))
        and (not followup_of)
    ):
        logger.warning("Skip message: too short for auto reply %s", normalized)
        return deps.ProcessingDecision(
            False,
            "too short for auto reply",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            retrieval_score=result.top_score,
            retrieval_coverage=result.query_coverage,
        )
    if mentioned:
        logger.info("Mention reply: knowledge context found %s", normalized)
        return deps.attach_knowledge_result(
            deps.ProcessingDecision(
                True,
                "mentioned with strong knowledge context",
                True,
                tuple(sources),
                query_text,
                followup_of,
                followup_scope,
                "knowledge",
                tuple(chat_context),
                result.top_score,
                result.query_coverage,
                semantic_intent=usable_plan.intent if usable_plan else "",
                semantic_topic=usable_plan.topic_summary if usable_plan else "",
                implicit_meaning=usable_plan.implicit_meaning if usable_plan else "",
                capability=(
                    usable_plan.capability
                    if usable_plan and usable_plan.intent == "bot_meta"
                    else "none"
                ),
                semantic_confidence=usable_plan.confidence if usable_plan else 0.0,
            ),
            query_text,
            result,
        )
    if not deps.auto_reply_enabled:
        logger.warning("Skip message: auto reply disabled %s", normalized)
        return deps.ProcessingDecision(
            False,
            "auto reply disabled",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            retrieval_score=result.top_score,
            retrieval_coverage=result.query_coverage,
        )
    if deps.looks_like_assignment_to_humans(normalized):
        logger.warning("Skip message: looks like assignment to humans %s", normalized)
        return deps.ProcessingDecision(
            False,
            "looks like assignment to humans",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            retrieval_score=result.top_score,
            retrieval_coverage=result.query_coverage,
        )
    if not deps.looks_like_direct_question(
        normalized
    ) or not deps.has_auto_reply_keyword(query_text):
        decision = deps.consider_chat_reply(
            normalized,
            group_id=group_id,
            chat_context=chat_context,
            mentions_other=mentions_other,
            has_context=True,
            sources=sources,
            query_text=query_text,
            followup_of=followup_of,
            followup_scope=followup_scope,
        )
        decision.retrieval_score = result.top_score
        decision.retrieval_coverage = result.query_coverage
        return decision
    if deps.has_auto_reply_keyword(query_text):
        logger.info("Auto reply: matched Squad keyword %s", normalized)
        return deps.attach_knowledge_result(
            deps.ProcessingDecision(
                True,
                "matched Squad keyword with strong context",
                True,
                tuple(sources),
                query_text,
                followup_of,
                followup_scope,
                "knowledge",
                tuple(chat_context),
                result.top_score,
                result.query_coverage,
            ),
            query_text,
            result,
        )
