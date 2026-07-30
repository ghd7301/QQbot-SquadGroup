from __future__ import annotations


def process_chat_item(
    deps,
    item: dict,
    decision: deps.ProcessingDecision,
    lifecycle: deps.PendingItemLifecycle | None = None,
) -> deps.PendingItemLifecycle:
    lifecycle = lifecycle or deps.PendingItemLifecycle(item)
    for _ in (None,):
        model_started = deps.time.monotonic()
        routing_latency_ms = int(item.get("_routing_latency_ms") or 0)
        try:
            question = str(item["question"])
            group_id = int(item["group_id"])
            user_id = item.get("user_id")
            event_time = item.get("time")
            deadline = deps.reply_deadline(
                (
                    event_time
                    if event_time is not None
                    else item.get("_pending_created_at")
                ),
                False,
            )
            social_event_kind = deps.celebration_kind(question)
            mention_user_id = deps.response_mention_user_id(
                mentioned=bool(item.get("mentioned")),
                user_id=user_id,
                reply_mode="chat",
                question=question,
                mentioned_user_ids=tuple(item.get("mentioned_user_ids") or ()),
            )
            if deps.is_message_too_old(
                event_time, False, fallback_time=item.get("_pending_created_at")
            ):
                deps.write_message_audit(
                    decision="ignored",
                    reason="queued chat message too old",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue
            quota_reason = deps.chat_reply_quota_reason(group_id)
            if quota_reason:
                deps.write_message_audit(
                    decision="skipped",
                    reason=quota_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode="chat",
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue
            debounce_seconds = max(
                0.0, getattr(deps.settings, "chat_reply_debounce_seconds", 2.0)
            )
            if debounce_seconds:
                if deps.time.monotonic() + debounce_seconds >= deadline:
                    deps.write_message_audit(
                        decision="skipped",
                        reason="reply deadline exhausted during chat debounce",
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        reply_mode="chat",
                        event_time=event_time,
                    )
                    continue
                deps.time.sleep(debounce_seconds)
            chat_sequence = int(item.get("chat_sequence") or 0)
            latest_context = deps.recent_group_chat_context(
                group_id, now=deps.time.time(), focus_sequence=chat_sequence
            )
            scene_context = deps.current_group_chat_scene(
                group_id, focus_sequence=chat_sequence
            )
            (refreshed_decision, semantic_refresh_reason) = (
                deps.refresh_semantic_decision_for_late_context(
                    decision,
                    item,
                    latest_context,
                    scene_context=scene_context,
                    deadline=deadline,
                )
            )
            if refreshed_decision is None:
                deps.write_message_audit(
                    decision="skipped",
                    reason=semantic_refresh_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    chat_context=latest_context,
                    scene_context=scene_context,
                    semantic_intent=decision.semantic_intent,
                    semantic_topic=decision.semantic_topic,
                    semantic_confidence=decision.semantic_confidence,
                    semantic_audience=decision.semantic_audience,
                    participation_role=decision.participation_role,
                    plan_context_revision=decision.plan_context_revision,
                    plan_scene_version=decision.plan_scene_version,
                    related_message_ids=decision.related_message_ids,
                    semantic_replan_count=decision.semantic_replan_count,
                    semantic_replan_reason=semantic_refresh_reason,
                    event_time=event_time,
                )
                continue
            decision = refreshed_decision
            (decision.memory_context, dropped) = deps.deduplicate_memory_context(
                decision.memory_context, decision.chat_context
            )
            decision.context_deduplicated_count += dropped
            deps.apply_context_budget(decision)
            baseline_revision = int(
                decision.plan_context_revision
                or item.get("chat_sequence")
                or deps.latest_group_user_sequence(group_id)
            )
            celebration_target_key = mention_user_id or "unknown"
            if social_event_kind and deps.celebration_was_replied(
                group_id, celebration_target_key, social_event_kind
            ):
                deps.write_message_audit(
                    decision="skipped",
                    reason="celebration already acknowledged",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    chat_context=decision.chat_context,
                    mention_user_id=mention_user_id,
                    event_time=event_time,
                )
                continue
            generation_timeout = deps.remaining_reply_timeout(
                deadline,
                cap=getattr(deps.settings, "chat_generation_timeout_seconds", 7),
                reserve=getattr(deps.settings, "final_reply_review_timeout_seconds", 4),
            )
            if decision.draft_reply:
                raw_answer = decision.draft_reply
            elif generation_timeout:
                raw_answer = deps.answer_chat(
                    base_url=deps.settings.llm_base_url,
                    api_key=deps.settings.llm_api_key,
                    model=deps.settings.chat_model,
                    message=question,
                    context=decision.chat_context,
                    scene_context=scene_context,
                    semantic_context=deps.semantic_context_for_decision(decision),
                    memory_context=decision.memory_context,
                    self_history_context=decision.self_history_context,
                    timeout=generation_timeout,
                )
            else:
                raw_answer = ""
            if deps.is_chat_no_reply(raw_answer):
                model_latency_ms = routing_latency_ms + int(
                    (deps.time.monotonic() - model_started) * 1000
                )
                print("Chat generation: NO_REPLY", group_id, question)
                deps.write_message_audit(
                    decision="skipped",
                    reason="chat generation declined",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode="chat",
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    model_name=deps.settings.chat_model,
                    chat_context=decision.chat_context,
                    scene_context=scene_context,
                    semantic_intent=decision.semantic_intent,
                    semantic_topic=decision.semantic_topic,
                    implicit_meaning=decision.implicit_meaning,
                    capability=decision.capability,
                    semantic_confidence=decision.semantic_confidence,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            accepted_reason = (
                f"{social_event_kind} celebration accepted"
                if social_event_kind
                else "chat generation accepted"
            )
            answer = deps.finalize_model_answer(raw_answer, unsolicited=True)
            if not answer:
                model_latency_ms = routing_latency_ms + int(
                    (deps.time.monotonic() - model_started) * 1000
                )
                deps.write_message_audit(
                    decision="skipped",
                    reason="chat generation failed",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    model_name=deps.settings.chat_model,
                    chat_context=decision.chat_context,
                    scene_context=scene_context,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            unsafe_reason = deps.unsafe_or_repeated_reply(group_id, answer)
            if unsafe_reason:
                deps.write_message_audit(
                    decision="skipped",
                    reason=unsafe_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    answer=answer,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            (answer, review_reason, reviewed_revision) = deps.review_and_refresh_answer(
                question=question,
                answer=answer,
                decision=decision,
                group_id=group_id,
                mentioned=bool(item.get("mentioned")),
                admin=False,
                deadline=deadline,
                baseline_revision=baseline_revision,
                target_item=item,
            )
            model_latency_ms = routing_latency_ms + int(
                (deps.time.monotonic() - model_started) * 1000
            )
            if not answer:
                deps.write_message_audit(
                    decision="skipped",
                    reason=review_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    model_name=deps.settings.chat_model,
                    chat_context=decision.chat_context,
                    scene_context=scene_context,
                    semantic_intent=decision.semantic_intent,
                    semantic_topic=decision.semantic_topic,
                    implicit_meaning=decision.implicit_meaning,
                    semantic_confidence=decision.semantic_confidence,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            unsafe_reason = deps.unsafe_or_repeated_reply(group_id, answer)
            if unsafe_reason:
                deps.write_message_audit(
                    decision="skipped",
                    reason=unsafe_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    answer=answer,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            if deps.is_message_too_old(
                event_time, False, fallback_time=item.get("_pending_created_at")
            ):
                deps.write_message_audit(
                    decision="skipped",
                    reason="chat became stale during generation",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    model_name=deps.settings.chat_model,
                    chat_context=decision.chat_context,
                    scene_context=scene_context,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            if deps.settings.dry_run:
                print("Dry run chat answer:", group_id, answer)
                deps.write_message_audit(
                    decision="answered_dry_run",
                    reason=accepted_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode="chat",
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    model_name=deps.settings.chat_model,
                    chat_context=decision.chat_context,
                    scene_context=scene_context,
                    answer=answer,
                    mention_user_id=mention_user_id,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                deps.mark_chat_replied(group_id)
                if social_event_kind:
                    deps.mark_celebration_replied(
                        group_id, celebration_target_key, social_event_kind
                    )
                continue
            if not deps.acquire_reply_slot(block=False, reserve_slots=1):
                deps.write_message_audit(
                    decision="skipped",
                    reason="chat global rate limit",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    model_name=deps.settings.chat_model,
                    chat_context=decision.chat_context,
                    scene_context=scene_context,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            (answer, late_review_reason, reviewed_revision) = (
                deps.refresh_answer_for_late_context(
                    question=question,
                    answer=answer,
                    decision=decision,
                    group_id=group_id,
                    mentioned=bool(item.get("mentioned")),
                    admin=False,
                    deadline=deadline,
                    reviewed_revision=reviewed_revision,
                    target_item=item,
                )
            )
            if not answer:
                deps.write_message_audit(
                    decision="skipped",
                    reason=late_review_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=routing_latency_ms
                    + int((deps.time.monotonic() - model_started) * 1000),
                    model_name=deps.settings.chat_model,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            if late_review_reason != "context unchanged after final review":
                review_reason = "; ".join(
                    (value for value in (review_reason, late_review_reason) if value)
                )
                model_latency_ms = routing_latency_ms + int(
                    (deps.time.monotonic() - model_started) * 1000
                )
            with deps.group_send_lock(group_id):
                quota_reason = deps.chat_reply_quota_reason(group_id)
                if quota_reason:
                    deps.write_message_audit(
                        decision="skipped",
                        reason=f"{quota_reason} while waiting for group send lock",
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        reply_mode="chat",
                        model_latency_ms=model_latency_ms,
                        **deps.semantic_relation_audit_fields(decision),
                        event_time=event_time,
                    )
                    continue
                (can_send, blocked_reason, context_note) = deps.validate_locked_send(
                    group_id, item, reviewed_revision
                )
                if not can_send:
                    deps.write_message_audit(
                        decision="skipped",
                        reason=blocked_reason,
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        reply_mode="chat",
                        model_latency_ms=model_latency_ms,
                        answer=answer,
                        **deps.semantic_relation_audit_fields(decision),
                        event_time=event_time,
                    )
                    continue
                if context_note:
                    review_reason = "; ".join(
                        (value for value in (review_reason, context_note) if value)
                    )
                (bot_message_id, trigger_message_ids, turn_id) = (
                    deps.send_and_record_bot_turn(
                        group_id=group_id,
                        item=item,
                        answer=answer,
                        reply_mode="chat",
                        semantic_topic=decision.semantic_topic,
                        mention_user_id=mention_user_id,
                    )
                )
                deps.mark_chat_replied(group_id)
            if social_event_kind:
                deps.mark_celebration_replied(
                    group_id, celebration_target_key, social_event_kind
                )
            print("Answered chat", group_id, question)
            deps.write_message_audit(
                decision="answered",
                reason=f"{accepted_reason}; {review_reason}",
                group_id=group_id,
                user_id=user_id,
                question=question,
                has_context=decision.has_context,
                sources=decision.sources,
                reply_mode="chat",
                retrieval_score=decision.retrieval_score,
                retrieval_coverage=decision.retrieval_coverage,
                model_latency_ms=model_latency_ms,
                model_name=deps.settings.chat_model,
                chat_context=decision.chat_context,
                scene_context=scene_context,
                answer=answer,
                mention_user_id=mention_user_id,
                semantic_intent=decision.semantic_intent,
                semantic_topic=decision.semantic_topic,
                implicit_meaning=decision.implicit_meaning,
                capability=decision.capability,
                semantic_confidence=decision.semantic_confidence,
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
                semantic_replan_count=decision.semantic_replan_count,
                semantic_replan_reason=decision.semantic_replan_reason,
                planner_status=decision.planner_status,
                planner_latency_ms=decision.planner_latency_ms,
                self_history_candidate_count=decision.self_history_candidate_count,
                self_history_selected_count=decision.self_history_selected_count,
                self_history_chars=decision.self_history_chars,
                self_history_selected_message_ids=decision.self_history_selected_message_ids,
                self_history_reasons=decision.self_history_reasons,
                bot_message_id=str(bot_message_id or ""),
                generated_for_message_ids=trigger_message_ids,
                turn_id=turn_id,
                event_time=event_time,
            )
            deps.remember_conversation(
                group_id,
                user_id,
                question,
                decision,
                answer=answer,
                bot_message_id=bot_message_id,
                user_message_id=str(item.get("message_id") or ""),
                trigger_message_ids=trigger_message_ids,
                turn_id=turn_id,
            )
        except Exception as exc:
            lifecycle.handle_failure(repr(exc), deps.handle_pending_worker_failure)
            print("Chat worker error:", repr(exc))
            deps.write_message_audit(
                decision="error",
                reason=repr(exc),
                group_id=item.get("group_id"),
                user_id=item.get("user_id"),
                question=item.get("question", ""),
                reply_mode="chat",
                chat_context=decision.chat_context,
                event_time=item.get("time"),
            )
        finally:
            pass
    return lifecycle
