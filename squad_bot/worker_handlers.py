from __future__ import annotations


def process_worker_item(
    deps, item: dict, lane: str, lifecycle: deps.PendingItemLifecycle | None = None
) -> deps.PendingItemLifecycle:
    lifecycle = lifecycle or deps.PendingItemLifecycle(item)
    for _ in (None,):
        try:
            question = str(item["question"])
            mentioned = bool(item["mentioned"])
            group_id = int(item["group_id"])
            user_id = item.get("user_id")
            sender_role = item.get("sender_role", "")
            event_time = item.get("time")
            deadline = deps.reply_deadline(
                (
                    event_time
                    if event_time is not None
                    else item.get("_pending_created_at")
                ),
                mentioned,
            )
            admin_user = deps.is_admin_user(user_id, sender_role)
            command = deps.get_admin_command(question) if admin_user else ""
            if deps.is_restored_admin_command(item):
                print("Drop restored admin command", group_id, user_id, question)
                deps.write_message_audit(
                    decision="ignored",
                    reason="restored admin command discarded",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue
            if deps.message_already_covered_by_bot(group_id, item.get("message_id")):
                deps.write_message_audit(
                    decision="skipped",
                    reason="message already covered by an earlier bot turn",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue
            if deps.is_message_too_old(
                event_time, mentioned, fallback_time=item.get("_pending_created_at")
            ):
                age_limit = deps.message_max_age_seconds(mentioned)
                message_kind = "mentioned" if mentioned else "normal"
                print("Drop queued event: too old", group_id, message_kind, age_limit)
                deps.write_message_audit(
                    decision="ignored",
                    reason=f"queued {message_kind} message too old ({age_limit}s)",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue
            if command:
                answer = deps.answer_admin_command(
                    command, group_id=group_id, user_id=str(user_id or "")
                )
                if deps.settings.dry_run:
                    print("Dry run admin answer:", group_id, answer)
                    deps.write_message_audit(
                        decision="answered_dry_run",
                        reason=f"admin command {command}",
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        mentioned=mentioned,
                        event_time=item.get("time"),
                    )
                    continue
                deps.wait_for_rate_limit()
                with deps.group_send_lock(group_id):
                    deps.begin_pending_dispatch(item)
                    bot_message_id = deps.send_group_msg(
                        deps.settings.onebot_api_url,
                        group_id,
                        answer,
                        deps.settings.onebot_access_token,
                        reply_to_message_id=str(item.get("message_id") or ""),
                    )
                    item["_dispatch_completed"] = True
                    item["_sent_message_id"] = str(bot_message_id or "")
                    (trigger_message_ids, turn_id) = deps.bot_turn_metadata(
                        item, bot_message_id
                    )
                    deps.record_group_chat_message(
                        group_id,
                        deps.settings.bot_qq,
                        answer,
                        message_id=bot_message_id,
                        reply_message_id=str(item.get("message_id") or ""),
                        reply_target_user_id=str(user_id or ""),
                        reply_text=question,
                        generated_for_message_ids=trigger_message_ids,
                        turn_id=turn_id,
                        reply_mode="admin",
                        semantic_topic=command,
                    )
                print("Answered admin command", group_id, user_id, command)
                deps.write_message_audit(
                    decision="answered",
                    reason=f"admin command {command}",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=item.get("time"),
                )
                continue
            if (
                not mentioned
                and (not item.get("reply_message_id"))
                and (not item.get("mentioned_user_ids"))
                and deps.is_recent_duplicate_group_message(
                    group_id,
                    question,
                    focus_sequence=int(item.get("chat_sequence") or 0),
                    event_time=event_time,
                )
            ):
                deps.write_message_audit(
                    decision="skipped",
                    reason="duplicate group message before semantic planning",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue
            item["chat_context"] = list(
                deps.recent_group_chat_context(
                    group_id,
                    now=deps.time.time(),
                    focus_sequence=int(item.get("chat_sequence") or 0),
                )
            )
            item["_recent_context_candidate_count"] = len(item["chat_context"])
            planning_context_revision = deps.context_revision(item["chat_context"])
            planner_scene_context = deps.current_group_chat_scene(
                group_id, focus_sequence=int(item.get("chat_sequence") or 0)
            )
            followup_match = deps.followup_context_for(
                group_id,
                user_id,
                question,
                mentioned,
                reply_message_id=str(item.get("reply_message_id") or ""),
                reply_target_user_id=str(item.get("reply_target_user_id") or ""),
                reply_text=str(item.get("reply_text") or ""),
                bot_qq=deps.settings.bot_qq,
            )
            effective_question = deps.build_effective_question(question, followup_match)
            generation_question = deps.build_generation_question(
                question, followup_match
            )
            memory_probe = deps.probe_chat_memory(item, effective_question or question)
            model_started = deps.time.monotonic()
            planner_timeout = deps.remaining_reply_timeout(
                deadline,
                cap=deps.semantic_planner_timeout_cap(
                    mentioned=mentioned,
                    reply_target_user_id=str(item.get("reply_target_user_id") or ""),
                    explicit_knowledge_command=bool(
                        item.get("explicit_knowledge_command")
                    ),
                ),
            )
            if not planner_timeout:
                deps.write_message_audit(
                    decision="skipped",
                    reason="reply deadline exhausted before routing",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue
            selected_plan: list[deps.MessagePlan] = []
            decision = deps.should_process_message(
                question,
                mentioned,
                effective_question=effective_question,
                followup_of=(
                    followup_match.state.last_question if followup_match else ""
                ),
                followup_scope=followup_match.scope if followup_match else "",
                group_id=group_id,
                chat_context=tuple(item.get("chat_context") or ()),
                scene_context=planner_scene_context,
                memory_candidates=memory_probe.context,
                mentions_other=bool(item.get("mentions_other")),
                reply_target_user_id=str(item.get("reply_target_user_id") or ""),
                user_id=str(user_id or ""),
                sender_role=sender_role,
                planner_timeout=planner_timeout,
                plan_out=selected_plan,
                explicit_knowledge_command=bool(item.get("explicit_knowledge_command")),
            )
            decision.planner_latency_ms = int(
                (deps.time.monotonic() - model_started) * 1000
            )
            if selected_plan:
                decision.planner_status = "ok"
            decision.recent_context_candidate_count = int(
                item.get("_recent_context_candidate_count")
                or len(item.get("chat_context") or ())
            )
            if selected_plan:
                decision = deps.apply_semantic_plan_metadata(
                    decision,
                    selected_plan[0],
                    explicitly_addressed=mentioned
                    or str(item.get("reply_target_user_id") or "")
                    == deps.settings.bot_qq,
                    context_revision=planning_context_revision,
                    scene_context=planner_scene_context,
                    target_message_ids=deps._message_ids(item),
                )
            decision = deps.enrich_decision_with_chat_memory(
                decision,
                item,
                selected_plan[0] if selected_plan else None,
                memory_probe,
            )
            if not decision.should_reply:
                print("Skip message: model/router decided no reply", group_id, question)
                deps.write_message_audit(
                    decision="skipped",
                    reason=decision.reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    followup_of=decision.followup_of,
                    followup_scope=decision.followup_scope,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=int(
                        (deps.time.monotonic() - model_started) * 1000
                    ),
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
                    scene_context=planner_scene_context,
                    self_history_candidate_count=decision.self_history_candidate_count,
                    self_history_selected_count=decision.self_history_selected_count,
                    self_history_chars=decision.self_history_chars,
                    self_history_selected_message_ids=decision.self_history_selected_message_ids,
                    self_history_reasons=decision.self_history_reasons,
                    event_time=item.get("time"),
                )
                continue
            if decision.reply_mode == "chat":
                item["_routing_latency_ms"] = int(
                    (deps.time.monotonic() - model_started) * 1000
                )
                deps.chat_queue.put((item, decision))
                lifecycle.transfer()
                continue
            current_topic_key = deps.topic_key(question, decision)
            if (
                not mentioned
                and decision.reply_mode == "knowledge"
                and deps.is_topic_on_cooldown(group_id, current_topic_key)
            ):
                print(
                    "Skip message: recent topic cooldown",
                    group_id,
                    current_topic_key,
                    question,
                )
                deps.write_message_audit(
                    decision="skipped",
                    reason="recent topic cooldown",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    followup_of=decision.followup_of,
                    followup_scope=decision.followup_scope,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=int(
                        (deps.time.monotonic() - model_started) * 1000
                    ),
                    event_time=item.get("time"),
                )
                continue
            baseline_revision = int(
                item.get("chat_sequence") or deps.latest_group_user_sequence(group_id)
            )
            if decision.reply_mode in deps.LOCAL_REPLY_MODES:
                generation_timeout = 1 if deps.time.monotonic() < deadline else 0
            else:
                review_reserve = getattr(
                    deps.settings, "final_reply_review_timeout_seconds", 4
                )
                generation_timeout = deps.remaining_reply_timeout(
                    deadline,
                    cap=getattr(
                        deps.settings, "knowledge_generation_timeout_seconds", 10
                    ),
                    reserve=review_reserve,
                )
            if not generation_timeout:
                deps.write_message_audit(
                    decision="skipped",
                    reason="reply deadline exhausted before generation",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    reply_mode=decision.reply_mode,
                    event_time=event_time,
                )
                continue
            answer = deps.answer_for_decision(
                question,
                decision,
                decision.effective_question or generation_question,
                admin=admin_user,
                timeout=generation_timeout,
            )
            if deps.unsafe_or_repeated_reply(group_id, answer) and decision.draft_reply:
                decision.draft_reply = ""
                retry_timeout = deps.remaining_reply_timeout(
                    deadline,
                    cap=getattr(
                        deps.settings, "knowledge_generation_timeout_seconds", 10
                    ),
                    reserve=getattr(
                        deps.settings, "final_reply_review_timeout_seconds", 4
                    ),
                )
                if retry_timeout:
                    answer = deps.answer_for_decision(
                        question,
                        decision,
                        decision.effective_question or generation_question,
                        admin=admin_user,
                        timeout=retry_timeout,
                    )
            review_reason = ""
            reviewed_revision = deps.latest_group_user_sequence(group_id)
            if decision.reply_mode not in deps.LOCAL_REPLY_MODES:
                (answer, review_reason, reviewed_revision) = (
                    deps.review_and_refresh_answer(
                        question=question,
                        answer=answer,
                        decision=decision,
                        group_id=group_id,
                        mentioned=mentioned,
                        admin=admin_user,
                        deadline=deadline,
                        baseline_revision=baseline_revision,
                        target_item=item,
                    )
                )
                grounding_issue = deps.fallback_grounding_issue(decision, answer)
                if grounding_issue:
                    answer = ""
                    review_reason = grounding_issue
                if not answer:
                    deps.write_message_audit(
                        decision="skipped",
                        reason=review_reason,
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        mentioned=mentioned,
                        has_context=decision.has_context,
                        sources=decision.sources,
                        reply_mode=decision.reply_mode,
                        retrieval_score=decision.retrieval_score,
                        retrieval_coverage=decision.retrieval_coverage,
                        model_latency_ms=int(
                            (deps.time.monotonic() - model_started) * 1000
                        ),
                        semantic_intent=decision.semantic_intent,
                        semantic_topic=decision.semantic_topic,
                        semantic_confidence=decision.semantic_confidence,
                        **deps.semantic_relation_audit_fields(decision),
                        event_time=event_time,
                    )
                    continue
            model_latency_ms = int((deps.time.monotonic() - model_started) * 1000)
            unsafe_reason = deps.unsafe_or_repeated_reply(group_id, answer)
            if unsafe_reason:
                deps.write_message_audit(
                    decision="skipped",
                    reason=unsafe_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    answer=answer,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            mention_user_id = deps.response_mention_user_id(
                mentioned=mentioned,
                user_id=user_id,
                reply_mode=decision.reply_mode,
                question=question,
                mentioned_user_ids=tuple(item.get("mentioned_user_ids") or ()),
            )
            if deps.settings.dry_run:
                print("Dry run answer:", group_id, answer)
                deps.write_message_audit(
                    decision="answered_dry_run",
                    reason=decision.reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    followup_of=decision.followup_of,
                    followup_scope=decision.followup_scope,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    reply_message_id=str(item.get("reply_message_id") or ""),
                    reply_target_user_id=str(item.get("reply_target_user_id") or ""),
                    mention_user_id=mention_user_id,
                    answer=answer,
                    semantic_intent=decision.semantic_intent,
                    semantic_topic=decision.semantic_topic,
                    implicit_meaning=decision.implicit_meaning,
                    capability=decision.capability,
                    semantic_confidence=decision.semantic_confidence,
                    self_history_candidate_count=decision.self_history_candidate_count,
                    self_history_selected_count=decision.self_history_selected_count,
                    self_history_chars=decision.self_history_chars,
                    self_history_selected_message_ids=decision.self_history_selected_message_ids,
                    self_history_reasons=decision.self_history_reasons,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=item.get("time"),
                )
                if decision.reply_mode == "knowledge":
                    deps.mark_topic_replied(group_id, current_topic_key)
                continue
            if not deps.wait_for_rate_limit(deadline):
                deps.write_message_audit(
                    decision="skipped",
                    reason="reply deadline exhausted in global rate limit",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    **deps.semantic_relation_audit_fields(decision),
                    event_time=event_time,
                )
                continue
            if decision.reply_mode not in deps.LOCAL_REPLY_MODES:
                (answer, late_review_reason, reviewed_revision) = (
                    deps.refresh_answer_for_late_context(
                        question=question,
                        answer=answer,
                        decision=decision,
                        group_id=group_id,
                        mentioned=mentioned,
                        admin=admin_user,
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
                        mentioned=mentioned,
                        has_context=decision.has_context,
                        sources=decision.sources,
                        reply_mode=decision.reply_mode,
                        retrieval_score=decision.retrieval_score,
                        retrieval_coverage=decision.retrieval_coverage,
                        model_latency_ms=int(
                            (deps.time.monotonic() - model_started) * 1000
                        ),
                        **deps.semantic_relation_audit_fields(decision),
                        event_time=event_time,
                    )
                    continue
                if late_review_reason != "context unchanged after final review":
                    review_reason = "; ".join(
                        (
                            value
                            for value in (review_reason, late_review_reason)
                            if value
                        )
                    )
                    model_latency_ms = int(
                        (deps.time.monotonic() - model_started) * 1000
                    )
            with deps.group_send_lock(group_id):
                (can_send, blocked_reason, context_note) = deps.validate_locked_send(
                    group_id,
                    item,
                    reviewed_revision,
                    check_context=decision.reply_mode not in deps.LOCAL_REPLY_MODES,
                )
                if not can_send:
                    deps.write_message_audit(
                        decision="skipped",
                        reason=blocked_reason,
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        mentioned=mentioned,
                        has_context=decision.has_context,
                        sources=decision.sources,
                        reply_mode=decision.reply_mode,
                        retrieval_score=decision.retrieval_score,
                        retrieval_coverage=decision.retrieval_coverage,
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
                        reply_mode=decision.reply_mode,
                        semantic_topic=decision.semantic_topic,
                        mention_user_id=mention_user_id,
                        reply_to_trigger=mentioned,
                    )
                )
            print("Answered group", group_id, "question", question)
            deps.write_message_audit(
                decision="answered",
                reason=(
                    f"{decision.reason}; {review_reason}"
                    if review_reason
                    else decision.reason
                ),
                group_id=group_id,
                user_id=user_id,
                question=question,
                mentioned=mentioned,
                has_context=decision.has_context,
                sources=decision.sources,
                followup_of=decision.followup_of,
                followup_scope=decision.followup_scope,
                reply_mode=decision.reply_mode,
                retrieval_score=decision.retrieval_score,
                retrieval_coverage=decision.retrieval_coverage,
                model_latency_ms=model_latency_ms,
                reply_message_id=str(item.get("reply_message_id") or ""),
                reply_target_user_id=str(item.get("reply_target_user_id") or ""),
                mention_user_id=mention_user_id,
                answer=answer,
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
                event_time=item.get("time"),
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
            if decision.reply_mode == "knowledge":
                deps.mark_topic_replied(group_id, current_topic_key)
        except Exception as exc:
            lifecycle.handle_failure(repr(exc), deps.handle_pending_worker_failure)
            print(f"{lane} worker error:", repr(exc))
            deps.write_message_audit(
                decision="error",
                reason=repr(exc),
                group_id=item.get("group_id"),
                user_id=item.get("user_id"),
                question=item.get("question", ""),
                mentioned=item.get("mentioned", False),
                followup_of="",
                followup_scope="",
                event_time=item.get("time"),
            )
        finally:
            pass
    return lifecycle


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
