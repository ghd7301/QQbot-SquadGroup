from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .fact_guard import unsupported_fallback_precise_facts
from .knowledge import ContextResult
from .llm import is_provider_refusal_text
from .models import ProcessingDecision


def is_model_error_answer(answer: str) -> bool:
    if not answer:
        return False
    return (
        answer.startswith(("模型接口", "还没有配置模型 API Key"))
        or is_provider_refusal_text(answer)
        or answer.startswith(("{", "["))  # raw JSON error body
        or "api key" in answer.lower()[:200]
        or "rate limit" in answer.lower()[:200]
    )


def finalize_model_answer(deps, answer: str, *, unsolicited: bool = False) -> str:
    if is_model_error_answer(answer):
        if unsolicited:
            return ""
        return "这会儿回复服务有点忙，稍后再问我一下。"
    normalized = deps.normalize_model_answer(answer, deps.settings.max_answer_chars)
    if normalized:
        return normalized
    if unsolicited:
        return ""
    return "这会儿回复服务有点忙，稍后再问我一下。"


def answer_bot_meta(deps, capability: str, *, admin: bool) -> str:
    if not admin:
        return "这类内部状态只对管理员开放。"
    knowledge_path = Path(deps.settings.knowledge_dir)
    files = sorted(path.name for path in knowledge_path.glob("*.md"))
    if capability == "knowledge_files":
        if not files:
            return "当前没有发现可加载的知识库文件。"
        return "当前加载的知识库文件有：" + "、".join(files)
    if capability == "knowledge_status":
        return f"知识库已加载，当前有 {len(files)} 个文件、{len(deps.kb.chunks)} 个片段。"
    if capability == "model_status":
        return f"知识问答使用 {deps.settings.llm_model}，闲聊使用 {deps.settings.chat_model}。"
    if capability in {"runtime_status", "health"}:
        queued = deps.message_queue.qsize() + deps.normal_message_queue.qsize() + deps.chat_queue.qsize()
        return f"服务正在运行，知识片段 {len(deps.kb.chunks)} 个，队列 {queued} 条。"
    return "可以查看知识库加载状态、知识库文件、当前模型和服务健康状态。"


def answer_question(
    deps,
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
    if deps.is_identity_question(question):
        return (
            "叫我新兵营教官就行，主要给刚入坑 Squad 的兄弟答疑。"
            "HAB、FOB、医疗兵、反坦、搜不到服、卡三点、TS 设置这些都能问。"
            "要是问到本服规则，我不乱拍板，按群公告和管理员说法来。"
        )

    llm_question = effective_question or question
    result = knowledge_result or deps.retrieve_knowledge(
        retrieval_question or llm_question,
        deps.settings.max_context_chars,
    )
    strong_match = deps.is_strong_knowledge_match(
        result.top_score,
        result.query_coverage,
    )
    if not result.context or not strong_match:
        if deps.settings.llm_fallback_enabled and allow_fallback:
            answer = deps.ask_fallback_llm(
                base_url=deps.settings.llm_base_url,
                api_key=deps.settings.llm_api_key,
                model=deps.settings.llm_model,
                question=llm_question,
                context=tuple(chat_context[-8:]),
                memory_context=tuple(memory_context),
                self_history_context=tuple(self_history_context),
                semantic_context=semantic_context,
                candidate_knowledge_context=result.context,
                timeout=timeout or getattr(deps.settings, "knowledge_generation_timeout_seconds", 10),
            )
            if unsupported_fallback_precise_facts(answer, result.context):
                return "这个具体数值我没有可靠依据，不能给你拍一个。"
            return finalize_model_answer(deps, answer)
        return "这个我库里暂时没有准确信息。你可以换个更具体的问法，或者问一下小队长和管理员；涉及服务器规则的话，还是以本服公告为准。"

    answer = deps.ask_llm(
        base_url=deps.settings.llm_base_url,
        api_key=deps.settings.llm_api_key,
        model=deps.settings.llm_model,
        question=llm_question,
        context=result.context,
        chat_context=tuple(chat_context[-8:]),
        memory_context=tuple(memory_context),
        self_history_context=tuple(self_history_context),
        semantic_context=semantic_context,
        timeout=timeout or getattr(deps.settings, "knowledge_generation_timeout_seconds", 10),
    )
    # Fact guard only for weak matches; strong matches trust the model to
    # synthesize precise values from the grounded knowledge context.
    if not strong_match and unsupported_fallback_precise_facts(answer, result.context):
        return "这个具体数值我没有可靠依据，不能给你拍一个。"
    return finalize_model_answer(deps, answer)


def fallback_grounding_issue(deps, decision: ProcessingDecision, answer: str) -> str:
    if decision.reply_mode != "fallback":
        return ""
    candidate_context = (
        decision.knowledge_result.context
        if decision.knowledge_result is not None
        else ""
    )
    unsupported = unsupported_fallback_precise_facts(answer, candidate_context)
    if not unsupported:
        return ""
    rendered = ", ".join(f"{value}{unit}" for value, unit in sorted(unsupported))
    return f"unsupported precise fact in fallback reply: {rendered}"


def deterministic_review_failure_answer(
    deps,
    decision: ProcessingDecision,
    candidate: str,
    *,
    mentioned: bool,
    context_changed: bool = False,
) -> str:
    explicitly_addressed = bool(
        mentioned
        or decision.participation_role == "addressed"
        or decision.semantic_audience == "bot"
    )
    if not explicitly_addressed:
        return ""
    if decision.semantic_intent == "control_attempt":
        return "这类操作不能通过普通聊天执行。"
    if (
        decision.reply_mode == "knowledge"
        and candidate
        and not context_changed
        and not decision.risk_flags
        and deps.is_strong_knowledge_match(
            decision.retrieval_score,
            decision.retrieval_coverage,
        )
    ):
        knowledge_context = (
            decision.knowledge_result.context
            if decision.knowledge_result is not None
            else ""
        )
        if not unsupported_fallback_precise_facts(candidate, knowledge_context):
            return finalize_model_answer(deps, candidate)
    if decision.semantic_intent == "knowledge":
        return "这个具体问题我暂时没有可靠信息，不能确定。"
    if decision.planner_status in {"unavailable", "circuit_open", "low_confidence"}:
        return "这次我没判断清楚你在问什么，换个说法再问我一次。"
    return ""


def answer_for_decision(
    deps,
    question: str,
    decision: ProcessingDecision,
    generation_question: str,
    *,
    admin: bool = False,
    timeout: int | None = None,
) -> str:
    llm_question = generation_question or decision.effective_question or question
    semantic_context = deps.semantic_context_for_decision(decision)
    if decision.reply_mode == "control_boundary":
        return "这类操作不能通过普通聊天执行。"
    if decision.reply_mode == "bot_meta":
        return answer_bot_meta(deps, decision.capability, admin=admin)
    if decision.draft_reply and decision.reply_mode in {"fallback", "chat"}:
        return finalize_model_answer(
            deps,
            decision.draft_reply,
            unsolicited=decision.reply_mode == "chat",
        )
    if decision.reply_mode == "fallback":
        candidate_knowledge_context = (
            decision.knowledge_result.context
            if decision.knowledge_result is not None
            else ""
        )
        answer = deps.ask_fallback_llm(
            base_url=deps.settings.llm_base_url,
            api_key=deps.settings.llm_api_key,
            model=deps.settings.llm_model,
            question=llm_question,
            context=decision.chat_context,
            memory_context=decision.memory_context,
            self_history_context=decision.self_history_context,
            semantic_context=semantic_context,
            candidate_knowledge_context=candidate_knowledge_context,
            timeout=timeout or getattr(deps.settings, "knowledge_generation_timeout_seconds", 10),
        )
        if fallback_grounding_issue(deps, decision, answer):
            return "这个具体数值我没有可靠依据，不能给你拍一个。"
        return finalize_model_answer(deps, answer)
    if decision.reply_mode == "chat":
        answer = deps.answer_chat(
            base_url=deps.settings.llm_base_url,
            api_key=deps.settings.llm_api_key,
            model=deps.settings.chat_model,
            message=question,
            context=decision.chat_context,
            memory_context=decision.memory_context,
            self_history_context=decision.self_history_context,
            semantic_context=semantic_context,
            timeout=timeout or getattr(deps.settings, "chat_generation_timeout_seconds", 7),
        )
        return finalize_model_answer(deps, answer, unsolicited=True)
    return answer_question(
        deps,
        question,
        llm_question,
        retrieval_question=decision.effective_question or question,
        allow_fallback=False,
        chat_context=decision.chat_context,
        memory_context=decision.memory_context,
        self_history_context=decision.self_history_context,
        semantic_context=semantic_context,
        knowledge_result=(
            decision.knowledge_result
            if decision.knowledge_query == (decision.effective_question or question)
            else None
        ),
        timeout=timeout,
    )
