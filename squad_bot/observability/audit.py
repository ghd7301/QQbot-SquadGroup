from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from ..llm import SemanticTopicCandidate, SubjectCandidate


def _rotate_log_if_needed(
    log_path: Path, max_bytes: int = 5 * 1024 * 1024, keep: int = 5
) -> None:
    """Rotate log file if it exceeds max_bytes. Keeps the last `keep` rotated files."""
    try:
        if not log_path.exists() or log_path.stat().st_size < max_bytes:
            return
        for i in range(keep, 0, -1):
            src = log_path.with_suffix(f"{log_path.suffix}.{i}")
            if i >= keep:
                if src.exists():
                    src.unlink()
            else:
                dst = log_path.with_suffix(f"{log_path.suffix}.{i + 1}")
                if src.exists():
                    src.rename(dst)
        rotated = log_path.with_suffix(f"{log_path.suffix}.1")
        log_path.rename(rotated)
        print(f"Rotated audit log: {log_path} -> {rotated}")
    except Exception as exc:
        print("Audit log rotation failed:", repr(exc))


def write_message_audit(
    deps,
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
    event_time=None,
) -> None:
    try:
        total_latency_ms = max(0, int((time.time() - float(event_time)) * 1000))
    except (TypeError, ValueError):
        total_latency_ms = 0
    try:
        scene_payload = json.loads(scene_context) if scene_context else {}
    except (json.JSONDecodeError, TypeError):
        scene_payload = {}
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event_time": event_time,
        "group_id": str(group_id) if group_id is not None else "",
        "user_id": str(user_id) if user_id is not None else "",
        "question": question,
        "mentioned": bool(mentioned),
        "decision": decision,
        "reason": reason,
        "has_context": bool(has_context),
        "sources": list(sources),
        "followup_of": followup_of,
        "followup_scope": followup_scope,
        "reply_mode": reply_mode,
        "retrieval_score": round(float(retrieval_score), 4),
        "retrieval_coverage": round(float(retrieval_coverage), 4),
        "knowledge_strength": (
            "strong"
            if float(retrieval_score)
            >= float(getattr(deps.settings, "knowledge_strong_min_score", 0.18))
            and float(retrieval_coverage)
            >= float(getattr(deps.settings, "knowledge_strong_min_coverage", 0.6))
            else "weak" if sources or retrieval_score or retrieval_coverage else "none"
        ),
        "model_latency_ms": int(model_latency_ms),
        "total_latency_ms": total_latency_ms,
        "model": model_name,
        "reply_message_id": str(reply_message_id),
        "reply_target_user_id": str(reply_target_user_id),
        "chat_context": list(chat_context),
        "scene_context": scene_context,
        "answer": answer,
        "mention_user_id": str(mention_user_id or ""),
        "semantic_intent": semantic_intent,
        "semantic_topic": semantic_topic,
        "implicit_meaning": implicit_meaning,
        "capability": capability,
        "semantic_confidence": round(float(semantic_confidence), 4),
        "topic_candidates": [
            {
                "key": candidate.key,
                "label": candidate.label,
                "confidence": round(candidate.confidence, 4),
                "basis": candidate.basis,
                "anchor_message_ids": list(candidate.anchor_message_ids),
            }
            for candidate in topic_candidates
        ],
        "subject_candidates": [
            {
                "entity_type": candidate.entity_type,
                "entity_id": candidate.entity_id,
                "label": candidate.label,
                "confidence": round(candidate.confidence, 4),
                "evidence_message_ids": list(candidate.evidence_message_ids),
            }
            for candidate in subject_candidates
        ],
        "subject_ambiguity": subject_ambiguity,
        "bot_involvement": bot_involvement,
        "reply_perspective": reply_perspective,
        "semantic_audience": semantic_audience,
        "participation_role": participation_role,
        "plan_context_revision": int(plan_context_revision),
        "plan_scene_version": int(plan_scene_version),
        "related_message_ids": list(related_message_ids),
        "semantic_replan_count": int(semantic_replan_count),
        "semantic_replan_reason": semantic_replan_reason,
        "planner_status": planner_status,
        "planner_lane": (
            "addressed"
            if mentioned
            or (
                bool(str(getattr(deps.settings, "bot_qq", "") or ""))
                and str(reply_target_user_id or "")
                == str(getattr(deps.settings, "bot_qq", "") or "")
            )
            else "unsolicited"
        ),
        "planner_circuit_open": planner_status == "circuit_open",
        "planner_latency_ms": int(planner_latency_ms),
        "scene_version": (
            int(scene_payload.get("version") or 0)
            if isinstance(scene_payload, dict)
            else 0
        ),
        "scene_updated_through_sequence": (
            int(scene_payload.get("updated_through_sequence") or 0)
            if isinstance(scene_payload, dict)
            else 0
        ),
        "memory_query": memory_query,
        "memory_hit_count": int(memory_hit_count),
        "memory_retrieval_attempted": bool(memory_retrieval_attempted),
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_candidate_count": int(memory_candidate_count),
        "memory_rejection_reason": memory_rejection_reason,
        "recent_context_candidate_count": int(recent_context_candidate_count),
        "recent_context_selected_count": int(recent_context_selected_count),
        "recent_context_chars": int(recent_context_chars),
        "memory_context_chars": int(memory_context_chars),
        "context_deduplicated_count": int(context_deduplicated_count),
        "recent_context_selected_ids": list(recent_context_selected_ids),
        "memory_selected_chunk_ids": [
            int(value) for value in memory_selected_chunk_ids
        ],
        "memory_selected_by_planner": bool(memory_selected_by_planner),
        "self_history_candidate_count": int(self_history_candidate_count),
        "self_history_selected_count": int(self_history_selected_count),
        "self_history_chars": int(self_history_chars),
        "self_history_selected_message_ids": list(self_history_selected_message_ids),
        "self_history_reasons": list(self_history_reasons),
        "bot_message_id": str(bot_message_id or ""),
        "generated_for_message_ids": list(generated_for_message_ids),
        "turn_id": str(turn_id or ""),
    }
    log_path = Path(deps.settings.message_audit_log)
    try:
        with deps.audit_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_log_if_needed(log_path, max_bytes=5 * 1024 * 1024, keep=5)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print("Audit log write failed:", repr(exc))


def recent_audit_entries(deps, limit: int = 5) -> list[dict]:
    log_path = Path(deps.settings.message_audit_log)
    if not log_path.exists():
        return []
    entries: list[dict] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("decision") not in {"skipped", "ignored"}:
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def record_knowledge_gap(deps, query: str, result) -> bool:
    normalized = " ".join(str(query or "").strip().split())
    if len(normalized) < 2:
        return False
    safe_query = deps.redact_for_model(normalized)[:300]
    now = time.time()
    dedupe_seconds = max(
        0, getattr(deps.settings, "knowledge_gap_dedupe_seconds", 3600)
    )
    with deps.knowledge_gap_lock:
        previous = deps.recent_knowledge_gap_queries.get(safe_query, 0)
        if now - previous < dedupe_seconds:
            return False
        deps.recent_knowledge_gap_queries[safe_query] = now
        cutoff = now - max(dedupe_seconds, 86400)
        for key, timestamp in tuple(deps.recent_knowledge_gap_queries.items()):
            if timestamp < cutoff:
                deps.recent_knowledge_gap_queries.pop(key, None)
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "query": safe_query,
            "top_score": round(float(result.top_score), 4),
            "coverage": round(float(result.query_coverage), 4),
            "sources": list(result.sources),
            "matched_tokens": list(result.matched_query_tokens),
            "missing_tokens": list(result.missing_query_tokens),
        }
        path = Path(
            getattr(deps.settings, "knowledge_gap_log", "work/knowledge_gaps.jsonl")
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as exc:
            print("Knowledge gap log write failed:", repr(exc))
            return False


def recent_knowledge_gap_entries(deps, limit: int = 5) -> list[dict]:
    path = Path(
        getattr(deps.settings, "knowledge_gap_log", "work/knowledge_gaps.jsonl")
    )
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
        if len(entries) >= limit:
            break
    return entries
