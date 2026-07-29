from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .knowledge import ContextResult
from .models import ProcessingDecision


@dataclass(frozen=True)
class KnowledgeSelection:
    query: str
    result: ContextResult
    strong_match: bool


class KnowledgeRoutingService:
    def __init__(
        self,
        retrieve: Callable[[str], ContextResult],
        strong_match: Callable[[float, float], bool],
    ) -> None:
        self._retrieve = retrieve
        self._strong_match = strong_match
        self._results: dict[str, ContextResult] = {}

    def lookup(self, query: str) -> KnowledgeSelection:
        if query not in self._results:
            self._results[query] = self._retrieve(query)
        result = self._results[query]
        return KnowledgeSelection(
            query=query,
            result=result,
            strong_match=self._strong_match(
                result.top_score,
                result.query_coverage,
            ),
        )

    def contextual_candidate(
        self,
        current: KnowledgeSelection,
        rewrite: tuple[str, float] | None,
        *,
        min_confidence: float,
    ) -> KnowledgeSelection:
        if not rewrite:
            return current
        standalone_question, confidence = rewrite
        if confidence < min_confidence or standalone_question == current.query:
            return current
        candidate = self.lookup(standalone_question)
        candidate_is_better = (
            candidate.result.query_coverage,
            candidate.result.top_score,
        ) > (
            current.result.query_coverage,
            current.result.top_score,
        )
        if candidate.result.context and (candidate.strong_match or candidate_is_better):
            return candidate
        return current


def attach_result(
    decision: ProcessingDecision,
    query: str,
    result: ContextResult,
) -> ProcessingDecision:
    decision.knowledge_query = query
    decision.knowledge_result = result
    return decision
