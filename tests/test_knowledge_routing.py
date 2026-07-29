import unittest

from squad_bot.knowledge import ContextResult
from squad_bot.knowledge_routing import KnowledgeRoutingService


class KnowledgeRoutingServiceTests(unittest.TestCase):
    def test_lookup_caches_each_query(self) -> None:
        calls = []

        def retrieve(query: str) -> ContextResult:
            calls.append(query)
            return ContextResult(query, ("source",), 0.5, 0.8)

        service = KnowledgeRoutingService(
            retrieve,
            lambda score, coverage: score >= 0.3 and coverage >= 0.6,
        )

        first = service.lookup("队包多久一轮")
        second = service.lookup("队包多久一轮")

        self.assertIs(first.result, second.result)
        self.assertTrue(first.strong_match)
        self.assertEqual(calls, ["队包多久一轮"])

    def test_contextual_candidate_requires_confidence_and_improvement(self) -> None:
        results = {
            "那个地址是多少": ContextResult("弱资料", ("weak",), 0.1, 0.1),
            "ST 战队 TS 地址是多少": ContextResult(
                "地址资料",
                ("strong",),
                0.5,
                1.0,
            ),
        }
        service = KnowledgeRoutingService(
            results.__getitem__,
            lambda score, coverage: score >= 0.3 and coverage >= 0.6,
        )
        initial = service.lookup("那个地址是多少")

        low_confidence = service.contextual_candidate(
            initial,
            ("ST 战队 TS 地址是多少", 0.4),
            min_confidence=0.75,
        )
        improved = service.contextual_candidate(
            initial,
            ("ST 战队 TS 地址是多少", 0.9),
            min_confidence=0.75,
        )

        self.assertIs(low_confidence, initial)
        self.assertEqual(improved.query, "ST 战队 TS 地址是多少")
        self.assertTrue(improved.strong_match)


if __name__ == "__main__":
    unittest.main()
