import unittest
from pathlib import Path

from squad_bot.knowledge import KnowledgeBase
from squad_bot.llm import SYSTEM_PROMPT


class KnowledgeTermTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge_base = KnowledgeBase("knowledge")

    def test_english_rally_query_finds_plain_language_term(self) -> None:
        matches = self.knowledge_base.search("Rally 是什么")

        self.assertTrue(matches)
        self.assertEqual(matches[0][0].title, "队包是什么")
        self.assertIn("队包", matches[0][0].text)

    def test_st_teamspeak_address_queries_find_server_address(self) -> None:
        for query in ("ts网址是什么", "ts地址", "ST战队ts地址", "语音地址"):
            with self.subTest(query=query):
                result = self.knowledge_base.build_context_with_metrics(query, 1200)
                self.assertGreaterEqual(result.query_coverage, 0.6)
                self.assertIn("GPFWD.ts5.plus", result.context)
                self.assertEqual(
                    result.sources[0],
                    "05-语音软件TeamSpeak教程.md / ST 战队 TS 地址是什么",
                )

    def test_teamspeak_download_queries_find_group_files(self) -> None:
        for query in ("TS3去哪下载", "teamspeak安装包在哪", "TS汉化包在哪"):
            with self.subTest(query=query):
                result = self.knowledge_base.build_context_with_metrics(query, 1200)
                self.assertIn("本体安装包和汉化包都在 QQ 群文件", result.context)
                self.assertTrue(
                    any("安装包和汉化包在哪里下载" in source for source in result.sources)
                )

    def test_answer_prompt_requires_plain_language_term(self) -> None:
        self.assertIn("统一说“队包”", SYSTEM_PROMPT)
        self.assertIn("不要直接输出英文 Rally", SYSTEM_PROMPT)

    def test_retrieval_metrics_separate_strong_and_weak_matches(self) -> None:
        strong = self.knowledge_base.build_context_with_metrics("医疗兵怎么玩", 1200)
        weak = self.knowledge_base.build_context_with_metrics("这个新武器要怎么玩", 1200)

        self.assertGreaterEqual(strong.top_score, 0.18)
        self.assertGreaterEqual(strong.query_coverage, 0.6)
        self.assertTrue(weak.context)
        self.assertLess(weak.query_coverage, 0.6)

    def test_multi_topic_question_keeps_strong_coverage(self) -> None:
        result = self.knowledge_base.build_context_with_metrics(
            "医疗要咋玩，还有榴弹要咋玩",
            1200,
        )

        self.assertGreaterEqual(result.top_score, 0.18)
        self.assertGreaterEqual(result.query_coverage, 0.6)
        self.assertTrue(any("医疗" in source for source in result.sources))
        self.assertTrue(any("榴弹" in source for source in result.sources))

    def test_fob_and_hab_queries_find_correct_explanations(self) -> None:
        expected_titles = {
            "FOB和HAB有什么区别": {"FOB 是什么", "FOB、电台、兵站、HAB 是什么关系"},
            "FOB是不是兵站": {"FOB 是什么", "FOB、电台、兵站、HAB 是什么关系"},
            "电台有什么用": {"FOB 是什么", "FOB、电台、兵站、HAB 是什么关系"},
            "卡FOB圈是什么意思": {"卡 FOB 圈是什么意思"},
            "兵站为什么不能复活": {"HAB 兵站为什么不能复活"},
        }

        for query, titles in expected_titles.items():
            with self.subTest(query=query):
                matches = self.knowledge_base.search(query)
                self.assertTrue(matches)
                self.assertIn(matches[0][0].title, titles)

    def test_fob_hab_facts_remain_distinct(self) -> None:
        fob = self.knowledge_base.search("电台有什么用")[0][0].text
        overrun = self.knowledge_base.search("兵站为什么不能复活")[0][0].text

        self.assertTrue(
            "电台本身不提供出生" in fob or "FOB/电台本身不是出生建筑" in fob
        )
        self.assertIn("至少 2 名敌人", overrun)
        self.assertIn("敌人越多", overrun)

    def test_knowledge_has_no_known_fob_hab_conflations(self) -> None:
        knowledge = "\n".join(
            path.read_text(encoding="utf-8") for path in Path("knowledge").glob("*.md")
        )

        self.assertNotIn("FOB/HAB", knowledge)
        self.assertNotIn("FOB/Radio", knowledge)
        self.assertNotIn("FOB：无线电据点范围", knowledge)
        self.assertNotIn("敌方 Radio 没被彻底处理前，它的排斥圈", knowledge)


if __name__ == "__main__":
    unittest.main()
