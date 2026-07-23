import unittest

from squad_bot.server import is_identity_question


class IdentityQuestionTests(unittest.TestCase):
    def test_common_identity_question_variants(self) -> None:
        questions = (
            "你是谁",
            "你能干嘛",
            "你能干什么",
            "你可以干什么",
            "你会干什么",
            "你能做什么",
            "你可以做什么",
            "你有什么用",
            "你是干什么的",
            "你是做什么的",
            "介绍一下你自己",
        )

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_identity_question(question))

    def test_content_questions_are_not_identity_questions(self) -> None:
        questions = (
            "介绍一下每个兵种，干啥的，大概怎么玩",
            "介绍一下医疗兵",
            "HAB能做什么",
            "小队长是干什么的",
        )

        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(is_identity_question(question))


if __name__ == "__main__":
    unittest.main()
