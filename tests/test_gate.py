import time

from qqbot.config import Config
from qqbot.envelope import MessageEnvelope
from qqbot.gate import gate


def _env(**kw):
    base = dict(group_id=1, user_id="u", message_id="m", event_time=int(time.time()), question="你好呀")
    base.update(kw)
    return MessageEnvelope(**base)


def test_must_classify_when_mentioned():
    c = Config()
    c.bot_qq = "b"
    assert gate(_env(mentioned=True), c) == "classify"


def test_skip_plain_chatter():
    c = Config()
    c.bot_qq = "b"
    assert gate(_env(question="哈哈哈哈"), c) == "skip"


def test_classify_with_question():
    c = Config()
    c.bot_qq = "b"
    assert gate(_env(question="医疗兵怎么玩"), c) == "classify"


def test_explicit_command():
    c = Config()
    c.bot_qq = "b"
    assert gate(_env(explicit_knowledge_command=True), c) == "classify"


def test_reply_to_bot():
    c = Config()
    c.bot_qq = "b"
    assert gate(_env(reply_target_user_id="b"), c) == "classify"
