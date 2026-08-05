import time

from qqbot.config import Config
from qqbot.envelope import MessageEnvelope
from qqbot.knowledge import Chunk
from qqbot.safety import UNCERTAIN_REPLY, check


def _chunk(text):
    return Chunk(text=text, source="a.md", title="t", section_path=["t"])


def _env(**kw):
    base = dict(group_id=1, user_id="12345", message_id="m", event_time=1, question="q")
    base.update(kw)
    return MessageEnvelope(**base)


def test_numeric_grounding_fails():
    c = Config()
    env = _env()
    ok, reason, reply = check(
        env,
        "医疗兵有 99 种玩法",
        reply_type="knowledge",
        chunks=[_chunk("医疗兵有 3 种玩法")],
        config=c,
    )
    assert reason == "numeric_grounding"
    assert reply == UNCERTAIN_REPLY


def test_numeric_grounding_passes():
    c = Config()
    env = _env()
    ok, reason, reply = check(
        env,
        "医疗兵有 3 种玩法",
        reply_type="knowledge",
        chunks=[_chunk("医疗兵有 3 种玩法")],
        config=c,
    )
    assert ok is True
    assert reason is None


def test_dedup():
    c = Config()
    env = _env()
    ok, reason, _ = check(
        env, "哈哈", reply_type="chat", config=c, recent_replies=[(time.time(), "哈哈")]
    )
    assert ok is False and reason == "duplicate"


def test_identity_protection():
    c = Config()
    env = _env(user_id="12345")
    ok, reason, reply = check(env, "用户 12345 好", reply_type="chat", config=c)
    assert "12345" not in reply
