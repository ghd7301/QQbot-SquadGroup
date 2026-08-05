import asyncio
import time

from qqbot.config import Config
from qqbot.envelope import MessageEnvelope
from qqbot.prefilter import should_skip


def _env(**kw):
    base = dict(group_id=1, user_id="u", message_id="m", event_time=int(time.time()), question="你好呀")
    base.update(kw)
    return MessageEnvelope(**base)


def test_skip_non_whitelisted():
    c = Config()
    c.whitelisted_groups = [1]
    assert should_skip(_env(group_id=2), c) is True
    assert should_skip(_env(group_id=1), c) is False


def test_skip_bot_self():
    c = Config()
    c.whitelisted_groups = [1]
    c.bot_qq = "u"
    assert should_skip(_env(user_id="u"), c) is True


def test_skip_short_no_mention():
    c = Config()
    c.whitelisted_groups = [1]
    assert should_skip(_env(question="哈"), c) is True
    assert should_skip(_env(question="哈", mentioned=True), c) is False


def test_skip_expired():
    c = Config()
    c.whitelisted_groups = [1]
    assert should_skip(_env(event_time=1), c, now=999999) is True


def test_skip_recalled():
    c = Config()
    c.whitelisted_groups = [1]
    assert should_skip(_env(is_recalled=True), c) is True
