import io
import json
import unittest
from unittest.mock import patch

from squad_bot.onebot import (
    extract_mentioned_user_ids,
    extract_plain_text,
    extract_reply_message_id,
    get_message_info,
    get_message_sender_id,
    send_group_msg,
)
from squad_bot.server import classify_reply_target


class FakeResponse:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body.read()


class ReplyDetectionTests(unittest.TestCase):
    def test_extracts_other_mentions_without_bot_or_all(self):
        segments = [
            {"type": "at", "data": {"qq": "3119065126"}},
            {"type": "at", "data": {"qq": "10001"}},
            {"type": "at", "data": {"qq": "all"}},
            {"type": "at", "data": {"qq": "10001"}},
        ]
        self.assertEqual(
            extract_mentioned_user_ids("3119065126", segments),
            ("10001",),
        )
        self.assertEqual(
            extract_mentioned_user_ids(
                "3119065126",
                "[CQ:at,qq=3119065126] [CQ:at,qq=10002]生日快乐",
            ),
            ("10002",),
        )

    @patch("squad_bot.onebot.urllib.request.urlopen")
    def test_send_group_message_uses_structured_mention(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"status": "ok", "data": {"message_id": 456789}}
        )

        message_id = send_group_msg(
            "http://127.0.0.1:3000",
            123,
            "生日快乐！",
            "token",
            mention_user_id="10001",
        )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["group_id"], 123)
        self.assertEqual(
            payload["message"],
            [
                {"type": "at", "data": {"qq": "10001"}},
                {"type": "text", "data": {"text": " 生日快乐！"}},
            ],
        )
        self.assertEqual(message_id, "456789")

    @patch("squad_bot.onebot.urllib.request.urlopen")
    def test_send_group_message_prefers_native_reply_target(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"status": "ok", "data": {"message_id": 456790}}
        )

        send_group_msg(
            "http://127.0.0.1:3000",
            123,
            "回答内容",
            "token",
            mention_user_id="10001",
            reply_to_message_id="9988",
        )

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(
            payload["message"],
            [
                {"type": "reply", "data": {"id": "9988"}},
                {"type": "text", "data": {"text": "回答内容"}},
            ],
        )

    def test_extracts_reply_id_from_segments(self):
        message = [
            {"type": "reply", "data": {"id": 123456}},
            {"type": "text", "data": {"text": "然后呢"}},
        ]
        self.assertEqual(extract_reply_message_id(message), "123456")

    def test_extracts_reply_id_from_cq_string(self):
        self.assertEqual(
            extract_reply_message_id("[CQ:reply,id=-98765]然后呢"),
            "-98765",
        )
        self.assertEqual(extract_plain_text("[CQ:reply,id=-98765]然后呢"), "然后呢")

    def test_no_reply_returns_empty_string(self):
        self.assertEqual(extract_reply_message_id("普通消息"), "")

    @patch("squad_bot.onebot.urllib.request.urlopen")
    def test_get_message_sender_id_reads_nested_sender(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"status": "ok", "data": {"sender": {"user_id": 3119065126}}}
        )

        sender_id = get_message_sender_id("http://127.0.0.1:3000", "123", "token")

        self.assertEqual(sender_id, "3119065126")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:3000/get_msg")
        self.assertEqual(json.loads(request.data), {"message_id": 123})
        self.assertEqual(request.headers["Authorization"], "Bearer token")

    @patch("squad_bot.onebot.urllib.request.urlopen")
    def test_get_message_info_includes_quoted_text(self, urlopen):
        urlopen.return_value = FakeResponse(
            {
                "status": "ok",
                "data": {
                    "sender": {"user_id": 10001},
                    "message": [{"type": "text", "data": {"text": "被引用的原话"}}],
                },
            }
        )

        self.assertEqual(
            get_message_info("http://127.0.0.1:3000", "123"),
            ("10001", "被引用的原话"),
        )

    @patch("squad_bot.onebot.urllib.request.urlopen", side_effect=OSError("offline"))
    def test_get_message_sender_id_returns_empty_on_api_failure(self, _urlopen):
        self.assertEqual(get_message_sender_id("http://127.0.0.1:3000", "123"), "")

    @patch("squad_bot.onebot.urllib.request.urlopen")
    def test_get_message_sender_id_returns_empty_on_malformed_response(self, urlopen):
        urlopen.return_value = FakeResponse({"status": "ok", "data": None})
        self.assertEqual(get_message_sender_id("http://127.0.0.1:3000", "123"), "")

    def test_reply_to_bot_becomes_direct_question(self):
        self.assertEqual(
            classify_reply_target("123", "3119065126", False, "3119065126"),
            (True, True, "reply to bot"),
        )

    def test_reply_to_another_member_is_ignored(self):
        self.assertEqual(
            classify_reply_target("123", "10001", False, "3119065126"),
            (False, False, "reply directed at another member"),
        )

    def test_explicit_mention_overrides_reply_target(self):
        self.assertEqual(
            classify_reply_target("123", "", True, "3119065126"),
            (True, True, "explicit mention"),
        )

    def test_unknown_reply_target_is_ignored(self):
        self.assertEqual(
            classify_reply_target("123", "", False, "3119065126"),
            (False, False, "reply target unknown"),
        )

    def test_non_reply_keeps_existing_mention_state(self):
        self.assertEqual(
            classify_reply_target("", "", False, "3119065126"),
            (True, False, ""),
        )


if __name__ == "__main__":
    unittest.main()
