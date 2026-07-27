import json
import urllib.request
import re


AT_RE_TEMPLATE = r"\[CQ:at,qq={qq}\]"
AT_RE = re.compile(r"\[CQ:at,qq=\d+\]")
REPLY_RE = re.compile(r"\[CQ:reply,(?:[^,\]]+,)*id=([^,\]]+)(?:,[^\]]*)?\]")


def extract_plain_text(message) -> str:
    if isinstance(message, str):
        return REPLY_RE.sub("", AT_RE.sub("", message)).strip()
    if not isinstance(message, list):
        return ""

    parts: list[str] = []
    for segment in message:
        if segment.get("type") == "text":
            parts.append(segment.get("data", {}).get("text", ""))
    return "".join(parts).strip()


def extract_context_text(message) -> str:
    """Keep lightweight media placeholders so non-text turns remain in context."""
    if not isinstance(message, list):
        return extract_plain_text(message)
    labels = {
        "image": "[图片]",
        "face": "[表情]",
        "mface": "[表情]",
        "video": "[视频]",
        "record": "[语音]",
        "file": "[文件]",
        "json": "[卡片消息]",
    }
    parts: list[str] = []
    for segment in message:
        if not isinstance(segment, dict):
            continue
        kind = str(segment.get("type") or "")
        if kind == "text":
            parts.append(str(segment.get("data", {}).get("text", "")))
        elif kind in labels:
            parts.append(labels[kind])
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def extract_reply_message_id(message) -> str:
    if isinstance(message, list):
        for segment in message:
            if not isinstance(segment, dict) or segment.get("type") != "reply":
                continue
            message_id = segment.get("data", {}).get("id", "")
            if message_id is not None and str(message_id).strip():
                return str(message_id).strip()
        return ""

    if isinstance(message, str):
        match = REPLY_RE.search(message)
        if match:
            return match.group(1).strip()

    return ""


def is_mentioned(bot_qq: str, raw_message) -> bool:
    if not bot_qq:
        return False

    if isinstance(raw_message, list):
        return any(
            seg.get("type") == "at" and str(seg.get("data", {}).get("qq", "")) == bot_qq
            for seg in raw_message
        )

    if isinstance(raw_message, str):
        at_pattern = AT_RE_TEMPLATE.format(qq=re.escape(bot_qq))
        return bool(re.search(at_pattern, raw_message))

    return False


def has_other_mention(bot_qq: str, raw_message) -> bool:
    return bool(extract_mentioned_user_ids(bot_qq, raw_message))


def extract_mentioned_user_ids(bot_qq: str, raw_message) -> tuple[str, ...]:
    if isinstance(raw_message, list):
        candidates = [
            str(segment.get("data", {}).get("qq", "")).strip()
            for segment in raw_message
            if isinstance(segment, dict) and segment.get("type") == "at"
        ]
    elif isinstance(raw_message, str):
        candidates = re.findall(r"\[CQ:at,qq=(\d+)\]", raw_message)
    else:
        candidates = []

    result: list[str] = []
    for candidate in candidates:
        if not re.fullmatch(r"\d+", candidate) or candidate == bot_qq:
            continue
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def should_respond(text: str, prefix: str, bot_qq: str, raw_message) -> tuple[bool, str]:
    stripped = text.strip()
    if prefix and stripped.startswith(prefix):
        stripped = stripped[len(prefix) :].strip()
    return bool(stripped), stripped


def send_group_msg(
    api_url: str,
    group_id: int,
    message: str,
    access_token: str = "",
    *,
    mention_user_id: str = "",
    reply_to_message_id: str = "",
) -> str:
    mention_id = str(mention_user_id or "").strip()
    reply_id = str(reply_to_message_id or "").strip()
    message_payload: str | list[dict]
    if reply_id:
        message_payload = [
            {"type": "reply", "data": {"id": reply_id}},
            {"type": "text", "data": {"text": message}},
        ]
    elif re.fullmatch(r"\d+", mention_id):
        message_payload = [
            {"type": "at", "data": {"qq": mention_id}},
            {"type": "text", "data": {"text": " " + message}},
        ]
    else:
        message_payload = message
    payload = {"group_id": group_id, "message": message_payload}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    request = urllib.request.Request(
        api_url.rstrip("/") + "/send_group_msg",
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        raw_response = response.read()
    try:
        result = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, TypeError):
        return ""
    response_data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(response_data, dict):
        return ""
    message_id = response_data.get("message_id", "")
    return str(message_id).strip() if message_id is not None else ""


def get_message_sender_id(
    api_url: str,
    message_id: str,
    access_token: str = "",
    timeout: float = 3,
) -> str:
    sender_id, _ = get_message_info(
        api_url,
        message_id,
        access_token,
        timeout,
    )
    return sender_id


def get_message_info(
    api_url: str,
    message_id: str,
    access_token: str = "",
    timeout: float = 3,
) -> tuple[str, str]:
    if not message_id:
        return "", ""

    message_id_value = int(message_id) if re.fullmatch(r"-?\d+", message_id) else message_id
    payload = {"message_id": message_id_value}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    request = urllib.request.Request(
        api_url.rstrip("/") + "/get_msg",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, TypeError):
        return "", ""

    response_data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(response_data, dict):
        return "", ""
    sender_id = ""
    sender = response_data.get("sender")
    if isinstance(sender, dict) and sender.get("user_id") is not None:
        sender_id = str(sender["user_id"])
    elif response_data.get("user_id") is not None:
        sender_id = str(response_data["user_id"])

    quoted_message = response_data.get("message", response_data.get("raw_message", ""))
    return sender_id, extract_plain_text(quoted_message)
