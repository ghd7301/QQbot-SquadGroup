from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Sequence

from .models import (
    ConversationState,
    FollowupMatch,
    ProcessingDecision,
)


# ---------------------------------------------------------------------------
# Constants – session continuity
# ---------------------------------------------------------------------------

SEMANTIC_CHAT_INTENTS = {
    "chat",
    "normal_chat",
    "banter_at_bot",
    "third_party_attack",
    "genuine_criticism",
    "hostile_abuse",
}
LOCAL_REPLY_MODES = {"bot_meta", "identity", "control_boundary"}

IDENTITY_KEYWORDS = (
    "你是谁",
    "你是啥",
    "你是什么",
    "你干嘛",
    "你能干嘛",
    "你能干什么",
    "你可以干什么",
    "你会什么",
    "你会干什么",
    "你能做什么",
    "你可以做什么",
    "你有什么用",
    "你是干什么的",
    "你是做什么的",
    "自我介绍",
    "help",
    "帮助",
    "怎么用",
)

SELF_REFERENCE_KEYWORDS = (
    "你",
    "自己",
    "机器人",
    "bot",
    "qqbot",
    "猫",
    "教官",
)

# ---------------------------------------------------------------------------
# Constants – group chat participation strategy
# ---------------------------------------------------------------------------

AUTO_REPLY_KEYWORDS = (
    "raas",
    "aas",
    "tc",
    "invasion",
    "skirmish",
    "seed",
    "hab",
    "fob",
    "rally",
    "radio",
    "logi",
    "op",
    "tb",
    "tk",
    "zcc",
    "pjp",
    "tow",
    "ied",
    "lat",
    "hat",
    "ts",
    "ts3",
    "b键",
    "v键",
    "g键",
    "m键",
    "小蓝熊",
    "反作弊",
    "虚幻",
    "崩",
    "闪退",
    "搜不到",
    "服务器",
    "卡三点",
    "跑小人",
    "无法连接",
    "断开连接",
    "队列",
    "排队",
    "武器",
    "医疗兵",
    "医生",
    "反坦",
    "轻反",
    "重反",
    "小队长",
    "指挥官",
    "后勤",
    "补给",
    "卸货",
    "卸建",
    "卸弹",
    "建设点",
    "弹药点",
    "载具",
    "装甲",
    "坦克",
    "步战",
    "轮战",
    "轮式",
    "履带",
    "断履",
    "炮塔",
    "发动机",
    "侦察车",
    "侦查车",
    "直升机",
    "报点",
    "方位",
    "票数",
    "掉票",
    "回防",
    "入侵",
    "随机攻防",
    "工兵",
    "步枪兵",
    "自动步枪兵",
    "榴弹兵",
    "机枪兵",
    "机枪手",
    "轻机枪",
    "精确射手",
    "神射手",
    "狙击手",
    "载具兵",
    "车组",
    "飞行员",
    "指挥",
    "指挥频道",
    "指挥官",
    "火力标",
    "战术支援",
    "小键盘",
    "page up",
    "page down",
    "批车",
    "要车",
    "批准",
    "拉点",
    "拉点队",
    "抢点队",
    "抢点",
    "卡点",
    "双白",
    "拉双白",
    "拉点速度",
    "白旗",
    "龟壳队",
    "支援队",
    "单载",
    "压家",
    "压家圈",
    "归属权",
    "特装",
    "特装限制",
    "灰圈",
    "蓝圈",
    "卡fob",
    "卡 fob",
    "电台",
    "无线电",
    "归零",
    "respawn",
    "自杀",
    "重部署",
    "管理员",
    "sorry",
    "暖服",
    "凉服",
    "坦克大战",
    "地图",
    "新手",
    "萌新",
    "入坑",
    "教程",
    "选服",
    "进服",
    "服务器列表",
    "筛选",
    "阴兵",
    "机器人",
    "部署",
    "复活点",
    "出生点",
    "阵营",
    "部署界面",
    "队伍列表",
    "主基地",
    "队包",
    "小队包",
    "进攻点",
    "防守点",
    "防御点",
    "兵站",
    "兵种",
    "卡装备",
    "卡兵种",
    "耳机",
    "麦克风",
    "语音",
    "体力",
    "体力条",
    "耐力",
    "带镜",
    "机瞄",
    "红点",
    "腰射",
    "架枪",
    "掩体",
    "推进",
    "窗口",
    "丛林",
    "城区",
    "补给线",
    "步战协同",
    "载具协同",
    "国服",
    "外服",
    "木塔哈",
    "mutaha",
    "纳尔瓦",
    "narva",
    "费卢杰",
    "fallujah",
    "叶城",
    "叶霍里夫卡",
    "yehorivka",
    "格罗多克",
    "gorodok",
)

QUESTION_CUES = (
    "?",
    "？",
    "怎么",
    "咋",
    "如何",
    "什么",
    "啥",
    "为啥",
    "为什么",
    "区别",
    "能不能",
    "可以吗",
    "是不是",
    "咋办",
    "怎么办",
    "求助",
    "请问",
    "问一下",
    "有没有",
    "在哪",
    "哪里",
    "多久",
    "多少",
    "谁能",
    "有无",
    "有吗",
    "能回吗",
)

HELP_CUES = (
    "进不去",
    "搜不到",
    "卡三点",
    "跑小人",
    "无法连接",
    "断开连接",
    "闪退",
    "崩溃",
    "报错",
    "没声音",
    "听不到",
    "不会",
    "不懂",
    "不知道",
)

ASSIGNMENT_CUES = (
    "介绍一下",
    "整理",
    "做一份",
    "写一份",
    "弄一份",
    "发一份",
    "出一份",
    "搞一份",
    "资料",
    "报告",
    "文档",
    "帖子",
    "教程",
    "攻略",
)

NON_BOT_TARGET_CUES = (
    "别人",
    "其他人",
    "谁",
    "谁能",
    "哪个",
    "哪位",
    "兄弟",
    "老哥",
    "管理员",
    "队长",
    "群主",
    "你们",
    "大家",
)

BROAD_CONTENT_CUES = (
    "每个",
    "各个",
    "所有",
    "全部",
    "一份",
    "详解",
    "介绍报告",
    "介绍资料",
)

META_CONTENT_CUES = (
    "知识库",
    "资料",
    "报告",
    "文档",
    "帖子",
    "教程",
    "攻略",
    "素材",
    "空缺",
    "完善",
    "更新",
)

BOT_META_CUES = (
    "做成bot",
    "做成 bot",
    "通用bot",
    "通用 bot",
    "可以艾特",
    "艾特他",
    "艾特它",
    "问他",
    "问它",
    "封装进去",
    "整理到知识库",
    "贴合st",
)

CHAT_BLOCK_CUES = (
    "http://",
    "https://",
    "www.",
    "公告",
    "通知",
    "全体成员",
    "招募",
    "报名",
    "训练安排",
    "机器人",
    "bot",
    "知识库",
    "提示词",
    "回复冷却",
    "回复限制",
    "文档",
    "攻略",
    "资料整理",
)

BARE_CHAT_REACTIONS = {
    "6",
    "66",
    "666",
    "哈哈",
    "哈哈哈",
    "哈哈哈哈",
    "hh",
    "hhh",
    "hhhh",
    "笑死",
    "笑死我了",
    "草",
    "艹",
    "神了",
    "nb",
    "牛",
    "牛逼",
    "来了",
    "来了来了",
    "嗯",
    "嗯嗯",
    "对",
    "对对",
    "对对对",
    "好",
    "好的",
    "行",
    "可以",
    "确实",
    "真的",
    "假的",
    "啊",
    "啊这",
    "哦",
    "哦哦",
    "ok",
    "OK",
    "okay",
    "1",
    "11",
    "111",
    "顶",
    "赞",
    "冲",
    "冲冲冲",
    "乐",
    "蚌",
    "绷",
    "典",
    "孝",
    "急",
    "赢",
    "麻",
    "寄",
    "润",
}

BIRTHDAY_CELEBRATION_CUES = (
    "生日快乐",
    "过生日",
    "今天生日",
    "今儿生日",
    "生日啦",
    "生日了",
    "寿星",
)

BIRTHDAY_DISCUSSION_CUES = (
    "生日礼物",
    "去年生日",
    "明年生日",
    "下次生日",
    "什么时候生日",
    "生日是哪天",
    "不是生日",
    "不过生日",
)

GRADUATION_CELEBRATION_CUES = (
    "毕业快乐",
    "毕业了",
    "毕业啦",
    "顺利毕业",
    "恭喜毕业",
)

GRADUATION_DISCUSSION_CUES = (
    "毕业论文",
    "毕业设计",
    "去年毕业",
    "明年毕业",
    "什么时候毕业",
    "不能毕业",
    "没毕业",
)


# ---------------------------------------------------------------------------
# Batch 1 – session continuity and runtime state
# ---------------------------------------------------------------------------


def topic_key(deps, question: str, decision: ProcessingDecision) -> str:
    if decision.sources:
        return decision.sources[0]
    return deps.normalize_command_text(question)[:40]


def is_topic_on_cooldown(deps, group_id: int, key: str) -> bool:
    if deps.settings.same_topic_cooldown_seconds <= 0:
        return False
    now = time.time()
    with deps.topic_cooldown_lock:
        # Size-triggered full cleanup to prevent unbounded growth
        if len(deps.recent_reply_topics) > 5000:
            deps.recent_reply_topics.clear()
        expired = [
            item_key
            for item_key, timestamp in deps.recent_reply_topics.items()
            if now - timestamp > deps.settings.same_topic_cooldown_seconds
        ]
        for item_key in expired:
            deps.recent_reply_topics.pop(item_key, None)
        last_reply = deps.recent_reply_topics.get((group_id, key))
    return last_reply is not None and now - last_reply <= deps.settings.same_topic_cooldown_seconds


def mark_topic_replied(deps, group_id: int, key: str) -> None:
    if deps.settings.same_topic_cooldown_seconds <= 0:
        return
    with deps.topic_cooldown_lock:
        deps.recent_reply_topics[(group_id, key)] = time.time()


def check_and_mark_topic_replied(deps, group_id: int, key: str) -> bool:
    """Atomically check cooldown and mark if not on cooldown. Returns True if on cooldown."""
    if deps.settings.same_topic_cooldown_seconds <= 0:
        return False
    now = time.time()
    with deps.topic_cooldown_lock:
        expired = [
            item_key
            for item_key, timestamp in deps.recent_reply_topics.items()
            if now - timestamp > deps.settings.same_topic_cooldown_seconds
        ]
        for item_key in expired:
            deps.recent_reply_topics.pop(item_key, None)
        last_reply = deps.recent_reply_topics.get((group_id, key))
        if last_reply is not None and now - last_reply <= deps.settings.same_topic_cooldown_seconds:
            return True
        deps.recent_reply_topics[(group_id, key)] = now
        return False


def followup_context_for(
    deps,
    group_id: int,
    user_id,
    question: str,
    mentioned: bool,
    *,
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    reply_text: str = "",
    bot_qq: str = "",
    db_path: str | Path | None = None,
) -> FollowupMatch | None:
    now = time.time()
    is_reply_to_bot = bool(
        reply_message_id
        and bot_qq
        and str(reply_target_user_id or "") == str(bot_qq)
    )
    if is_reply_to_bot:
        state = deps.load_conversation_turn_by_bot_message_id(
            group_id,
            reply_message_id,
            db_path=db_path,
        )
        if state:
            return FollowupMatch(state=state, scope="reply")
        if reply_text.strip():
            return FollowupMatch(
                state=ConversationState(
                    last_question="",
                    sources=(),
                    timestamp=now,
                    last_answer=reply_text.strip(),
                    reply_mode="quoted",
                    bot_message_id=reply_message_id,
                ),
                scope="reply_text",
            )
        # An explicit reply must never silently bind to an unrelated recent turn.
        return None

    # Without an explicit QQ reply, relation selection belongs to the semantic
    # planner, which sees speaker IDs, message IDs and parallel topic anchors.
    return None


def build_effective_question(question: str, followup_match: FollowupMatch | None) -> str:
    if not followup_match or not followup_match.state.last_question:
        return question
    return f"上一轮问题：{followup_match.state.last_question}\n当前追问：{question}"


def build_generation_question(question: str, followup_match: FollowupMatch | None) -> str:
    if not followup_match:
        return question
    parts: list[str] = []
    if followup_match.state.last_question:
        parts.append(f"上一轮问题：{followup_match.state.last_question}")
    if followup_match.state.last_answer:
        parts.append(f"上一轮回答：{followup_match.state.last_answer[:1200]}")
    parts.append(f"当前追问：{question}")
    return "\n".join(parts)


def remember_conversation(
    deps,
    group_id: int,
    user_id,
    question: str,
    decision: ProcessingDecision,
    *,
    answer: str = "",
    bot_message_id: str = "",
    user_message_id: str = "",
    trigger_message_ids: Sequence[str] = (),
    turn_id: str = "",
    db_path: str | Path | None = None,
) -> None:
    state = ConversationState(
        last_question=question,
        sources=decision.sources,
        timestamp=time.time(),
        user_id=str(user_id) if user_id is not None else "",
        last_answer=answer,
        reply_mode=decision.reply_mode,
        bot_message_id=str(bot_message_id or ""),
        user_message_id=str(user_message_id or ""),
        trigger_message_ids=tuple(dict.fromkeys(
            str(value or "").strip()
            for value in trigger_message_ids
            if str(value or "").strip()
        )),
        turn_id=str(turn_id or "").strip(),
        semantic_intent=decision.semantic_intent,
        semantic_topic=decision.semantic_topic,
    )
    try:
        deps.persist_conversation_turn(group_id, state, db_path=db_path)
    except Exception as exc:
        print("Persist conversation turn failed:", repr(exc))


def group_chat_has_newer_user_message(deps, group_id: int, sequence: int) -> bool:
    return deps.chat_history_state.has_newer_user_message(
        group_id,
        sequence,
        bot_user_id=deps.settings.bot_qq,
    )


def latest_group_user_sequence(deps, group_id: int) -> int:
    return deps.chat_history_state.latest_user_sequence(
        group_id,
        bot_user_id=deps.settings.bot_qq,
    )


def clear_chat_state(deps) -> None:
    deps.chat_message_sequence = 0
    with deps.chat_history_lock:
        deps.chat_history_state.clear()
    deps.chat_scene_state.clear()
    with deps.hostile_reply_lock:
        deps.hostile_reply_history.clear()
    deps.semantic_planner_health.reset()
    deps.clear_fragment_state()


def group_send_lock(deps, group_id: int) -> threading.Lock:
    with deps.group_send_locks_lock:
        return deps.group_send_locks.setdefault(group_id, threading.Lock())


def allow_hostile_reply(deps, group_id: int, user_id: str, *, now: float | None = None) -> bool:
    current_time = time.time() if now is None else now
    key = (group_id, str(user_id or ""))
    with deps.hostile_reply_lock:
        # Size-triggered cleanup to prevent unbounded key growth
        if len(deps.hostile_reply_history) > 5000:
            deps.hostile_reply_history.clear()
        recent = [
            timestamp
            for timestamp in deps.hostile_reply_history.get(key, ())
            if current_time - timestamp < 600
        ]
        allowed = len(recent) < 2
        recent.append(current_time)
        deps.hostile_reply_history[key] = recent
        return allowed


# ---------------------------------------------------------------------------
# Batch 2 – group chat participation strategy
# ---------------------------------------------------------------------------


def has_substantive_chat_context(deps, chat_context: Sequence[str]) -> bool:
    for line in chat_context:
        text = line.split("：", 1)[-1].strip().lower()
        if text and text not in BARE_CHAT_REACTIONS:
            return True
    return False


def chat_reply_quota_reason(
    deps,
    group_id: int,
    *,
    now: float | None = None,
    cooldown_seconds: int | None = None,
    max_per_hour: int | None = None,
    db_path: str | Path | None = None,
) -> str:
    current_time = time.time() if now is None else now
    cooldown = (
        deps.settings.chat_reply_cooldown_seconds
        if cooldown_seconds is None
        else cooldown_seconds
    )
    hourly_limit = (
        deps.settings.max_chat_replies_per_hour
        if max_per_hour is None
        else max_per_hour
    )
    with deps.chat_reply_lock:
        return deps.pending_store.chat_reply_quota_reason(
            group_id,
            db_path=deps._pending_db_path(db_path),
            now=current_time,
            cooldown_seconds=cooldown,
            max_per_hour=hourly_limit,
        )


def mark_chat_replied(
    deps,
    group_id: int,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    timestamp = time.time() if now is None else now
    with deps.chat_reply_lock:
        deps.pending_store.mark_chat_replied(
            group_id,
            db_path=deps._pending_db_path(db_path),
            now=timestamp,
        )


def check_and_mark_chat_reply_quota(
    deps,
    group_id: int,
    *,
    now: float | None = None,
    cooldown_seconds: int | None = None,
    max_per_hour: int | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Atomically check quota and mark if allowed. Returns empty string if OK, reason if blocked."""
    current_time = time.time() if now is None else now
    cooldown = (
        deps.settings.chat_reply_cooldown_seconds
        if cooldown_seconds is None
        else cooldown_seconds
    )
    hourly_limit = (
        deps.settings.max_chat_replies_per_hour
        if max_per_hour is None
        else max_per_hour
    )
    with deps.chat_reply_lock:
        reason = deps.pending_store.chat_reply_quota_reason(
            group_id,
            db_path=deps._pending_db_path(db_path),
            now=current_time,
            cooldown_seconds=cooldown,
            max_per_hour=hourly_limit,
        )
        if not reason:
            deps.pending_store.mark_chat_replied(
                group_id,
                db_path=deps._pending_db_path(db_path),
                now=current_time,
            )
        return reason


def celebration_was_replied(
    deps,
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    now: float | None = None,
    window_seconds: int = 86400,
    db_path: str | Path | None = None,
) -> bool:
    current_time = time.time() if now is None else now
    with deps.chat_reply_lock:
        return deps.pending_store.celebration_was_replied(
            group_id,
            target_key,
            event_kind,
            db_path=deps._pending_db_path(db_path),
            now=current_time,
            window_seconds=window_seconds,
        )


def mark_celebration_replied(
    deps,
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    timestamp = time.time() if now is None else now
    with deps.chat_reply_lock:
        deps.pending_store.mark_celebration_replied(
            group_id,
            target_key,
            event_kind,
            db_path=deps._pending_db_path(db_path),
            now=timestamp,
        )


def has_auto_reply_keyword(deps, message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in AUTO_REPLY_KEYWORDS)


def looks_like_direct_question(deps, message: str) -> bool:
    lowered = message.lower()
    return any(cue in lowered for cue in QUESTION_CUES + HELP_CUES)


def looks_like_birthday_celebration(deps, message: str) -> bool:
    lowered = message.lower()
    if "生日" not in lowered and "寿星" not in lowered:
        return False
    if any(cue in lowered for cue in BIRTHDAY_DISCUSSION_CUES):
        return False
    if any(cue in lowered for cue in BIRTHDAY_CELEBRATION_CUES):
        return True
    return "生日" in lowered and any(cue in lowered for cue in ("祝", "快乐", "今天", "今儿"))


def celebration_kind(deps, message: str) -> str:
    if looks_like_birthday_celebration(deps, message):
        return "birthday"
    lowered = message.lower()
    if "毕业" not in lowered:
        return ""
    if any(cue in lowered for cue in GRADUATION_DISCUSSION_CUES):
        return ""
    if any(cue in lowered for cue in GRADUATION_CELEBRATION_CUES):
        return "graduation"
    if "毕业" in lowered and any(cue in lowered for cue in ("祝", "恭喜", "今天", "终于")):
        return "graduation"
    return ""


def is_self_celebration(deps, message: str, kind: str) -> bool:
    compact = "".join(message.lower().split())
    if kind == "birthday":
        return any(cue in compact for cue in ("我生日", "我今天生日", "今天我生日", "我过生日"))
    if kind == "graduation":
        return any(cue in compact for cue in ("我毕业了", "我毕业啦", "我顺利毕业", "我终于毕业"))
    return False


def response_mention_user_id(
    deps,
    *,
    mentioned: bool,
    user_id,
    reply_mode: str,
    question: str,
    mentioned_user_ids: Sequence[str] = (),
) -> str:
    bot_qq = str(getattr(deps.settings, "bot_qq", ""))
    kind = celebration_kind(deps, question) if reply_mode == "chat" else ""
    if kind:
        for candidate in mentioned_user_ids:
            candidate_id = str(candidate).strip()
            if candidate_id.isdigit() and candidate_id != bot_qq:
                return candidate_id
        if is_self_celebration(deps, question, kind):
            sender_id = str(user_id or "").strip()
            return sender_id if sender_id.isdigit() and sender_id != bot_qq else ""
        return ""
    if mentioned:
        sender_id = str(user_id or "").strip()
        return sender_id if sender_id.isdigit() and sender_id != bot_qq else ""
    return ""


def looks_like_assignment_to_humans(deps, message: str) -> bool:
    lowered = message.lower()
    if any(cue in lowered for cue in BOT_META_CUES):
        return True
    has_assignment = any(cue in lowered for cue in ASSIGNMENT_CUES)
    if not has_assignment:
        return False
    if "介绍一下" in lowered and (
        any(cue in lowered for cue in BROAD_CONTENT_CUES)
        or any(cue in lowered for cue in META_CONTENT_CUES)
    ):
        return True
    return any(cue in lowered for cue in NON_BOT_TARGET_CUES) or not looks_like_direct_question(deps, lowered)


def is_identity_question(deps, question: str) -> bool:
    normalized_question = question.strip().lower()
    if any(keyword in normalized_question for keyword in IDENTITY_KEYWORDS):
        return True
    if "介绍一下" in normalized_question:
        return any(keyword in normalized_question for keyword in SELF_REFERENCE_KEYWORDS)
    return False


# ---------------------------------------------------------------------------
# Chat reply decision (depends heavily on batch 2 functions and constants)
# ---------------------------------------------------------------------------


def consider_chat_reply(
    deps,
    normalized: str,
    *,
    group_id: int,
    chat_context: Sequence[str],
    mentions_other: bool,
    has_context: bool = False,
    sources: Sequence[str] = (),
    query_text: str = "",
    followup_of: str = "",
    followup_scope: str = "",
) -> ProcessingDecision:
    if not deps.auto_reply_enabled:
        print("Skip message: auto reply disabled", normalized)
        return ProcessingDecision(False, "auto reply disabled", has_context, tuple(sources))
    if not deps.settings.chat_reply_enabled:
        reason = "not a direct question or help request" if has_context else "no knowledge context"
        print("Skip message:", reason, normalized)
        return ProcessingDecision(False, reason, has_context, tuple(sources))
    if deps.settings.chat_allowed_group_ids and str(group_id) not in deps.settings.chat_allowed_group_ids:
        return ProcessingDecision(False, "chat group not allowed", has_context, tuple(sources))
    social_celebration = bool(deps.celebration_kind(normalized))
    if len(normalized) < 2:
        return ProcessingDecision(False, "too short for chat reply", has_context, tuple(sources))
    if (
        normalized.strip().lower() in BARE_CHAT_REACTIONS
        and not deps.has_substantive_chat_context(chat_context)
    ):
        return ProcessingDecision(False, "bare reaction without context", has_context, tuple(sources))
    # Fast filter: very short messages without question marks or help cues
    # These rarely need an LLM call to decide
    # Skip if already handled by BARE_CHAT_REACTIONS filter above
    if (
        len(normalized) <= 4
        and "?" not in normalized
        and "？" not in normalized
        and not social_celebration
        and not deps.looks_like_direct_question(normalized)
        and normalized.strip().lower() not in BARE_CHAT_REACTIONS
    ):
        return ProcessingDecision(False, "short reaction fast-skip", has_context, tuple(sources))
    if mentions_other and not social_celebration:
        return ProcessingDecision(False, "chat directed at another member", has_context, tuple(sources))
    if deps.looks_like_assignment_to_humans(normalized):
        return ProcessingDecision(False, "chat filtered assignment or bot meta", has_context, tuple(sources))
    lowered = normalized.lower()
    if any(cue in lowered for cue in CHAT_BLOCK_CUES):
        return ProcessingDecision(False, "chat filtered announcement, link, or meta", has_context, tuple(sources))
    if (
        deps.looks_like_direct_question(normalized)
        and deps.has_auto_reply_keyword(normalized)
        and not social_celebration
    ):
        reason = (
            "weak knowledge context for factual question"
            if has_context
            else "no knowledge context for factual question"
        )
        return ProcessingDecision(False, reason, has_context, tuple(sources))
    quota_reason = deps.chat_reply_quota_reason(group_id)
    if quota_reason:
        print("Skip chat message:", quota_reason, group_id, normalized)
        return ProcessingDecision(False, quota_reason, has_context, tuple(sources))
    return ProcessingDecision(
        True,
        "social celebration candidate queued" if social_celebration else "chat candidate queued",
        has_context,
        tuple(sources),
        query_text or normalized,
        followup_of,
        followup_scope,
        "chat",
        tuple(chat_context),
    )
