from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence
from urllib.parse import parse_qs

from .config import settings
from .knowledge import KnowledgeBase
from .llm import (
    answer_chat,
    analyze_scene,
    ask_fallback_llm,
    ask_llm,
    is_chat_no_reply,
    normalize_model_answer,
    should_auto_reply,
    should_reply_to_chat,
)
from .onebot import (
    extract_mentioned_user_ids,
    extract_plain_text,
    extract_reply_message_id,
    get_message_info,
    is_mentioned,
    send_group_msg,
    should_respond,
)


kb = KnowledgeBase(settings.knowledge_dir)
message_queue: queue.PriorityQueue = queue.PriorityQueue()
normal_message_queue: queue.PriorityQueue = queue.PriorityQueue()
chat_queue: queue.Queue = queue.Queue()
reply_timestamps: list[float] = []
rate_limit_lock = threading.Lock()
kb_lock = threading.RLock()
sequence_lock = threading.Lock()
sequence_number = 0
audit_lock = threading.Lock()
auto_reply_enabled = settings.auto_reply_enabled
topic_cooldown_lock = threading.Lock()
recent_reply_topics: dict[tuple[int, str], float] = {}
followup_lock = threading.Lock()
user_followup_state: dict[tuple[int, str], "ConversationState"] = {}
group_followup_state: dict[int, "ConversationState"] = {}
chat_history_lock = threading.Lock()
group_chat_history: dict[int, list["GroupChatMessage"]] = {}
chat_message_sequence = 0
chat_reply_lock = threading.Lock()


@dataclass
class ProcessingDecision:
    should_reply: bool
    reason: str
    has_context: bool = False
    sources: tuple[str, ...] = ()
    effective_question: str = ""
    followup_of: str = ""
    followup_scope: str = ""
    reply_mode: str = "knowledge"
    chat_context: tuple[str, ...] = ()
    retrieval_score: float = 0.0
    retrieval_coverage: float = 0.0


@dataclass
class ConversationState:
    last_question: str
    sources: tuple[str, ...]
    timestamp: float
    user_id: str = ""


@dataclass
class FollowupMatch:
    state: ConversationState
    scope: str


@dataclass
class GroupChatMessage:
    text: str
    user_id: str
    timestamp: float
    sequence: int = 0
    message_id: str = ""
    reply_message_id: str = ""
    reply_target_user_id: str = ""
    reply_text: str = ""


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

FOLLOWUP_CUES = (
    "那",
    "那么",
    "这个",
    "这种",
    "这样",
    "为啥",
    "为什么",
    "然后",
    "还有呢",
    "怎么说",
    "怎么处理",
    "咋办",
    "怎么办",
    "要不要",
    "还要",
    "先奶",
    "奶满",
    "补满",
    "先拉",
    "拉起来",
    "一堆人",
    "倒一起",
    "上车",
    "转点",
    "呢",
    "吗",
)

COMMAND_ALIASES = {
    "重载知识库": "reload",
    "reload": "reload",
    "健康状态": "health",
    "查看健康状态": "health",
    "状态": "health",
    "health": "health",
    "最近跳过": "recent_skips",
    "最近跳过消息": "recent_skips",
    "skipped": "recent_skips",
    "skips": "recent_skips",
    "自动回复开": "auto_reply_on",
    "开启自动回复": "auto_reply_on",
    "打开自动回复": "auto_reply_on",
    "auto on": "auto_reply_on",
    "自动回复关": "auto_reply_off",
    "关闭自动回复": "auto_reply_off",
    "关掉自动回复": "auto_reply_off",
    "auto off": "auto_reply_off",
}


def _rotate_log_if_needed(log_path: Path, max_bytes: int = 5 * 1024 * 1024, keep: int = 5) -> None:
    """Rotate log file if it exceeds max_bytes. Keeps the last `keep` rotated files."""
    try:
        if not log_path.exists() or log_path.stat().st_size < max_bytes:
            return
        # Shift existing rotated files: .5 -> delete, .4 -> .5, ..., .1 -> .2
        for i in range(keep, 0, -1):
            src = log_path.with_suffix(f"{log_path.suffix}.{i}")
            if i >= keep:
                if src.exists():
                    src.unlink()
            else:
                dst = log_path.with_suffix(f"{log_path.suffix}.{i + 1}")
                if src.exists():
                    src.rename(dst)
        # Rotate current file to .1
        rotated = log_path.with_suffix(f"{log_path.suffix}.1")
        log_path.rename(rotated)
        print(f"Rotated audit log: {log_path} -> {rotated}")
    except Exception as exc:
        print("Audit log rotation failed:", repr(exc))


def write_message_audit(
    *,
    decision: str,
    reason: str,
    group_id=None,
    user_id=None,
    question: str = "",
    mentioned: bool = False,
    has_context: bool = False,
    sources: Sequence[str] = (),
    followup_of: str = "",
    followup_scope: str = "",
    reply_mode: str = "",
    retrieval_score: float = 0.0,
    retrieval_coverage: float = 0.0,
    model_latency_ms: int = 0,
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    chat_context: Sequence[str] = (),
    answer: str = "",
    mention_user_id: str = "",
    event_time=None,
) -> None:
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event_time": event_time,
        "group_id": str(group_id) if group_id is not None else "",
        "user_id": str(user_id) if user_id is not None else "",
        "question": question,
        "mentioned": bool(mentioned),
        "decision": decision,
        "reason": reason,
        "has_context": bool(has_context),
        "sources": list(sources),
        "followup_of": followup_of,
        "followup_scope": followup_scope,
        "reply_mode": reply_mode,
        "retrieval_score": round(float(retrieval_score), 4),
        "retrieval_coverage": round(float(retrieval_coverage), 4),
        "model_latency_ms": int(model_latency_ms),
        "reply_message_id": str(reply_message_id),
        "reply_target_user_id": str(reply_target_user_id),
        "chat_context": list(chat_context),
        "answer": answer,
        "mention_user_id": str(mention_user_id or ""),
    }
    log_path = Path(settings.message_audit_log)
    try:
        with audit_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_log_if_needed(log_path, max_bytes=5 * 1024 * 1024, keep=5)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print("Audit log write failed:", repr(exc))


def extract_event_question(event: dict) -> tuple[str, bool]:
    raw_message = event.get("message", "")
    text = extract_plain_text(raw_message)
    mentioned = is_mentioned(settings.bot_qq, raw_message)
    _ok, question = should_respond(text, settings.command_prefix, settings.bot_qq, raw_message)
    return question, mentioned


def classify_reply_target(
    reply_message_id: str,
    reply_target_user_id: str,
    mentioned: bool,
    bot_qq: str,
) -> tuple[bool, bool, str]:
    if not reply_message_id:
        return True, mentioned, ""
    if mentioned:
        return True, True, "explicit mention"
    if reply_target_user_id and reply_target_user_id == bot_qq:
        return True, True, "reply to bot"
    if reply_target_user_id:
        return False, False, "reply directed at another member"
    return False, False, "reply target unknown"


def normalize_command_text(message: str) -> str:
    return " ".join(message.strip().lower().split())


def get_admin_command(message: str) -> str:
    normalized = normalize_command_text(message)
    return COMMAND_ALIASES.get(normalized, "")


def is_restored_admin_command(item: dict) -> bool:
    return bool(item.get("_restored") and get_admin_command(str(item.get("question", ""))))


def is_admin_user(user_id, sender_role: str = "") -> bool:
    if str(sender_role).lower() in {"owner", "admin"}:
        return True
    return bool(settings.admin_qq_ids) and str(user_id) in settings.admin_qq_ids


def recent_audit_entries(limit: int = 5) -> list[dict]:
    log_path = Path(settings.message_audit_log)
    if not log_path.exists():
        return []
    entries: list[dict] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("decision") not in {"skipped", "ignored"}:
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def answer_admin_command(command: str) -> str:
    global auto_reply_enabled
    if command == "reload":
        with kb_lock:
            count = kb.reload()
        clear_conversation_state()
        return f"知识库已重载，共 {count} 个片段。"
    if command == "health":
        auto_status = "开" if auto_reply_enabled else "关"
        chat_status = "开" if settings.chat_reply_enabled and auto_reply_enabled else "关"
        queued = message_queue.qsize() + normal_message_queue.qsize() + chat_queue.qsize()
        return (
            f"服务正常。知识片段 {len(kb.chunks)} 个，队列 {queued} 条，"
            f"自动回复{auto_status}，闲聊{chat_status}，"
            f"每分钟最多回复 {settings.max_replies_per_minute} 条。"
        )
    if command == "recent_skips":
        entries = recent_audit_entries()
        if not entries:
            return "最近还没有记录到跳过消息。"
        parts: list[str] = []
        for entry in entries:
            question = str(entry.get("question") or "").strip()
            if len(question) > 28:
                question = question[:28] + "..."
            reason = entry.get("reason") or "unknown"
            parts.append(f"{entry.get('group_id') or '未知群'}：{reason}：{question or '空消息'}")
        return "最近跳过消息：\n" + "\n".join(parts)
    if command == "auto_reply_on":
        auto_reply_enabled = True
        return "自动回复已开启。"
    if command == "auto_reply_off":
        auto_reply_enabled = False
        return "自动回复已关闭。被 @ 时仍可回答。"
    return "未知维护命令。"


def topic_key(question: str, decision: ProcessingDecision) -> str:
    if decision.sources:
        return decision.sources[0]
    return normalize_command_text(question)[:40]


def is_topic_on_cooldown(group_id: int, key: str) -> bool:
    if settings.same_topic_cooldown_seconds <= 0:
        return False
    now = time.time()
    with topic_cooldown_lock:
        expired = [
            item_key
            for item_key, timestamp in recent_reply_topics.items()
            if now - timestamp > settings.same_topic_cooldown_seconds
        ]
        for item_key in expired:
            recent_reply_topics.pop(item_key, None)
        last_reply = recent_reply_topics.get((group_id, key))
    return last_reply is not None and now - last_reply <= settings.same_topic_cooldown_seconds


def mark_topic_replied(group_id: int, key: str) -> None:
    if settings.same_topic_cooldown_seconds <= 0:
        return
    with topic_cooldown_lock:
        recent_reply_topics[(group_id, key)] = time.time()


def cleanup_followup_state(now: float) -> None:
    max_window = max(
        settings.followup_same_user_seconds,
        settings.followup_group_seconds,
        settings.followup_mention_seconds,
    )
    if max_window <= 0:
        return
    with followup_lock:
        user_expired = [
            state_key
            for state_key, state in user_followup_state.items()
            if now - state.timestamp > max_window
        ]
        for state_key in user_expired:
            user_followup_state.pop(state_key, None)
        group_expired = [
            group_id
            for group_id, state in group_followup_state.items()
            if now - state.timestamp > max_window
        ]
        for group_id in group_expired:
            group_followup_state.pop(group_id, None)


def looks_like_followup(question: str, mentioned: bool) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False
    if mentioned and len(normalized) <= 28:
        return True
    if len(normalized) <= 16:
        return True
    return any(cue in normalized for cue in FOLLOWUP_CUES)


def followup_context_for(
    group_id: int,
    user_id,
    question: str,
    mentioned: bool,
) -> FollowupMatch | None:
    if (
        settings.followup_same_user_seconds <= 0
        and settings.followup_group_seconds <= 0
        and settings.followup_mention_seconds <= 0
    ):
        return None
    if not looks_like_followup(question, mentioned):
        return None
    now = time.time()
    cleanup_followup_state(now)
    user_key = str(user_id) if user_id is not None else ""
    user_ttl = settings.followup_mention_seconds if mentioned else settings.followup_same_user_seconds
    group_ttl = settings.followup_mention_seconds if mentioned else settings.followup_group_seconds

    candidates: list[tuple[int, ConversationState, str]] = []
    if user_key:
        with followup_lock:
            state = user_followup_state.get((group_id, user_key))
        if state and now - state.timestamp <= user_ttl:
            candidates.append((2, state, "user"))

    with followup_lock:
        group_state = group_followup_state.get(group_id)
    if group_state and now - group_state.timestamp <= group_ttl:
        candidates.append((1, group_state, "group"))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
    _priority, state, scope = candidates[0]
    return FollowupMatch(state=state, scope=scope)


def build_effective_question(question: str, followup_match: FollowupMatch | None) -> str:
    if not followup_match:
        return question
    return f"上一轮问题：{followup_match.state.last_question}\n当前追问：{question}"


def remember_conversation(group_id: int, user_id, question: str, decision: ProcessingDecision) -> None:
    if (
        settings.followup_same_user_seconds <= 0
        and settings.followup_group_seconds <= 0
        and settings.followup_mention_seconds <= 0
    ):
        return
    if decision.reply_mode != "knowledge" or not decision.sources:
        return
    state = ConversationState(
        last_question=question,
        sources=decision.sources,
        timestamp=time.time(),
        user_id=str(user_id) if user_id is not None else "",
    )
    with followup_lock:
        if user_id is not None:
            user_followup_state[(group_id, str(user_id))] = state
        group_followup_state[group_id] = state


def clear_conversation_state() -> None:
    with followup_lock:
        user_followup_state.clear()
        group_followup_state.clear()


def record_group_chat_message(
    group_id: int,
    user_id,
    text: str,
    event_time=None,
    *,
    message_id: str = "",
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    reply_text: str = "",
) -> int:
    global chat_message_sequence
    normalized = text.strip()
    if not normalized:
        return 0
    try:
        timestamp = float(event_time)
    except (TypeError, ValueError):
        timestamp = time.time()
    window = max(0, settings.chat_context_seconds)
    max_messages = max(1, settings.chat_context_messages)
    with chat_history_lock:
        chat_message_sequence += 1
        entry = GroupChatMessage(
            normalized,
            str(user_id or ""),
            timestamp,
            chat_message_sequence,
            str(message_id or ""),
            str(reply_message_id or ""),
            str(reply_target_user_id or ""),
            str(reply_text or "").strip(),
        )
        history = group_chat_history.setdefault(group_id, [])
        history.append(entry)
        if window > 0:
            history[:] = [item for item in history if timestamp - item.timestamp <= window]
        history[:] = history[-max(max_messages * 4, 24):]
        return entry.sequence


def find_group_chat_message(group_id: int, message_id: str) -> GroupChatMessage | None:
    target = str(message_id or "").strip()
    if not target:
        return None
    with chat_history_lock:
        for item in reversed(group_chat_history.get(group_id, ())):
            if item.message_id == target:
                return item
    return None


def recent_group_chat_context(
    group_id: int,
    *,
    now: float | None = None,
    context_seconds: int | None = None,
    max_messages: int | None = None,
    focus_sequence: int = 0,
) -> tuple[str, ...]:
    current_time = time.time() if now is None else now
    window = settings.chat_context_seconds if context_seconds is None else context_seconds
    limit = settings.chat_context_messages if max_messages is None else max_messages
    if window <= 0 or limit <= 0:
        return ()
    with chat_history_lock:
        history = group_chat_history.get(group_id, [])
        recent = [item for item in history if current_time - item.timestamp <= window]
        if recent:
            group_chat_history[group_id] = recent
        else:
            group_chat_history.pop(group_id, None)
    selected = recent[-limit:]
    if focus_sequence:
        focus = next((item for item in recent if item.sequence == focus_sequence), None)
        if focus and focus.reply_message_id:
            replied = next(
                (item for item in recent if item.message_id == focus.reply_message_id),
                None,
            )
            if replied and replied not in selected:
                tail = selected[-(limit - 1):] if limit > 1 else []
                selected = sorted(
                    [replied, *tail],
                    key=lambda item: item.sequence,
                )
    aliases: dict[str, str] = {}
    lines: list[str] = []

    def speaker(user_id: str) -> str:
        user_key = user_id or "未知"
        if user_key == settings.bot_qq:
            return "机器人自己"
        if user_key not in aliases:
            aliases[user_key] = chr(ord("A") + len(aliases))
        return f"群友{aliases[user_key]}"

    for item in selected:
        label = speaker(item.user_id)
        relation = ""
        if item.reply_message_id:
            target_label = speaker(item.reply_target_user_id) if item.reply_target_user_id else "某条消息"
            quoted = item.reply_text.replace("\n", " ").strip()
            if len(quoted) > 60:
                quoted = quoted[:59] + "…"
            relation = f"（回复{target_label}"
            if quoted:
                relation += f"“{quoted}”"
            relation += "）"
        current = "【当前消息】" if item.sequence == focus_sequence else ""
        lines.append(f"{current}{label}{relation}：{item.text}")
    return tuple(lines)


def has_substantive_chat_context(chat_context: Sequence[str]) -> bool:
    for line in chat_context:
        text = line.split("：", 1)[-1].strip().lower()
        if text and text not in BARE_CHAT_REACTIONS:
            return True
    return False


def group_chat_has_newer_user_message(group_id: int, sequence: int) -> bool:
    with chat_history_lock:
        return any(
            item.sequence > sequence and item.user_id != settings.bot_qq
            for item in group_chat_history.get(group_id, ())
        )


def clear_chat_state() -> None:
    global chat_message_sequence
    with chat_history_lock:
        group_chat_history.clear()
        chat_message_sequence = 0


def save_chat_history(path: str | Path | None = None) -> None:
    """Persist chat history to disk for restart recovery."""
    save_path = Path(path or "work/chat_history.json")
    try:
        with chat_history_lock:
            data = {}
            for group_id, messages in group_chat_history.items():
                data[str(group_id)] = [
                    {
                        "text": m.text,
                        "user_id": m.user_id,
                        "timestamp": m.timestamp,
                        "sequence": m.sequence,
                        "message_id": m.message_id,
                        "reply_message_id": m.reply_message_id,
                        "reply_target_user_id": m.reply_target_user_id,
                        "reply_text": m.reply_text,
                    }
                    for m in messages
                ]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print("Save chat history failed:", repr(exc))


def load_chat_history(path: str | Path | None = None) -> int:
    """Load persisted chat history on startup."""
    global chat_message_sequence
    load_path = Path(path or "work/chat_history.json")
    if not load_path.exists():
        return 0
    try:
        data = json.loads(load_path.read_text(encoding="utf-8"))
        count = 0
        with chat_history_lock:
            for group_id_str, messages in data.items():
                group_id = int(group_id_str)
                history = group_chat_history.setdefault(group_id, [])
                for m in messages:
                    entry = GroupChatMessage(
                        text=m["text"],
                        user_id=m["user_id"],
                        timestamp=m["timestamp"],
                        sequence=m["sequence"],
                        message_id=m.get("message_id", ""),
                        reply_message_id=m.get("reply_message_id", ""),
                        reply_target_user_id=m.get("reply_target_user_id", ""),
                        reply_text=m.get("reply_text", ""),
                    )
                    history.append(entry)
                    chat_message_sequence = max(chat_message_sequence, entry.sequence)
                    count += 1
        print(f"Loaded {count} chat history entries from {load_path}")
        return count
    except Exception as exc:
        print("Load chat history failed:", repr(exc))
        return 0


def chat_reply_quota_reason(
    group_id: int,
    *,
    now: float | None = None,
    cooldown_seconds: int | None = None,
    max_per_hour: int | None = None,
    db_path: str | Path | None = None,
) -> str:
    current_time = time.time() if now is None else now
    cooldown = (
        settings.chat_reply_cooldown_seconds
        if cooldown_seconds is None
        else cooldown_seconds
    )
    hourly_limit = (
        settings.max_chat_replies_per_hour
        if max_per_hour is None
        else max_per_hour
    )
    with chat_reply_lock:
        connection = open_pending_queue_db(db_path)
        try:
            row = connection.execute(
                "SELECT MAX(replied_at), COUNT(*) FROM chat_reply_history "
                "WHERE group_id = ? AND replied_at > ?",
                (str(group_id), current_time - 3600),
            ).fetchone()
        finally:
            connection.close()
    previous = float(row[0]) if row and row[0] is not None else None
    recent_count = int(row[1]) if row else 0
    if cooldown > 0 and previous is not None and current_time - previous < cooldown:
        return "chat cooldown"
    if hourly_limit > 0 and recent_count >= hourly_limit:
        return "chat hourly limit"
    return ""


def mark_chat_replied(
    group_id: int,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    timestamp = time.time() if now is None else now
    with chat_reply_lock:
        connection = open_pending_queue_db(db_path)
        try:
            connection.execute(
                "DELETE FROM chat_reply_history WHERE replied_at <= ?",
                (timestamp - 3600,),
            )
            connection.execute(
                "INSERT INTO chat_reply_history (group_id, replied_at) VALUES (?, ?)",
                (str(group_id), timestamp),
            )
            connection.commit()
        finally:
            connection.close()


def celebration_was_replied(
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    now: float | None = None,
    window_seconds: int = 86400,
    db_path: str | Path | None = None,
) -> bool:
    current_time = time.time() if now is None else now
    with chat_reply_lock:
        connection = open_pending_queue_db(db_path)
        try:
            row = connection.execute(
                "SELECT 1 FROM celebration_reply_history "
                "WHERE group_id = ? AND target_key = ? AND event_kind = ? "
                "AND replied_at > ? LIMIT 1",
                (str(group_id), target_key, event_kind, current_time - window_seconds),
            ).fetchone()
        finally:
            connection.close()
    return row is not None


def mark_celebration_replied(
    group_id: int,
    target_key: str,
    event_kind: str,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    timestamp = time.time() if now is None else now
    with chat_reply_lock:
        connection = open_pending_queue_db(db_path)
        try:
            connection.execute(
                "DELETE FROM celebration_reply_history WHERE replied_at <= ?",
                (timestamp - 604800,),
            )
            connection.execute(
                "INSERT INTO celebration_reply_history "
                "(group_id, target_key, event_kind, replied_at) VALUES (?, ?, ?, ?)",
                (str(group_id), target_key, event_kind, timestamp),
            )
            connection.commit()
        finally:
            connection.close()


def has_auto_reply_keyword(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in AUTO_REPLY_KEYWORDS)


def looks_like_direct_question(message: str) -> bool:
    lowered = message.lower()
    return any(cue in lowered for cue in QUESTION_CUES + HELP_CUES)


def looks_like_birthday_celebration(message: str) -> bool:
    lowered = message.lower()
    if "生日" not in lowered and "寿星" not in lowered:
        return False
    if any(cue in lowered for cue in BIRTHDAY_DISCUSSION_CUES):
        return False
    if any(cue in lowered for cue in BIRTHDAY_CELEBRATION_CUES):
        return True
    return "生日" in lowered and any(cue in lowered for cue in ("祝", "快乐", "今天", "今儿"))


def celebration_kind(message: str) -> str:
    if looks_like_birthday_celebration(message):
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


def is_self_celebration(message: str, kind: str) -> bool:
    compact = "".join(message.lower().split())
    if kind == "birthday":
        return any(cue in compact for cue in ("我生日", "我今天生日", "今天我生日", "我过生日"))
    if kind == "graduation":
        return any(cue in compact for cue in ("我毕业了", "我毕业啦", "我顺利毕业", "我终于毕业"))
    return False


def response_mention_user_id(
    *,
    mentioned: bool,
    user_id,
    reply_mode: str,
    question: str,
    mentioned_user_ids: Sequence[str] = (),
) -> str:
    bot_qq = str(getattr(settings, "bot_qq", ""))
    kind = celebration_kind(question) if reply_mode == "chat" else ""
    if kind:
        for candidate in mentioned_user_ids:
            candidate_id = str(candidate).strip()
            if candidate_id.isdigit() and candidate_id != bot_qq:
                return candidate_id
        if is_self_celebration(question, kind):
            sender_id = str(user_id or "").strip()
            return sender_id if sender_id.isdigit() and sender_id != bot_qq else ""
        return ""
    if mentioned:
        sender_id = str(user_id or "").strip()
        return sender_id if sender_id.isdigit() and sender_id != bot_qq else ""
    return ""


def looks_like_assignment_to_humans(message: str) -> bool:
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
    return any(cue in lowered for cue in NON_BOT_TARGET_CUES) or not looks_like_direct_question(lowered)


def is_identity_question(question: str) -> bool:
    normalized_question = question.strip().lower()
    if any(keyword in normalized_question for keyword in IDENTITY_KEYWORDS):
        return True
    if "介绍一下" in normalized_question:
        return any(keyword in normalized_question for keyword in SELF_REFERENCE_KEYWORDS)
    return False


def retrieve_knowledge(query: str, max_chars: int):
    with kb_lock:
        return kb.build_context_with_metrics(query, max_chars)


def is_strong_knowledge_match(top_score: float, query_coverage: float) -> bool:
    return (
        top_score >= settings.knowledge_strong_min_score
        and query_coverage >= settings.knowledge_strong_min_coverage
    )


def finalize_model_answer(answer: str, *, unsolicited: bool = False) -> str:
    if is_model_error_answer(answer):
        if unsolicited:
            return ""
        return "这会儿回复服务有点忙，稍后再问我一下。"
    normalized = normalize_model_answer(answer, settings.max_answer_chars)
    if normalized:
        return normalized
    if unsolicited:
        return ""
    return "这会儿回复服务有点忙，稍后再问我一下。"


def answer_question(
    question: str,
    effective_question: str | None = None,
    *,
    allow_fallback: bool = True,
) -> str:
    if is_identity_question(question):
        return (
            "叫我新兵营教官就行，主要给刚入坑 Squad 的兄弟答疑。"
            "HAB、FOB、医疗兵、反坦、搜不到服、卡三点、TS 设置这些都能问。"
            "要是问到本服规则，我不乱拍板，按群公告和管理员说法来。"
        )

    llm_question = effective_question or question
    result = retrieve_knowledge(llm_question, settings.max_context_chars)
    strong_match = is_strong_knowledge_match(
        result.top_score,
        result.query_coverage,
    )
    if not result.context or not strong_match:
        if settings.llm_fallback_enabled and allow_fallback:
            answer = ask_fallback_llm(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                question=llm_question,
            )
            return finalize_model_answer(answer)
        return "这个我库里暂时没有准确信息。你可以换个更具体的问法，或者问一下小队长和管理员；涉及服务器规则的话，还是以本服公告为准。"

    answer = ask_llm(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        question=llm_question,
        context=result.context,
    )
    return finalize_model_answer(answer)


def answer_for_decision(
    question: str,
    decision: ProcessingDecision,
    effective_question: str,
) -> str:
    llm_question = decision.effective_question or effective_question or question
    if decision.reply_mode == "fallback":
        answer = ask_fallback_llm(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            question=llm_question,
            context=decision.chat_context,
        )
        return finalize_model_answer(answer)
    if decision.reply_mode == "chat":
        answer = answer_chat(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            message=question,
            context=decision.chat_context,
        )
        return finalize_model_answer(answer, unsolicited=True)
    return answer_question(question, llm_question, allow_fallback=False)


def is_model_error_answer(answer: str) -> bool:
    return answer.startswith(("模型接口", "还没有配置模型 API Key"))


def next_sequence() -> int:
    global sequence_number
    with sequence_lock:
        sequence_number += 1
        return sequence_number


def open_pending_queue_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or settings.pending_queue_db)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            priority INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            payload TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_reply_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            replied_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_reply_group_time "
        "ON chat_reply_history (group_id, replied_at)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS celebration_reply_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id TEXT NOT NULL,
            target_key TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            replied_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_celebration_reply_lookup "
        "ON celebration_reply_history (group_id, target_key, event_kind, replied_at)"
    )
    connection.commit()
    return connection


def persist_pending_message(
    priority: int,
    sequence: int,
    item: dict,
    db_path: str | Path | None = None,
) -> int:
    connection = open_pending_queue_db(db_path)
    try:
        cursor = connection.execute(
            "INSERT INTO pending_messages (priority, sequence, payload, created_at) VALUES (?, ?, ?, ?)",
            (priority, sequence, json.dumps(item, ensure_ascii=False), time.time()),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def load_pending_messages(
    db_path: str | Path | None = None,
) -> list[tuple[int, int, dict]]:
    connection = open_pending_queue_db(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, priority, sequence, payload, created_at
            FROM pending_messages
            ORDER BY priority, sequence
            """
        ).fetchall()
    finally:
        connection.close()

    pending: list[tuple[int, int, dict]] = []
    for pending_id, priority, sequence, payload, created_at in rows:
        try:
            item = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            delete_pending_message(int(pending_id), db_path)
            continue
        if not isinstance(item, dict):
            delete_pending_message(int(pending_id), db_path)
            continue
        item["_pending_id"] = int(pending_id)
        item["_pending_created_at"] = float(created_at)
        item["_restored"] = True
        pending.append((int(priority), int(sequence), item))
    return pending


def delete_pending_message(
    pending_id: int,
    db_path: str | Path | None = None,
) -> None:
    connection = open_pending_queue_db(db_path)
    try:
        connection.execute("DELETE FROM pending_messages WHERE id = ?", (pending_id,))
        connection.commit()
    finally:
        connection.close()


def pending_message_count(db_path: str | Path | None = None) -> int:
    connection = open_pending_queue_db(db_path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM pending_messages").fetchone()
        return int(row[0]) if row else 0
    finally:
        connection.close()


def enqueue_persistent_message(priority: int, item: dict) -> int:
    sequence = next_sequence()
    pending_id = persist_pending_message(priority, sequence, item)
    queued_item = dict(item)
    queued_item["_pending_id"] = pending_id
    target_queue = message_queue if priority == 0 else normal_message_queue
    target_queue.put((priority, sequence, queued_item))
    return pending_id


def restore_pending_messages() -> int:
    global sequence_number
    pending = load_pending_messages()
    if not pending:
        return 0
    with sequence_lock:
        sequence_number = max(sequence_number, max(sequence for _priority, sequence, _item in pending))
    for queued_item in pending:
        priority = queued_item[0]
        target_queue = message_queue if priority == 0 else normal_message_queue
        target_queue.put(queued_item)
    return len(pending)


def message_max_age_seconds(mentioned: bool) -> int:
    if mentioned:
        return settings.mentioned_message_max_age_seconds
    return settings.normal_message_max_age_seconds


def is_message_too_old(
    event_time,
    mentioned: bool,
    *,
    fallback_time=None,
    now: float | None = None,
) -> bool:
    timestamp_value = event_time if event_time is not None else fallback_time
    try:
        event_timestamp = float(timestamp_value)
    except (TypeError, ValueError):
        return False
    max_age = message_max_age_seconds(mentioned)
    if max_age <= 0:
        return False
    current_time = time.time() if now is None else now
    return current_time - event_timestamp > max_age


def is_event_too_old(event: dict) -> bool:
    raw_message = event.get("message", "")
    mentioned = is_mentioned(settings.bot_qq, raw_message)
    return is_message_too_old(event.get("time"), mentioned)


def acquire_reply_slot(*, block: bool = True, reserve_slots: int = 0) -> bool:
    while True:
        now = time.time()
        with rate_limit_lock:
            reply_timestamps[:] = [
                timestamp for timestamp in reply_timestamps if now - timestamp < 60
            ]
            capacity = max(0, settings.max_replies_per_minute - reserve_slots)
            if len(reply_timestamps) < capacity:
                reply_timestamps.append(now)
                return True
            if not block:
                return False
            sleep_for = (
                max(1, 60 - (now - reply_timestamps[0]))
                if reply_timestamps
                else 60
            )
        time.sleep(sleep_for)


def wait_for_rate_limit() -> None:
    acquire_reply_slot()


def consider_chat_reply(
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
    if not auto_reply_enabled:
        print("Skip message: auto reply disabled", normalized)
        return ProcessingDecision(False, "auto reply disabled", has_context, tuple(sources))
    if not settings.chat_reply_enabled:
        reason = "not a direct question or help request" if has_context else "no knowledge context"
        print("Skip message:", reason, normalized)
        return ProcessingDecision(False, reason, has_context, tuple(sources))
    if settings.chat_allowed_group_ids and str(group_id) not in settings.chat_allowed_group_ids:
        return ProcessingDecision(False, "chat group not allowed", has_context, tuple(sources))
    social_celebration = bool(celebration_kind(normalized))
    if len(normalized) < 2:
        return ProcessingDecision(False, "too short for chat reply", has_context, tuple(sources))
    if (
        normalized.strip().lower() in BARE_CHAT_REACTIONS
        and not has_substantive_chat_context(chat_context)
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
        and not looks_like_direct_question(normalized)
        and normalized.strip().lower() not in BARE_CHAT_REACTIONS
    ):
        return ProcessingDecision(False, "short reaction fast-skip", has_context, tuple(sources))
    if mentions_other and not social_celebration:
        return ProcessingDecision(False, "chat directed at another member", has_context, tuple(sources))
    if looks_like_assignment_to_humans(normalized):
        return ProcessingDecision(False, "chat filtered assignment or bot meta", has_context, tuple(sources))
    lowered = normalized.lower()
    if any(cue in lowered for cue in CHAT_BLOCK_CUES):
        return ProcessingDecision(False, "chat filtered announcement, link, or meta", has_context, tuple(sources))
    if (
        looks_like_direct_question(normalized)
        and has_auto_reply_keyword(normalized)
        and not social_celebration
    ):
        reason = (
            "weak knowledge context for factual question"
            if has_context
            else "no knowledge context for factual question"
        )
        return ProcessingDecision(False, reason, has_context, tuple(sources))
    quota_reason = chat_reply_quota_reason(group_id)
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


def should_process_message(
    question: str,
    mentioned: bool,
    effective_question: str | None = None,
    followup_of: str = "",
    followup_scope: str = "",
    group_id: int = 0,
    chat_context: Sequence[str] = (),
    mentions_other: bool = False,
) -> ProcessingDecision:
    if mentioned and is_identity_question(question):
        return ProcessingDecision(True, "mentioned identity request", reply_mode="identity")

    normalized = question.strip()
    query_text = effective_question or normalized
    result = retrieve_knowledge(query_text, min(settings.max_context_chars, 1200))
    context = result.context
    sources = result.sources
    strong_match = is_strong_knowledge_match(
        result.top_score,
        result.query_coverage,
    )
    if not context or not strong_match:
        fallback_allowed = settings.llm_fallback_enabled and (
            mentioned or not settings.fallback_only_when_mentioned
        )
        if fallback_allowed:
            fallback_reason = (
                "weak-context llm fallback" if context else "llm fallback"
            )
            if mentioned:
                fallback_reason = "mentioned " + fallback_reason
            print("Fallback reply:", fallback_reason, normalized)
            return ProcessingDecision(
                should_reply=True,
                reason=fallback_reason,
                has_context=bool(context),
                sources=tuple(sources),
                effective_question=query_text,
                followup_of=followup_of,
                followup_scope=followup_scope,
                reply_mode="fallback",
                retrieval_score=result.top_score,
                retrieval_coverage=result.query_coverage,
                chat_context=tuple(chat_context),
            )

        if mentioned:
            print("Skip mentioned message: fallback disabled", normalized)
            return ProcessingDecision(
                should_reply=False,
                reason="no strong knowledge context and fallback disabled",
                has_context=bool(context),
                sources=tuple(sources),
                effective_question=query_text,
                followup_of=followup_of,
                followup_scope=followup_scope,
                retrieval_score=result.top_score,
                retrieval_coverage=result.query_coverage,
            )
        decision = consider_chat_reply(
            normalized,
            group_id=group_id,
            chat_context=chat_context,
            mentions_other=mentions_other,
            has_context=bool(context),
            sources=sources,
            query_text=query_text,
            followup_of=followup_of,
            followup_scope=followup_scope,
        )
        decision.retrieval_score = result.top_score
        decision.retrieval_coverage = result.query_coverage
        return decision

    if len(normalized) < 5 and not has_auto_reply_keyword(normalized) and not followup_of:
        print("Skip message: too short for auto reply", normalized)
        return ProcessingDecision(
            False,
            "too short for auto reply",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            retrieval_score=result.top_score,
            retrieval_coverage=result.query_coverage,
        )

    if mentioned:
        print("Mention reply: knowledge context found", normalized)
        return ProcessingDecision(
            True,
            "mentioned with strong knowledge context",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            "knowledge",
            (),
            result.top_score,
            result.query_coverage,
        )

    if not auto_reply_enabled:
        print("Skip message: auto reply disabled", normalized)
        return ProcessingDecision(
            False,
            "auto reply disabled",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            retrieval_score=result.top_score,
            retrieval_coverage=result.query_coverage,
        )

    if looks_like_assignment_to_humans(normalized):
        print("Skip message: looks like assignment to humans", normalized)
        return ProcessingDecision(
            False,
            "looks like assignment to humans",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            retrieval_score=result.top_score,
            retrieval_coverage=result.query_coverage,
        )

    if not looks_like_direct_question(normalized) or not has_auto_reply_keyword(normalized):
        decision = consider_chat_reply(
            normalized,
            group_id=group_id,
            chat_context=chat_context,
            mentions_other=mentions_other,
            has_context=True,
            sources=sources,
            query_text=query_text,
            followup_of=followup_of,
            followup_scope=followup_scope,
        )
        decision.retrieval_score = result.top_score
        decision.retrieval_coverage = result.query_coverage
        return decision

    if has_auto_reply_keyword(normalized):
        print("Auto reply: matched Squad keyword", normalized)
        return ProcessingDecision(
            True,
            "matched Squad keyword with strong context",
            True,
            tuple(sources),
            query_text,
            followup_of,
            followup_scope,
            "knowledge",
            (),
            result.top_score,
            result.query_coverage,
        )

    should_reply = should_auto_reply(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        message=query_text,
    )
    print("Auto reply router:", "YES" if should_reply else "NO", normalized)
    return ProcessingDecision(
        should_reply,
        "llm router accepted" if should_reply else "llm router rejected",
        True,
        tuple(sources),
        query_text,
        followup_of,
        followup_scope,
        "knowledge",
        (),
        result.top_score,
        result.query_coverage,
    )


def worker(work_queue: queue.PriorityQueue, lane: str) -> None:
    while True:
        priority, seq, item = work_queue.get()
        terminal = True
        try:
            question = str(item["question"])
            mentioned = bool(item["mentioned"])
            group_id = int(item["group_id"])
            user_id = item.get("user_id")
            sender_role = item.get("sender_role", "")
            event_time = item.get("time")
            command = get_admin_command(question)

            if is_restored_admin_command(item):
                print("Drop restored admin command", group_id, user_id, question)
                write_message_audit(
                    decision="ignored",
                    reason="restored admin command discarded",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue

            if is_message_too_old(
                event_time,
                mentioned,
                fallback_time=item.get("_pending_created_at"),
            ):
                age_limit = message_max_age_seconds(mentioned)
                message_kind = "mentioned" if mentioned else "normal"
                print("Drop queued event: too old", group_id, message_kind, age_limit)
                write_message_audit(
                    decision="ignored",
                    reason=f"queued {message_kind} message too old ({age_limit}s)",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=event_time,
                )
                continue

            if lane == "normal" and settings.normal_reply_delay_seconds > 0:
                time.sleep(settings.normal_reply_delay_seconds)

            if command:
                if not is_admin_user(user_id, sender_role):
                    print("Skip admin command: user not allowed", group_id, user_id, question)
                    write_message_audit(
                        decision="skipped",
                        reason="admin command denied",
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        mentioned=mentioned,
                        event_time=item.get("time"),
                    )
                    continue

                answer = answer_admin_command(command)
                if settings.dry_run:
                    print("Dry run admin answer:", group_id, answer)
                    write_message_audit(
                        decision="answered_dry_run",
                        reason=f"admin command {command}",
                        group_id=group_id,
                        user_id=user_id,
                        question=question,
                        mentioned=mentioned,
                        event_time=item.get("time"),
                    )
                    continue

                wait_for_rate_limit()
                send_group_msg(
                    settings.onebot_api_url,
                    group_id,
                    answer,
                    settings.onebot_access_token,
                )
                record_group_chat_message(group_id, settings.bot_qq, answer)
                print("Answered admin command", group_id, user_id, command)
                write_message_audit(
                    decision="answered",
                    reason=f"admin command {command}",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    event_time=item.get("time"),
                )
                continue

            item["chat_context"] = list(
                recent_group_chat_context(
                    group_id,
                    now=time.time(),
                    focus_sequence=int(item.get("chat_sequence") or 0),
                )
            )
            followup_match = followup_context_for(group_id, user_id, question, mentioned)
            effective_question = build_effective_question(question, followup_match)
            model_started = time.monotonic()
            decision = should_process_message(
                question,
                mentioned,
                effective_question=effective_question,
                followup_of=followup_match.state.last_question if followup_match else "",
                followup_scope=followup_match.scope if followup_match else "",
                group_id=group_id,
                chat_context=tuple(item.get("chat_context") or ()),
                mentions_other=bool(item.get("mentions_other")),
            )
            if not decision.should_reply:
                print("Skip message: model/router decided no reply", group_id, question)
                write_message_audit(
                    decision="skipped",
                    reason=decision.reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    followup_of=decision.followup_of,
                    followup_scope=decision.followup_scope,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=int((time.monotonic() - model_started) * 1000),
                    event_time=item.get("time"),
                )
                continue

            if decision.reply_mode == "chat":
                chat_queue.put((item, decision))
                terminal = False
                continue

            current_topic_key = topic_key(question, decision)
            if (
                not mentioned
                and decision.reply_mode == "knowledge"
                and is_topic_on_cooldown(group_id, current_topic_key)
            ):
                print("Skip message: recent topic cooldown", group_id, current_topic_key, question)
                write_message_audit(
                    decision="skipped",
                    reason="recent topic cooldown",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    followup_of=decision.followup_of,
                    followup_scope=decision.followup_scope,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=int((time.monotonic() - model_started) * 1000),
                    event_time=item.get("time"),
                )
                continue

            answer = answer_for_decision(question, decision, effective_question)
            model_latency_ms = int((time.monotonic() - model_started) * 1000)
            mention_user_id = response_mention_user_id(
                mentioned=mentioned,
                user_id=user_id,
                reply_mode=decision.reply_mode,
                question=question,
                mentioned_user_ids=tuple(item.get("mentioned_user_ids") or ()),
            )
            if settings.dry_run:
                print("Dry run answer:", group_id, answer)
                write_message_audit(
                    decision="answered_dry_run",
                    reason=decision.reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    mentioned=mentioned,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    followup_of=decision.followup_of,
                    followup_scope=decision.followup_scope,
                    reply_mode=decision.reply_mode,
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    mention_user_id=mention_user_id,
                    event_time=item.get("time"),
                )
                remember_conversation(group_id, user_id, question, decision)
                if decision.reply_mode == "knowledge":
                    mark_topic_replied(group_id, current_topic_key)
                continue

            wait_for_rate_limit()
            send_group_msg(
                settings.onebot_api_url,
                group_id,
                answer,
                settings.onebot_access_token,
                mention_user_id=mention_user_id,
            )
            record_group_chat_message(group_id, settings.bot_qq, answer)
            print("Answered group", group_id, "question", question)
            write_message_audit(
                decision="answered",
                reason=decision.reason,
                group_id=group_id,
                user_id=user_id,
                question=question,
                mentioned=mentioned,
                has_context=decision.has_context,
                sources=decision.sources,
                followup_of=decision.followup_of,
                followup_scope=decision.followup_scope,
                reply_mode=decision.reply_mode,
                retrieval_score=decision.retrieval_score,
                retrieval_coverage=decision.retrieval_coverage,
                model_latency_ms=model_latency_ms,
                mention_user_id=mention_user_id,
                event_time=item.get("time"),
            )
            remember_conversation(group_id, user_id, question, decision)
            if decision.reply_mode == "knowledge":
                mark_topic_replied(group_id, current_topic_key)
        except Exception as exc:
            terminal = False
            print(f"{lane} worker error:", repr(exc))
            write_message_audit(
                decision="error",
                reason=repr(exc),
                group_id=item.get("group_id"),
                user_id=item.get("user_id"),
                question=item.get("question", ""),
                mentioned=item.get("mentioned", False),
                followup_of="",
                followup_scope="",
                event_time=item.get("time"),
            )
        finally:
            pending_id = item.get("_pending_id")
            if terminal and pending_id is not None:
                try:
                    delete_pending_message(int(pending_id))
                except Exception as exc:
                    print("Pending queue acknowledge failed:", repr(exc))
            work_queue.task_done()


def chat_worker() -> None:
    while True:
        item, decision = chat_queue.get()
        terminal = True
        model_started = time.monotonic()
        try:
            question = str(item["question"])
            group_id = int(item["group_id"])
            user_id = item.get("user_id")
            event_time = item.get("time")
            social_event_kind = celebration_kind(question)
            mention_user_id = response_mention_user_id(
                mentioned=bool(item.get("mentioned")),
                user_id=user_id,
                reply_mode="chat",
                question=question,
                mentioned_user_ids=tuple(item.get("mentioned_user_ids") or ()),
            )

            if is_message_too_old(
                event_time,
                False,
                fallback_time=item.get("_pending_created_at"),
            ):
                write_message_audit(
                    decision="ignored",
                    reason="queued chat message too old",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            quota_reason = chat_reply_quota_reason(group_id)
            if quota_reason:
                write_message_audit(
                    decision="skipped",
                    reason=quota_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode="chat",
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            debounce_seconds = max(0.0, getattr(settings, "chat_reply_debounce_seconds", 2.0))
            if debounce_seconds:
                time.sleep(debounce_seconds)
            chat_sequence = int(item.get("chat_sequence") or 0)
            if group_chat_has_newer_user_message(group_id, chat_sequence):
                write_message_audit(
                    decision="skipped",
                    reason="chat candidate superseded",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            decision.chat_context = recent_group_chat_context(
                group_id,
                now=time.time(),
                focus_sequence=chat_sequence,
            )

            celebration_target_key = mention_user_id or "unknown"
            if social_event_kind and celebration_was_replied(
                group_id,
                celebration_target_key,
                social_event_kind,
            ):
                write_message_audit(
                    decision="skipped",
                    reason="celebration already acknowledged",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    chat_context=decision.chat_context,
                    mention_user_id=mention_user_id,
                    event_time=event_time,
                )
                continue

            # Scene analysis: extract conversation topic before generating reply
            scene_topic = analyze_scene(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                context=decision.chat_context,
            )
            enhanced_context = decision.chat_context
            if scene_topic:
                enhanced_context = (scene_topic,) + tuple(decision.chat_context)

            raw_answer = answer_chat(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                message=question,
                context=enhanced_context,
            )
            if is_chat_no_reply(raw_answer):
                model_latency_ms = int((time.monotonic() - model_started) * 1000)
                print("Chat generation: NO_REPLY", group_id, question)
                write_message_audit(
                    decision="skipped",
                    reason="chat generation declined",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode="chat",
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            accepted_reason = (
                f"{social_event_kind} celebration accepted"
                if social_event_kind
                else "chat generation accepted"
            )
            answer = finalize_model_answer(raw_answer, unsolicited=True)
            model_latency_ms = int((time.monotonic() - model_started) * 1000)
            if not answer:
                write_message_audit(
                    decision="skipped",
                    reason="chat generation failed",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            if group_chat_has_newer_user_message(group_id, chat_sequence):
                write_message_audit(
                    decision="skipped",
                    reason="chat superseded during generation",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            if is_message_too_old(
                event_time,
                False,
                fallback_time=item.get("_pending_created_at"),
            ):
                write_message_audit(
                    decision="skipped",
                    reason="chat became stale during generation",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            if settings.dry_run:
                print("Dry run chat answer:", group_id, answer)
                write_message_audit(
                    decision="answered_dry_run",
                    reason=accepted_reason,
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    has_context=decision.has_context,
                    sources=decision.sources,
                    reply_mode="chat",
                    retrieval_score=decision.retrieval_score,
                    retrieval_coverage=decision.retrieval_coverage,
                    model_latency_ms=model_latency_ms,
                    chat_context=decision.chat_context,
                    answer=answer,
                    mention_user_id=mention_user_id,
                    event_time=event_time,
                )
                mark_chat_replied(group_id)
                if social_event_kind:
                    mark_celebration_replied(
                        group_id,
                        celebration_target_key,
                        social_event_kind,
                    )
                continue

            if not acquire_reply_slot(block=False, reserve_slots=1):
                write_message_audit(
                    decision="skipped",
                    reason="chat global rate limit",
                    group_id=group_id,
                    user_id=user_id,
                    question=question,
                    reply_mode="chat",
                    model_latency_ms=model_latency_ms,
                    chat_context=decision.chat_context,
                    event_time=event_time,
                )
                continue

            send_group_msg(
                settings.onebot_api_url,
                group_id,
                answer,
                settings.onebot_access_token,
                mention_user_id=mention_user_id,
            )
            record_group_chat_message(group_id, settings.bot_qq, answer)
            mark_chat_replied(group_id)
            if social_event_kind:
                mark_celebration_replied(
                    group_id,
                    celebration_target_key,
                    social_event_kind,
                )
            print("Answered chat", group_id, question)
            write_message_audit(
                decision="answered",
                reason=accepted_reason,
                group_id=group_id,
                user_id=user_id,
                question=question,
                has_context=decision.has_context,
                sources=decision.sources,
                reply_mode="chat",
                retrieval_score=decision.retrieval_score,
                retrieval_coverage=decision.retrieval_coverage,
                model_latency_ms=model_latency_ms,
                chat_context=decision.chat_context,
                answer=answer,
                mention_user_id=mention_user_id,
                event_time=event_time,
            )
        except Exception as exc:
            terminal = False
            print("Chat worker error:", repr(exc))
            write_message_audit(
                decision="error",
                reason=repr(exc),
                group_id=item.get("group_id"),
                user_id=item.get("user_id"),
                question=item.get("question", ""),
                reply_mode="chat",
                chat_context=decision.chat_context,
                event_time=item.get("time"),
            )
        finally:
            pending_id = item.get("_pending_id")
            if terminal and pending_id is not None:
                try:
                    delete_pending_message(int(pending_id))
                except Exception as exc:
                    print("Chat pending queue acknowledge failed:", repr(exc))
            chat_queue.task_done()


class Handler(BaseHTTPRequestHandler):
    def _read_request_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            length = int(content_length)
            if length > 0:
                return self.rfile.read(length)

        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" not in transfer_encoding:
            return b""

        body = bytearray()
        while True:
            line = self.rfile.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                size = int(line.split(b";", 1)[0], 16)
            except ValueError:
                break
            if size == 0:
                while True:
                    trailer = self.rfile.readline()
                    if not trailer or trailer in {b"\r\n", b"\n", b""}:
                        break
                break
            body.extend(self.rfile.read(size))
            self.rfile.read(2)
        return bytes(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "chunks": len(kb.chunks),
                    "queued": (
                        message_queue.qsize()
                        + normal_message_queue.qsize()
                        + chat_queue.qsize()
                    ),
                    "priority_queued": message_queue.qsize(),
                    "normal_queued": normal_message_queue.qsize(),
                    "chat_queued": chat_queue.qsize(),
                    "pending": pending_message_count(),
                },
            )
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        raw_body = self._read_request_body()
        length = len(raw_body)
        try:
            text_body = raw_body.decode("utf-8")
            event = json.loads(text_body)
        except json.JSONDecodeError:
            try:
                parsed = parse_qs(text_body)
                event = {
                    key: values[0] if len(values) == 1 else values
                    for key, values in parsed.items()
                }
            except Exception:
                event = {}
            if not event:
                debug_path = Path("work/last_bad_onebot_payload.txt")
                debug_path.parent.mkdir(exist_ok=True)
                debug_path.write_text(
                    f"content-type: {self.headers.get('Content-Type', '')}\n"
                    f"transfer-encoding: {self.headers.get('Transfer-Encoding', '')}\n"
                    f"content-length: {length}\n\n"
                    f"{raw_body[:4000]!r}\n",
                    encoding="utf-8",
                )
                self._json(200, {"ok": True, "ignored": "bad payload"})
                return
        except UnicodeDecodeError:
            debug_path = Path("work/last_bad_onebot_payload.txt")
            debug_path.parent.mkdir(exist_ok=True)
            debug_path.write_bytes(raw_body[:4000])
            self._json(200, {"ok": True, "ignored": "bad encoding"})
            return

        if self.path == "/onebot":
            Path("work/last_onebot_raw.json").parent.mkdir(exist_ok=True)
            Path("work/last_onebot_raw.json").write_text(
                json.dumps(event, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if self.path == "/ask":
            question = str(event.get("question", "")).strip()
            self._json(200, {"ok": True, "answer": answer_question(question)})
            return

        if self.path != "/onebot":
            self._json(404, {"ok": False, "error": "not found"})
            return

        if event.get("message_type") != "group":
            print("Ignored event: not group", event.get("post_type"), event.get("message_type"))
            self._json(200, {"ok": True, "ignored": "not group message"})
            return

        group_id = event.get("group_id")
        if settings.allowed_group_ids and str(group_id) not in settings.allowed_group_ids:
            print("Ignored event: group not allowed", group_id)
            question, mentioned = extract_event_question(event)
            if mentioned:
                write_message_audit(
                    decision="ignored",
                    reason="group not allowed",
                    group_id=group_id,
                    user_id=event.get("user_id"),
                    question=question,
                    mentioned=mentioned,
                    event_time=event.get("time"),
                )
            self._json(200, {"ok": True, "ignored": "group not allowed"})
            return

        raw_message = event.get("message", "")
        text = extract_plain_text(raw_message)
        mentioned = is_mentioned(settings.bot_qq, raw_message)
        mentioned_user_ids = extract_mentioned_user_ids(settings.bot_qq, raw_message)
        user_id = event.get("user_id")
        if settings.bot_qq and str(user_id) == settings.bot_qq:
            self._json(200, {"ok": True, "ignored": "bot's own message"})
            return

        numeric_group_id = int(group_id)
        reply_message_id = extract_reply_message_id(raw_message)
        reply_target_user_id = ""
        reply_text = ""
        if reply_message_id:
            replied = find_group_chat_message(numeric_group_id, reply_message_id)
            if replied:
                reply_target_user_id = replied.user_id
                reply_text = replied.text
            else:
                reply_target_user_id, reply_text = get_message_info(
                    settings.onebot_api_url,
                    reply_message_id,
                    settings.onebot_access_token,
                    settings.onebot_message_lookup_timeout_seconds,
                )

        chat_sequence = record_group_chat_message(
            numeric_group_id,
            user_id,
            text,
            event.get("time"),
            message_id=str(event.get("message_id") or ""),
            reply_message_id=reply_message_id,
            reply_target_user_id=reply_target_user_id,
            reply_text=reply_text,
        )
        save_chat_history()

        # Check if message is too old AFTER recording context
        # Old messages are recorded for context but not processed for replies
        if is_event_too_old(event):
            write_message_audit(
                decision="ignored",
                reason="message too old",
                group_id=group_id,
                user_id=user_id,
                question=text,
                mentioned=mentioned,
                event_time=event.get("time"),
            )
            self._json(200, {"ok": True, "ignored": "message too old"})
            return

        continue_processing, mentioned, reply_reason = classify_reply_target(
            reply_message_id,
            reply_target_user_id,
            mentioned,
            settings.bot_qq,
        )
        if not continue_processing:
            write_message_audit(
                decision="ignored",
                reason=reply_reason,
                group_id=group_id,
                user_id=user_id,
                question=text,
                mentioned=mentioned,
                reply_message_id=reply_message_id,
                reply_target_user_id=reply_target_user_id,
                event_time=event.get("time"),
            )
            self._json(200, {"ok": True, "ignored": reply_reason})
            return

        try:
            context_now = float(event.get("time"))
        except (TypeError, ValueError):
            context_now = time.time()
        chat_context = recent_group_chat_context(
            numeric_group_id,
            now=context_now,
            focus_sequence=chat_sequence,
        )
        ok, question = should_respond(text, settings.command_prefix, settings.bot_qq, raw_message)
        print("Group event", group_id, "mentioned", mentioned, "queued", ok, "question", question)
        Path("work/last_onebot_event.json").parent.mkdir(exist_ok=True)
        Path("work/last_onebot_event.json").write_text(
            json.dumps(
                {
                    "group_id": group_id,
                    "raw_message": raw_message,
                    "text": text,
                    "mentioned": mentioned,
                    "reply_message_id": reply_message_id,
                    "reply_target_user_id": reply_target_user_id,
                    "reply_reason": reply_reason,
                    "queued": ok,
                    "question": question,
                    "mentioned_user_ids": list(mentioned_user_ids),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if not ok or not question:
            write_message_audit(
                decision="ignored",
                reason="no trigger",
                group_id=group_id,
                user_id=event.get("user_id"),
                question=question,
                mentioned=mentioned,
                event_time=event.get("time"),
            )
            self._json(200, {"ok": True, "ignored": "no trigger"})
            return

        priority = 0 if mentioned else 1
        item = {
            "group_id": group_id,
            "question": question,
            "mentioned": mentioned,
            "time": event.get("time"),
            "user_id": user_id,
            "sender_role": event.get("sender", {}).get("role", ""),
            "chat_context": list(chat_context),
            "mentions_other": bool(mentioned_user_ids),
            "mentioned_user_ids": list(mentioned_user_ids),
            "reply_message_id": reply_message_id,
            "reply_target_user_id": reply_target_user_id,
            "chat_sequence": chat_sequence,
        }
        try:
            enqueue_persistent_message(priority, item)
        except Exception as exc:
            print("Pending queue write failed:", repr(exc))
            write_message_audit(
                decision="error",
                reason=f"pending queue write failed: {exc!r}",
                group_id=group_id,
                user_id=event.get("user_id"),
                question=question,
                mentioned=mentioned,
                event_time=event.get("time"),
            )
            self._json(503, {"ok": False, "error": "queue unavailable"})
            return

        self._json(200, {"ok": True, "queued": True, "mentioned": mentioned})


def main() -> None:
    restored = restore_pending_messages()
    if restored:
        print(f"Restored pending messages: {restored}")
    loaded = load_chat_history()
    if loaded:
        print(f"Loaded chat history: {loaded} entries")
    threading.Thread(
        target=worker,
        args=(message_queue, "priority"),
        daemon=True,
    ).start()
    threading.Thread(
        target=worker,
        args=(normal_message_queue, "normal"),
        daemon=True,
    ).start()
    threading.Thread(target=chat_worker, daemon=True).start()
    server = ThreadingHTTPServer((settings.host, settings.port), Handler)
    print(f"Squad QQBot MVP listening on http://{settings.host}:{settings.port}")
    print(f"Knowledge chunks: {len(kb.chunks)}")
    print(
        f"Allowed groups: {','.join(settings.allowed_group_ids) or 'all'}, "
        f"max replies/min: {settings.max_replies_per_minute}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
