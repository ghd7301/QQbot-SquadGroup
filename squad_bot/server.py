from __future__ import annotations

import hashlib
import hmac
import json
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence
from urllib.parse import parse_qs

from . import pending_store
from .chat_history import ChatHistoryState
from .chat_scene import ChatSceneState
from .message_fragments import (
    FragmentAggregator,
    classify_audience,
    items_compatible,
    message_ids,
)
from . import message_router, semantic_routing, worker_handlers
from .config import settings
from .chat_memory import ChatMemoryManager, ChatMemoryStore, MemoryHit, MemoryMessage, redact_for_model
from .embedding import build_embedding_provider
from .fact_guard import (
    candidate_knowledge_segments,
    normalize_chinese_number,
    precise_fact_tokens,
    unsupported_fallback_precise_facts,
)
from .knowledge import ContextResult, KnowledgeBase
from .knowledge_routing import KnowledgeRoutingService, attach_result
from .runtime_dependencies import RuntimeDependencies
from .worker_runtime import PendingItemLifecycle, normal_lane_should_yield
from .models import (
    ConversationState,
    FollowupMatch,
    GroupChatMessage,
    GroupChatScene,
    MemoryProbeResult,
    MessageFragmentBuffer,
    PendingFailureResult,
    ProcessingDecision,
)
from .llm import (
    MessagePlan,
    SemanticTopicCandidate,
    SubjectCandidate,
    analyze_chat_scene,
    answer_chat,
    ask_fallback_llm,
    ask_llm,
    classify_bot_fragment_prefix,
    is_chat_no_reply,
    is_provider_refusal_text,
    normalize_model_answer,
    plan_group_message,
    review_candidate_reply,
    rewrite_contextual_question,
    verify_bot_capability,
)
from .onebot import (
    extract_content_segments,
    extract_mentioned_user_ids,
    extract_context_text,
    extract_plain_text,
    extract_reply_message_id,
    get_message_info,
    is_mentioned,
    send_group_msg,
    should_respond,
)
from .planner_health import SemanticPlannerHealth


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
chat_history_state = ChatHistoryState()
chat_history_lock = chat_history_state.lock
group_chat_history = chat_history_state.messages
chat_message_sequence = 0
chat_reply_lock = threading.Lock()
group_send_locks_lock = threading.Lock()
group_send_locks: dict[int, threading.Lock] = {}
chat_scene_state = ChatSceneState()
chat_scene_lock = chat_scene_state.lock
group_chat_scenes = chat_scene_state.scenes
chat_scene_requested_sequence = chat_scene_state.requested_sequences
chat_scene_pending_messages = chat_scene_state.pending_messages
chat_scene_running = chat_scene_state.running
hostile_reply_lock = threading.Lock()
hostile_reply_history: dict[tuple[int, str], list[float]] = {}
memory_clear_confirmations: dict[tuple[int, str], float] = {}
fragment_aggregator = FragmentAggregator()
fragment_condition = fragment_aggregator.condition
group_fragment_buffers = fragment_aggregator.buffers
ready_fragment_buffers = fragment_aggregator.ready
chat_memory_manager: ChatMemoryManager | None = None
chat_history_save_event = threading.Event()
knowledge_gap_lock = threading.Lock()
recent_knowledge_gap_queries: dict[str, float] = {}
semantic_planner_health = SemanticPlannerHealth()
runtime_dependencies = RuntimeDependencies(globals())

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
    "聊天记忆状态": "memory_status",
    "记忆状态": "memory_status",
    "memory status": "memory_status",
    "暂停聊天记忆": "memory_pause",
    "memory pause": "memory_pause",
    "恢复聊天记忆": "memory_resume",
    "memory resume": "memory_resume",
    "重建聊天记忆": "memory_rebuild",
    "memory rebuild": "memory_rebuild",
    "清空本群聊天记忆": "memory_clear_request",
    "清空本群聊天记忆 确认": "memory_clear_confirm",
    "最近知识未命中": "knowledge_gaps",
    "知识未命中": "knowledge_gaps",
    "knowledge gaps": "knowledge_gaps",
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
    model_name: str = "",
    reply_message_id: str = "",
    reply_target_user_id: str = "",
    chat_context: Sequence[str] = (),
    scene_context: str = "",
    answer: str = "",
    mention_user_id: str = "",
    semantic_intent: str = "",
    semantic_topic: str = "",
    implicit_meaning: str = "",
    capability: str = "none",
    semantic_confidence: float = 0.0,
    topic_candidates: Sequence[SemanticTopicCandidate] = (),
    subject_candidates: Sequence[SubjectCandidate] = (),
    subject_ambiguity: str = "unknown",
    bot_involvement: str = "uncertain",
    reply_perspective: str = "neutral",
    semantic_audience: str = "unclear",
    participation_role: str = "uncertain",
    plan_context_revision: int = 0,
    plan_scene_version: int = 0,
    related_message_ids: Sequence[str] = (),
    semantic_replan_count: int = 0,
    semantic_replan_reason: str = "",
    planner_status: str = "not_run",
    planner_latency_ms: int = 0,
    memory_query: str = "",
    memory_hit_count: int = 0,
    memory_retrieval_attempted: bool = False,
    memory_retrieval_mode: str = "",
    memory_candidate_count: int = 0,
    memory_rejection_reason: str = "",
    recent_context_candidate_count: int = 0,
    recent_context_selected_count: int = 0,
    recent_context_chars: int = 0,
    memory_context_chars: int = 0,
    context_deduplicated_count: int = 0,
    recent_context_selected_ids: Sequence[str] = (),
    memory_selected_chunk_ids: Sequence[int] = (),
    memory_selected_by_planner: bool = False,
    self_history_candidate_count: int = 0,
    self_history_selected_count: int = 0,
    self_history_chars: int = 0,
    self_history_selected_message_ids: Sequence[str] = (),
    self_history_reasons: Sequence[str] = (),
    bot_message_id: str = "",
    generated_for_message_ids: Sequence[str] = (),
    turn_id: str = "",
    event_time=None,
) -> None:
    try:
        total_latency_ms = max(0, int((time.time() - float(event_time)) * 1000))
    except (TypeError, ValueError):
        total_latency_ms = 0
    try:
        scene_payload = json.loads(scene_context) if scene_context else {}
    except (json.JSONDecodeError, TypeError):
        scene_payload = {}
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
        "knowledge_strength": (
            "strong"
            if (
                float(retrieval_score)
                >= float(getattr(settings, "knowledge_strong_min_score", 0.18))
                and float(retrieval_coverage)
                >= float(getattr(settings, "knowledge_strong_min_coverage", 0.6))
            )
            else ("weak" if sources or retrieval_score or retrieval_coverage else "none")
        ),
        "model_latency_ms": int(model_latency_ms),
        "total_latency_ms": total_latency_ms,
        "model": model_name,
        "reply_message_id": str(reply_message_id),
        "reply_target_user_id": str(reply_target_user_id),
        "chat_context": list(chat_context),
        "scene_context": scene_context,
        "answer": answer,
        "mention_user_id": str(mention_user_id or ""),
        "semantic_intent": semantic_intent,
        "semantic_topic": semantic_topic,
        "implicit_meaning": implicit_meaning,
        "capability": capability,
        "semantic_confidence": round(float(semantic_confidence), 4),
        "topic_candidates": [
            {
                "key": candidate.key,
                "label": candidate.label,
                "confidence": round(candidate.confidence, 4),
                "basis": candidate.basis,
                "anchor_message_ids": list(candidate.anchor_message_ids),
            }
            for candidate in topic_candidates
        ],
        "subject_candidates": [
            {
                "entity_type": candidate.entity_type,
                "entity_id": candidate.entity_id,
                "label": candidate.label,
                "confidence": round(candidate.confidence, 4),
                "evidence_message_ids": list(candidate.evidence_message_ids),
            }
            for candidate in subject_candidates
        ],
        "subject_ambiguity": subject_ambiguity,
        "bot_involvement": bot_involvement,
        "reply_perspective": reply_perspective,
        "semantic_audience": semantic_audience,
        "participation_role": participation_role,
        "plan_context_revision": int(plan_context_revision),
        "plan_scene_version": int(plan_scene_version),
        "related_message_ids": list(related_message_ids),
        "semantic_replan_count": int(semantic_replan_count),
        "semantic_replan_reason": semantic_replan_reason,
        "planner_status": planner_status,
        "planner_lane": (
            "addressed"
            if mentioned
            or (
                bool(str(getattr(settings, "bot_qq", "") or ""))
                and str(reply_target_user_id or "")
                == str(getattr(settings, "bot_qq", "") or "")
            )
            else "unsolicited"
        ),
        "planner_circuit_open": planner_status == "circuit_open",
        "planner_latency_ms": int(planner_latency_ms),
        "scene_version": int(scene_payload.get("version") or 0) if isinstance(scene_payload, dict) else 0,
        "scene_updated_through_sequence": (
            int(scene_payload.get("updated_through_sequence") or 0)
            if isinstance(scene_payload, dict)
            else 0
        ),
        "memory_query": memory_query,
        "memory_hit_count": int(memory_hit_count),
        "memory_retrieval_attempted": bool(memory_retrieval_attempted),
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_candidate_count": int(memory_candidate_count),
        "memory_rejection_reason": memory_rejection_reason,
        "recent_context_candidate_count": int(recent_context_candidate_count),
        "recent_context_selected_count": int(recent_context_selected_count),
        "recent_context_chars": int(recent_context_chars),
        "memory_context_chars": int(memory_context_chars),
        "context_deduplicated_count": int(context_deduplicated_count),
        "recent_context_selected_ids": list(recent_context_selected_ids),
        "memory_selected_chunk_ids": [int(value) for value in memory_selected_chunk_ids],
        "memory_selected_by_planner": bool(memory_selected_by_planner),
        "self_history_candidate_count": int(self_history_candidate_count),
        "self_history_selected_count": int(self_history_selected_count),
        "self_history_chars": int(self_history_chars),
        "self_history_selected_message_ids": list(self_history_selected_message_ids),
        "self_history_reasons": list(self_history_reasons),
        "bot_message_id": str(bot_message_id or ""),
        "generated_for_message_ids": list(generated_for_message_ids),
        "turn_id": str(turn_id or ""),
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
    return bool(
        item.get("_restored")
        and is_admin_user(item.get("user_id"), item.get("sender_role", ""))
        and get_admin_command(str(item.get("question", "")))
    )


def is_admin_user(user_id, sender_role: str = "") -> bool:
    admin_ids = tuple(getattr(settings, "admin_qq_ids", ()))
    return bool(admin_ids) and str(user_id) in admin_ids


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


def answer_admin_command(command: str, *, group_id: int = 0, user_id: str = "") -> str:
    global auto_reply_enabled
    if command == "reload":
        with kb_lock:
            count = kb.reload()
            stats = kb.last_reload_stats
        return (
            f"知识库已重载，共 {count} 个片段；新增 {stats.added}，修改 {stats.changed}，"
            f"删除 {stats.removed}，复用 {stats.reused}。"
        )
    if command == "health":
        auto_status = "开" if auto_reply_enabled else "关"
        chat_status = "开" if settings.chat_reply_enabled and auto_reply_enabled else "关"
        queued = message_queue.qsize() + normal_message_queue.qsize() + chat_queue.qsize()
        scene_count, _ = chat_scene_state.counts()
        return (
            f"服务正常。知识片段 {len(kb.chunks)} 个，队列 {queued} 条，"
            f"自动回复{auto_status}，闲聊{chat_status}，场景快照 {scene_count} 个，"
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
    if command == "memory_status":
        if not chat_memory_manager:
            return "聊天记忆没有启用。"
        status = chat_memory_manager.status()
        state = "暂停" if status["paused"] else "运行"
        return (
            f"聊天记忆{state}，消息 {status['messages']} 条，片段 {status['chunks']} 个，"
            f"话题关系 {status.get('topic_relations', 0)} 条，"
            f"待索引 {status['queued']} 条，向量方式 {status['provider']}。"
        )
    if command == "memory_pause":
        if not chat_memory_manager:
            return "聊天记忆没有启用。"
        chat_memory_manager.paused.set()
        return "聊天记忆索引已暂停，现有短期上下文仍然可用。"
    if command == "memory_resume":
        if not chat_memory_manager:
            return "聊天记忆没有启用。"
        chat_memory_manager.paused.clear()
        return "聊天记忆索引已恢复。"
    if command == "memory_rebuild":
        if not chat_memory_manager:
            return "聊天记忆没有启用。"
        chat_memory_manager.enqueue_rebuild(group_id)
        return "已安排重建本群聊天记忆索引。"
    if command == "memory_clear_request":
        memory_clear_confirmations[(group_id, str(user_id))] = time.time() + 60
        return "这是不可恢复操作。确认要清空本群聊天记忆，请发送：清空本群聊天记忆 确认"
    if command == "memory_clear_confirm":
        if not chat_memory_manager:
            return "聊天记忆没有启用。"
        key = (group_id, str(user_id))
        expires_at = memory_clear_confirmations.pop(key, 0)
        if time.time() > expires_at:
            return "确认已失效，请先发送：清空本群聊天记忆"
        chat_memory_manager.store.clear_group(group_id)
        return "本群聊天记忆已清空。"
    if command == "knowledge_gaps":
        entries = recent_knowledge_gap_entries()
        if not entries:
            return "最近没有记录到知识检索缺口。"
        lines = []
        for entry in entries:
            query = str(entry.get("query") or "")[:36]
            missing = "、".join(entry.get("missing_tokens") or ())[:50]
            lines.append(f"{query}（缺：{missing or '无明确词'}）")
        return "最近知识未命中：\n" + "\n".join(lines)
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


def followup_context_for(
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
        state = load_conversation_turn_by_bot_message_id(
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
        persist_conversation_turn(group_id, state, db_path=db_path)
    except Exception as exc:
        print("Persist conversation turn failed:", repr(exc))


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
    mentioned_bot: bool = False,
    mentioned_user_ids: Sequence[str] = (),
    display_name: str = "",
    generated_for_message_ids: Sequence[str] = (),
    turn_id: str = "",
    reply_mode: str = "",
    semantic_topic: str = "",
    received_time=None,
    content_segments: Sequence[dict[str, str]] = (),
    message_status: str = "active",
) -> int:
    global chat_message_sequence
    with chat_history_lock:
        entry = chat_history_state.record(
            group_id,
            user_id,
            text,
            event_time,
            context_seconds=settings.chat_context_seconds,
            context_messages=settings.chat_context_messages,
            message_id=message_id,
            reply_message_id=reply_message_id,
            reply_target_user_id=reply_target_user_id,
            reply_text=reply_text,
            mentioned_bot=mentioned_bot,
            mentioned_user_ids=mentioned_user_ids,
            display_name=display_name,
            generated_for_message_ids=generated_for_message_ids,
            turn_id=turn_id,
            reply_mode=reply_mode,
            semantic_topic=semantic_topic,
            received_time=received_time,
            content_segments=content_segments,
            message_status=message_status,
        )
        if entry is None:
            return 0
        chat_message_sequence = chat_history_state.sequence
    if chat_memory_manager:
        speaker_id = stable_member_id(group_id, entry.user_id)
        chat_memory_manager.enqueue(
            MemoryMessage(
                group_id=group_id,
                message_id=entry.message_id or f"local:{group_id}:{entry.sequence}",
                speaker_id=speaker_id,
                display_name=entry.display_name if speaker_id != "bot" else "机器人",
                speaker_role="bot" if speaker_id == "bot" else "member",
                text=entry.text,
                event_time=entry.timestamp,
                reply_message_id=entry.reply_message_id,
                reply_speaker_id=stable_member_id(group_id, entry.reply_target_user_id),
                quoted_text=entry.reply_text,
                mentions=tuple(
                    stable_member_id(group_id, value)
                    for value in entry.mentioned_user_ids
                ),
                generated_for_message_ids=entry.generated_for_message_ids,
                turn_id=entry.turn_id,
                reply_mode=entry.reply_mode,
                semantic_topic=entry.semantic_topic,
                sequence=entry.sequence,
                received_time=entry.received_time,
                content_segments=entry.content_segments,
                message_status=entry.message_status,
            )
        )
    return entry.sequence


def find_group_chat_message(group_id: int, message_id: str) -> GroupChatMessage | None:
    return chat_history_state.find(group_id, message_id)


def resolve_reply_message_context(
    group_id: int,
    message_id: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    target = str(message_id or "").strip()
    if not target:
        return "", ""
    replied = find_group_chat_message(group_id, target)
    if replied:
        return replied.user_id, replied.text
    sender_id, text = get_message_info(
        settings.onebot_api_url,
        target,
        settings.onebot_access_token,
        settings.onebot_message_lookup_timeout_seconds,
    )
    if sender_id:
        return sender_id, text
    turn = load_conversation_turn_by_bot_message_id(
        group_id,
        target,
        db_path=db_path,
    )
    if turn:
        return settings.bot_qq, turn.last_answer
    return "", text


def stable_member_id(group_id: int, user_id: str) -> str:
    user_key = str(user_id or "").strip()
    if user_key == settings.bot_qq:
        return "bot"
    if not user_key:
        return "unknown_member"
    secret = (
        getattr(settings, "member_id_secret", "")
        or getattr(settings, "onebot_access_token", "")
        or settings.bot_qq
        or "local-member-id"
    )
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{group_id}:{user_key}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:10]
    return f"member_{digest}"


def _context_message_payload(
    group_id: int,
    item: GroupChatMessage,
    *,
    current: bool,
) -> dict:
    speaker_id = stable_member_id(group_id, item.user_id)
    payload: dict = {
        "current": current,
        "message_id": item.message_id,
        "sequence": item.sequence,
        "event_time": item.timestamp,
        "received_time": item.received_time,
        "message_status": item.message_status,
        "speaker": {
            "id": speaker_id,
            "role": "bot" if speaker_id == "bot" else "member",
            "is_self": speaker_id == "bot",
            "display_name": (
                item.display_name
                if speaker_id != "bot"
                else "机器人"
            ),
        },
        "text": item.text,
        "content_segments": list(item.content_segments),
        "mentions": [
            stable_member_id(group_id, user_id)
            for user_id in item.mentioned_user_ids
        ],
        "mentions_bot": item.mentioned_bot,
        "generated_for_message_ids": list(item.generated_for_message_ids),
        "turn_id": item.turn_id,
        "reply_mode": item.reply_mode,
        "semantic_topic": item.semantic_topic,
    }
    if item.reply_message_id:
        target_id = stable_member_id(group_id, item.reply_target_user_id)
        payload["reply_to"] = {
            "message_id": item.reply_message_id,
            "speaker_id": target_id,
            "speaker_role": "bot" if target_id == "bot" else "member",
            "quoted_text": item.reply_text,
        }
    else:
        payload["reply_to"] = None
    return payload


def recent_group_chat_context(
    group_id: int,
    *,
    now: float | None = None,
    context_seconds: int | None = None,
    max_messages: int | None = None,
    focus_sequence: int = 0,
    through_sequence: int = 0,
) -> tuple[str, ...]:
    current_time = time.time() if now is None else now
    window = settings.chat_context_seconds if context_seconds is None else context_seconds
    limit = settings.chat_context_messages if max_messages is None else max_messages
    selected = chat_history_state.recent(
        group_id,
        now=current_time,
        context_seconds=window,
        max_messages=limit,
        focus_sequence=focus_sequence,
        through_sequence=through_sequence,
    )
    return tuple(
        json.dumps(
            _context_message_payload(
                group_id,
                item,
                current=item.sequence == focus_sequence,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in selected
    )


def current_group_chat_scene(
    group_id: int,
    *,
    focus_sequence: int = 0,
    now: float | None = None,
    stale_seconds: int | None = None,
) -> str:
    current_time = time.time() if now is None else now
    max_age = (
        getattr(settings, "chat_scene_stale_seconds", 600)
        if stale_seconds is None
        else stale_seconds
    )
    return chat_scene_state.current_summary(
        group_id,
        focus_sequence=focus_sequence,
        now=current_time,
        stale_seconds=max_age,
    )


def chat_scene_enabled_for_group(group_id: int) -> bool:
    if not auto_reply_enabled or not settings.chat_reply_enabled:
        return False
    if not getattr(settings, "chat_scene_enabled", True):
        return False
    return not settings.chat_allowed_group_ids or str(group_id) in settings.chat_allowed_group_ids


def _finish_chat_scene_update(group_id: int) -> None:
    chat_scene_state.finish(group_id)


def _chat_scene_update_loop(group_id: int) -> None:
    while True:
        debounce = max(0.0, getattr(settings, "chat_scene_debounce_seconds", 3.0))
        if debounce:
            time.sleep(debounce)

        existing = chat_scene_state.scene(group_id)
        last_updated = existing.updated_at if existing else 0.0
        min_interval = max(
            0.0,
            getattr(settings, "chat_scene_update_interval_seconds", 30.0),
        )
        wait_seconds = min_interval - (time.time() - last_updated)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        target_sequence, previous = chat_scene_state.begin_update(group_id)
        context = recent_group_chat_context(
            group_id,
            now=time.time(),
            focus_sequence=target_sequence,
            through_sequence=target_sequence,
        )
        min_messages = max(1, getattr(settings, "chat_scene_min_messages", 3))
        if len(context) < min_messages:
            _finish_chat_scene_update(group_id)
            return

        summary = analyze_chat_scene(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=getattr(settings, "chat_scene_model", settings.llm_model),
            context=context,
            previous_scene=previous.summary if previous else "",
            timeout=max(1, getattr(settings, "chat_scene_timeout_seconds", 30)),
        )
        if summary:
            chat_scene_state.set_scene(
                group_id,
                summary=summary,
                updated_at=time.time(),
                sequence=target_sequence,
            )
            print("Updated chat scene", group_id, target_sequence)
        else:
            print("Chat scene update failed", group_id, target_sequence)

        if not chat_scene_state.should_continue(
            group_id,
            min_messages=min_messages,
        ):
            return


def schedule_chat_scene_update(group_id: int, sequence: int) -> bool:
    if not sequence or not chat_scene_enabled_for_group(group_id):
        return False
    min_messages = max(1, getattr(settings, "chat_scene_min_messages", 3))
    if not chat_scene_state.request_update(
        group_id,
        sequence,
        min_messages=min_messages,
    ):
        return False
    threading.Thread(
        target=_chat_scene_update_loop,
        args=(group_id,),
        daemon=True,
        name=f"chat-scene-{group_id}",
    ).start()
    return True


def has_substantive_chat_context(chat_context: Sequence[str]) -> bool:
    for line in chat_context:
        text = line.split("：", 1)[-1].strip().lower()
        if text and text not in BARE_CHAT_REACTIONS:
            return True
    return False


def group_chat_has_newer_user_message(group_id: int, sequence: int) -> bool:
    return chat_history_state.has_newer_user_message(
        group_id,
        sequence,
        bot_user_id=settings.bot_qq,
    )


def latest_group_user_sequence(group_id: int) -> int:
    return chat_history_state.latest_user_sequence(
        group_id,
        bot_user_id=settings.bot_qq,
    )


def locked_send_context_change(
    group_id: int,
    reviewed_revision: int,
    item: dict,
) -> tuple[bool, tuple[str, ...], str]:
    """Classify messages that arrived after review without calling a model under the send lock."""
    bot_qq = str(settings.bot_qq or "")
    original_user_id = str(item.get("user_id") or "")
    trigger_message_ids = set(_message_ids(item))
    direct_to_bot = bool(
        item.get("mentioned")
        or (
            bot_qq
            and str(item.get("reply_target_user_id") or "") == bot_qq
        )
    )
    with chat_history_lock:
        delta = tuple(
            message
            for message in group_chat_history.get(int(group_id), ())
            if message.sequence > reviewed_revision
            and message.user_id != bot_qq
            and message.message_status == "active"
        )
    if not delta:
        return False, (), "unchanged"

    delta_ids = tuple(
        message.message_id or f"sequence:{message.sequence}"
        for message in delta
    )
    human_thread_by_sender: dict[str, tuple[str, float]] = {}
    continuation_window = max(
        0.0,
        float(getattr(settings, "message_fragment_max_wait_seconds", 8)),
    )

    for message in delta:
        reply_target = str(message.reply_target_user_id or "")
        if message.reply_message_id:
            if not reply_target:
                return True, delta_ids, "new reply target could not be resolved"
            if reply_target == bot_qq or message.reply_message_id in trigger_message_ids:
                return True, delta_ids, "new message directly extends or answers the bot turn"
            human_thread_by_sender[message.user_id] = (
                reply_target,
                message.received_time,
            )
            continue
        if message.mentioned_bot:
            if message.user_id == original_user_id:
                return True, delta_ids, "original sender sent another message to the bot"
            continue
        if message.mentioned_user_ids:
            human_thread_by_sender[message.user_id] = (
                str(message.mentioned_user_ids[0]),
                message.received_time,
            )
            continue
        if message.user_id == original_user_id:
            human_thread = human_thread_by_sender.get(message.user_id)
            if (
                human_thread
                and human_thread[0] != bot_qq
                and 0 <= message.received_time - human_thread[1] <= continuation_window
            ):
                human_thread_by_sender[message.user_id] = (
                    human_thread[0],
                    message.received_time,
                )
                continue
            return True, delta_ids, "original sender added an ambiguous follow-up"
        if not direct_to_bot:
            return True, delta_ids, "unsolicited reply has a newer ambiguous group message"

    return False, delta_ids, "only unrelated directed messages arrived"


def validate_locked_send(
    group_id: int,
    item: dict,
    reviewed_revision: int,
    *,
    check_context: bool = True,
) -> tuple[bool, str, str]:
    """Return whether a candidate may send, a blocking reason, and an audit note."""
    if message_already_covered_by_bot(group_id, item.get("message_id")):
        return False, "message covered while waiting to send", ""
    if not check_context:
        return True, "", ""
    invalidated, delta_message_ids, relation = locked_send_context_change(
        group_id,
        reviewed_revision,
        item,
    )
    delta_text = ",".join(delta_message_ids)
    if invalidated:
        return (
            False,
            "context invalidated while waiting for group send lock: "
            f"{relation}; delta={delta_text}",
            "",
        )
    note = (
        f"send-lock context preserved: {relation}; delta={delta_text}"
        if delta_message_ids
        else ""
    )
    return True, "", note


def unsafe_or_repeated_reply(group_id: int, answer: str, *, limit: int = 10) -> str:
    if re.search(r"\bmember_[0-9a-f]{6,}\b", answer, flags=re.I):
        return "internal member id leaked"
    normalized = re.sub(r"[\W_]+", "", answer.lower())
    if len(normalized) < 6:
        return ""
    with chat_history_lock:
        recent_bot_answers = [
            item.text
            for item in group_chat_history.get(group_id, ())
            if item.user_id == settings.bot_qq
        ][-limit:]
    for previous in recent_bot_answers:
        previous_normalized = re.sub(r"[\W_]+", "", previous.lower())
        if normalized == previous_normalized:
            return "duplicate recent bot reply"
    return ""


def is_recent_duplicate_group_message(
    group_id: int,
    text: str,
    *,
    focus_sequence: int,
    event_time=None,
    window_seconds: int = 60,
) -> bool:
    normalized = re.sub(r"[\W_]+", "", str(text or "").lower())
    if len(normalized) < 4 or focus_sequence <= 0:
        return False
    try:
        current_time = float(event_time)
    except (TypeError, ValueError):
        current_time = time.time()
    with chat_history_lock:
        for item in reversed(group_chat_history.get(group_id, ())):
            if item.sequence >= focus_sequence:
                continue
            if current_time - item.timestamp > max(0, window_seconds):
                break
            previous = re.sub(r"[\W_]+", "", item.text.lower())
            if previous == normalized:
                return True
    return False


def reply_deadline(event_time, mentioned: bool) -> float:
    total = (
        getattr(settings, "mentioned_reply_total_timeout_seconds", 15)
        if mentioned
        else getattr(settings, "normal_reply_total_timeout_seconds", 10)
    )
    try:
        elapsed = max(0.0, time.time() - float(event_time))
    except (TypeError, ValueError):
        elapsed = 0.0
    return time.monotonic() + max(0.0, float(total) - elapsed)


def remaining_reply_timeout(
    deadline: float,
    *,
    cap: int,
    reserve: int = 0,
) -> int:
    remaining = deadline - time.monotonic() - max(0, reserve)
    if remaining < 1:
        return 0
    return max(1, min(int(cap), int(remaining)))


def review_and_refresh_answer(
    *,
    question: str,
    answer: str,
    decision: ProcessingDecision,
    group_id: int,
    mentioned: bool,
    admin: bool,
    deadline: float,
    baseline_revision: int | None = None,
    target_item: dict | None = None,
) -> tuple[str, str, int]:
    original_context = tuple(decision.chat_context)
    candidate = answer
    regenerated = decision.reply_regenerated

    while True:
        latest_revision = latest_group_user_sequence(group_id)
        review_mode = getattr(settings, "final_reply_review_mode", "adaptive")
        risky_intents = {
            "banter_at_bot",
            "control_attempt",
            "third_party_attack",
            "genuine_criticism",
            "hostile_abuse",
            "action",
            "unclear",
        }
        context_changed = baseline_revision is None or latest_revision > baseline_revision
        context_change_note = ""
        if (
            context_changed
            and baseline_revision is not None
            and target_item
            and target_item.get("user_id")
            and target_item.get("chat_sequence")
        ):
            context_changed, delta_ids, relation = locked_send_context_change(
                group_id,
                baseline_revision,
                target_item,
            )
            if not context_changed:
                rendered_ids = ",".join(delta_ids)
                context_change_note = (
                    f"adaptive final review skipped: {relation}"
                    + (f"; delta={rendered_ids}" if rendered_ids else "")
                )
        requires_model_review = (
            review_mode == "always"
            or context_changed
            or regenerated
            or bool(decision.risk_flags)
            or decision.semantic_intent in risky_intents
            or (
                bool(decision.self_history_context)
                and decision.bot_involvement in {"subject", "participant", "uncertain"}
            )
            or (
                decision.reply_mode == "fallback"
                and (
                    decision.semantic_audience == "unclear"
                    or decision.participation_role == "uncertain"
                    or decision.semantic_confidence < 0.8
                    or (
                        decision.semantic_intent == "knowledge"
                        and not is_strong_knowledge_match(
                            decision.retrieval_score,
                            decision.retrieval_coverage,
                        )
                    )
                )
            )
            or (
                decision.reply_mode == "chat"
                and decision.reply_perspective in {"first_person", "neutral"}
            )
            or (
                decision.reply_mode == "chat"
                and (
                    decision.semantic_intent != "normal_chat"
                    or decision.semantic_confidence < 0.8
                )
            )
        )
        if not requires_model_review:
            unsafe_reason = unsafe_or_repeated_reply(group_id, candidate)
            if unsafe_reason:
                return "", unsafe_reason, latest_revision
            return (
                candidate,
                context_change_note or "adaptive final review skipped",
                latest_revision,
            )
        latest_context = recent_group_chat_context(
            group_id,
            now=time.time(),
            focus_sequence=int((target_item or {}).get("chat_sequence") or 0),
        )
        review_timeout = remaining_reply_timeout(
            deadline,
            cap=getattr(settings, "final_reply_review_timeout_seconds", 4),
        )
        if not review_timeout:
            return "", "reply deadline exhausted before final review", latest_revision
        review = review_candidate_reply(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=getattr(settings, "final_reply_review_model", settings.chat_model),
            original_message=question,
            candidate_reply=candidate,
            original_context=original_context,
            latest_context=latest_context,
            self_history_context=decision.self_history_context,
            reply_mode=decision.reply_mode,
            mentioned=mentioned,
            topic_summary=decision.semantic_topic,
            semantic_context=semantic_context_for_decision(decision),
            original_message_ids=_message_ids(target_item or {}),
            knowledge_sources=decision.sources,
            candidate_knowledge_context=(
                decision.knowledge_result.context
                if decision.knowledge_result is not None
                else ""
            ),
            retrieval_score=decision.retrieval_score,
            retrieval_coverage=decision.retrieval_coverage,
            allow_regenerate=not regenerated,
            timeout=review_timeout,
        )
        if not review or review.confidence < 0.6:
            failure_reason = (
                "final review unavailable"
                if review is None
                else "final review low confidence"
            )
            degraded = deterministic_review_failure_answer(
                decision,
                candidate,
                mentioned=mentioned,
                context_changed=context_changed,
            )
            if degraded:
                return (
                    degraded,
                    f"{failure_reason}; deterministic addressed degradation",
                    latest_revision,
                )
            return "", failure_reason, latest_revision
        merge_review_message_ids(target_item, review, latest_context)
        if review.action == "drop":
            return "", f"final review dropped [{review.context_relation}]: {review.reason}", latest_revision
        if review.action == "revise":
            candidate = finalize_model_answer(
                review.revised_reply,
                unsolicited=decision.reply_mode == "chat",
            )
            if not candidate:
                return "", "final review produced empty revision", latest_revision
            return candidate, f"final review revised [{review.context_relation}]: {review.reason}", latest_revision
        if review.action == "send":
            return candidate, f"final review accepted [{review.context_relation}]: {review.reason}", latest_revision

        regenerated = True
        decision.reply_regenerated = True
        semantic_replan_reason = ""
        if (
            target_item
            and decision.semantic_audience != "unclear"
            and decision.semantic_replan_count < 1
        ):
            refreshed_decision, semantic_replan_reason = (
                refresh_semantic_decision_for_late_context(
                    decision,
                    target_item,
                    latest_context,
                    scene_context=current_group_chat_scene(
                        group_id,
                        focus_sequence=int(target_item.get("chat_sequence") or 0),
                    ),
                    deadline=deadline,
                )
            )
            if refreshed_decision is None:
                return "", semantic_replan_reason, latest_revision
            decision = refreshed_decision
        updated_question = (
            decision.effective_question
            if semantic_replan_reason.startswith("related late context")
            else review.updated_question
        )
        if not semantic_replan_reason.startswith("related late context"):
            decision.chat_context = tuple(latest_context)
        decision.effective_question = updated_question
        decision.draft_reply = ""
        reserve = getattr(settings, "final_reply_review_timeout_seconds", 4)
        generation_cap = (
            getattr(settings, "chat_generation_timeout_seconds", 7)
            if decision.reply_mode == "chat"
            else getattr(settings, "knowledge_generation_timeout_seconds", 10)
        )
        generation_timeout = remaining_reply_timeout(
            deadline,
            cap=generation_cap,
            reserve=reserve,
        )
        if not generation_timeout:
            return "", "reply deadline exhausted before regeneration", latest_revision
        candidate = answer_for_decision(
            updated_question,
            decision,
            updated_question,
            admin=admin,
            timeout=generation_timeout,
        )
        if not candidate:
            return "", "regeneration produced no answer", latest_revision
        baseline_revision = latest_revision


def refresh_answer_for_late_context(
    *,
    question: str,
    answer: str,
    decision: ProcessingDecision,
    group_id: int,
    mentioned: bool,
    admin: bool,
    deadline: float,
    reviewed_revision: int,
    target_item: dict | None = None,
) -> tuple[str, str, int]:
    """Re-review only when messages arrived after the previous review."""
    latest_revision = latest_group_user_sequence(group_id)
    if latest_revision <= reviewed_revision:
        return answer, "context unchanged after final review", reviewed_revision
    refreshed_answer, reason, revision = review_and_refresh_answer(
        question=question,
        answer=answer,
        decision=decision,
        group_id=group_id,
        mentioned=mentioned,
        admin=admin,
        deadline=deadline,
        baseline_revision=reviewed_revision,
        target_item=target_item,
    )
    if refreshed_answer:
        unsafe_reason = unsafe_or_repeated_reply(group_id, refreshed_answer)
        if unsafe_reason:
            return "", unsafe_reason, revision
        grounding_issue = fallback_grounding_issue(decision, refreshed_answer)
        if grounding_issue:
            return "", grounding_issue, revision
    return refreshed_answer, reason, revision


def clear_chat_state() -> None:
    global chat_message_sequence
    with chat_history_lock:
        chat_history_state.clear()
        chat_message_sequence = 0
    chat_scene_state.clear()
    with hostile_reply_lock:
        hostile_reply_history.clear()
    semantic_planner_health.reset()
    clear_fragment_state()


def group_send_lock(group_id: int) -> threading.Lock:
    with group_send_locks_lock:
        return group_send_locks.setdefault(group_id, threading.Lock())


SEMANTIC_CHAT_INTENTS = {
    "chat",
    "normal_chat",
    "banter_at_bot",
    "third_party_attack",
    "genuine_criticism",
    "hostile_abuse",
}
LOCAL_REPLY_MODES = {"bot_meta", "identity", "control_boundary"}


def allow_hostile_reply(group_id: int, user_id: str, *, now: float | None = None) -> bool:
    current_time = time.time() if now is None else now
    key = (group_id, str(user_id or ""))
    with hostile_reply_lock:
        recent = [
            timestamp
            for timestamp in hostile_reply_history.get(key, ())
            if current_time - timestamp < 600
        ]
        allowed = len(recent) < 2
        recent.append(current_time)
        hostile_reply_history[key] = recent
        return allowed


def save_chat_history(path: str | Path | None = None) -> None:
    """Persist chat history to disk for restart recovery."""
    save_path = Path(path or "work/chat_history.json")
    try:
        chat_history_state.save(save_path)
    except Exception as exc:
        print("Save chat history failed:", repr(exc))


def schedule_chat_history_save() -> None:
    chat_history_save_event.set()


def chat_history_save_worker() -> None:
    while True:
        chat_history_save_event.wait()
        time.sleep(0.5)
        chat_history_save_event.clear()
        save_chat_history()


def initialize_chat_memory() -> bool:
    global chat_memory_manager
    if not getattr(settings, "chat_memory_enabled", True):
        return False
    try:
        store = ChatMemoryStore(
            getattr(settings, "chat_memory_db", "work/chat_memory.sqlite3"),
            build_embedding_provider(settings),
        )
        chat_memory_manager = ChatMemoryManager(
            store,
            retention_days=max(0, getattr(settings, "chat_memory_retention_days", 90)),
        )
        chat_memory_manager.start()
        return True
    except Exception as exc:
        chat_memory_manager = None
        print("Chat memory initialization failed:", type(exc).__name__, repr(exc))
        return False


def recall_group_chat_message(group_id: int, message_id: str) -> None:
    target = str(message_id or "").strip()
    if not target:
        return
    chat_history_state.recall(group_id, target)
    if chat_memory_manager:
        chat_memory_manager.enqueue_recall(group_id, target)


def load_chat_history(path: str | Path | None = None) -> int:
    """Load persisted chat history on startup."""
    global chat_message_sequence
    load_path = Path(path or "work/chat_history.json")
    if not load_path.exists():
        return 0
    try:
        count = chat_history_state.load(load_path)
        with chat_history_lock:
            chat_message_sequence = chat_history_state.sequence
        print(f"Loaded {count} chat history entries from {load_path}")
        return count
    except Exception as exc:
        print("Load chat history failed:", repr(exc))
        return 0


def migrate_loaded_chat_history_to_memory() -> int:
    if not chat_memory_manager:
        return 0
    snapshot = chat_history_state.snapshot()
    queued = 0
    for group_id, item in snapshot:
        speaker_id = stable_member_id(group_id, item.user_id)
        queued += int(chat_memory_manager.enqueue(MemoryMessage(
            group_id=group_id,
            message_id=item.message_id or f"local:{group_id}:{item.sequence}",
            speaker_id=speaker_id,
            display_name=item.display_name if speaker_id != "bot" else "机器人",
            speaker_role="bot" if speaker_id == "bot" else "member",
            text=item.text,
            event_time=item.timestamp,
            reply_message_id=item.reply_message_id,
            reply_speaker_id=stable_member_id(group_id, item.reply_target_user_id),
            quoted_text=item.reply_text,
            mentions=tuple(stable_member_id(group_id, value) for value in item.mentioned_user_ids),
            generated_for_message_ids=item.generated_for_message_ids,
            turn_id=item.turn_id,
            reply_mode=item.reply_mode,
            semantic_topic=item.semantic_topic,
            sequence=item.sequence,
            received_time=item.received_time,
            content_segments=item.content_segments,
            message_status=item.message_status,
        )))
    return queued


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
        return pending_store.chat_reply_quota_reason(
            group_id,
            db_path=_pending_db_path(db_path),
            now=current_time,
            cooldown_seconds=cooldown,
            max_per_hour=hourly_limit,
        )


def mark_chat_replied(
    group_id: int,
    *,
    now: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    timestamp = time.time() if now is None else now
    with chat_reply_lock:
        pending_store.mark_chat_replied(
            group_id,
            db_path=_pending_db_path(db_path),
            now=timestamp,
        )


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
        return pending_store.celebration_was_replied(
            group_id,
            target_key,
            event_kind,
            db_path=_pending_db_path(db_path),
            now=current_time,
            window_seconds=window_seconds,
        )


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
        pending_store.mark_celebration_replied(
            group_id,
            target_key,
            event_kind,
            db_path=_pending_db_path(db_path),
            now=timestamp,
        )


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


def record_knowledge_gap(query: str, result) -> bool:
    normalized = " ".join(str(query or "").strip().split())
    if len(normalized) < 2:
        return False
    safe_query = redact_for_model(normalized)[:300]
    now = time.time()
    dedupe_seconds = max(0, getattr(settings, "knowledge_gap_dedupe_seconds", 3600))
    with knowledge_gap_lock:
        previous = recent_knowledge_gap_queries.get(safe_query, 0)
        if now - previous < dedupe_seconds:
            return False
        recent_knowledge_gap_queries[safe_query] = now
        cutoff = now - max(dedupe_seconds, 86400)
        for key, timestamp in tuple(recent_knowledge_gap_queries.items()):
            if timestamp < cutoff:
                recent_knowledge_gap_queries.pop(key, None)
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "query": safe_query,
            "top_score": round(float(result.top_score), 4),
            "coverage": round(float(result.query_coverage), 4),
            "sources": list(result.sources),
            "matched_tokens": list(result.matched_query_tokens),
            "missing_tokens": list(result.missing_query_tokens),
        }
        path = Path(getattr(settings, "knowledge_gap_log", "work/knowledge_gaps.jsonl"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as exc:
            print("Knowledge gap log write failed:", repr(exc))
            return False


def recent_knowledge_gap_entries(limit: int = 5) -> list[dict]:
    path = Path(getattr(settings, "knowledge_gap_log", "work/knowledge_gaps.jsonl"))
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def retrieve_knowledge(query: str, max_chars: int):
    with kb_lock:
        return kb.build_context_with_metrics(query, max_chars)


def attach_knowledge_result(
    decision: ProcessingDecision,
    query: str,
    result: ContextResult,
) -> ProcessingDecision:
    return attach_result(decision, query, result)


def is_strong_knowledge_match(top_score: float, query_coverage: float) -> bool:
    return semantic_routing.is_strong_knowledge_match(
        top_score,
        query_coverage,
        min_score=settings.knowledge_strong_min_score,
        min_coverage=settings.knowledge_strong_min_coverage,
    )


def contextual_retrieval_question(
    question: str,
    chat_context: Sequence[str],
) -> tuple[str, float] | None:
    if not getattr(settings, "contextual_query_enabled", False) or not chat_context:
        return None
    return rewrite_contextual_question(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=getattr(settings, "contextual_query_model", settings.llm_model),
        question=question,
        context=tuple(chat_context[-8:]),
        timeout=getattr(settings, "contextual_query_timeout_seconds", 8),
    )


def semantic_plan_for_message(
    question: str,
    chat_context: Sequence[str],
    *,
    scene_context: str = "",
    memory_candidates: Sequence[str] = (),
    mentioned: bool,
    mentions_other: bool,
    reply_target_user_id: str = "",
    newer_message_ids: Sequence[str] = (),
    timeout: int | None = None,
) -> MessagePlan | None:
    if not getattr(settings, "semantic_planner_enabled", False):
        return None
    return semantic_routing.plan_message(
        question,
        chat_context,
        planner=plan_group_message,
        memory_budget=budget_memory_context,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=getattr(settings, "semantic_planner_model", settings.llm_model),
        scene_context=scene_context,
        memory_candidates=memory_candidates,
        mentioned=mentioned,
        mentions_other=mentions_other,
        reply_target_user_id=reply_target_user_id,
        bot_user_id=settings.bot_qq,
        newer_message_ids=newer_message_ids,
        timeout=timeout or getattr(settings, "semantic_planner_timeout_seconds", 4),
        context_messages=getattr(settings, "semantic_planner_context_messages", 10),
        context_max_chars=getattr(
            settings,
            "semantic_planner_context_max_chars",
            3200,
        ),
        memory_max_chars=getattr(settings, "semantic_planner_memory_max_chars", 800),
    )


def semantic_planner_timeout_cap(
    *,
    mentioned: bool,
    reply_target_user_id: str = "",
    explicit_knowledge_command: bool = False,
) -> int:
    explicitly_addressed = bool(
        mentioned
        or explicit_knowledge_command
        or reply_target_user_id == settings.bot_qq
    )
    default_timeout = getattr(settings, "semantic_planner_timeout_seconds", 3)
    return semantic_routing.planner_timeout_cap(
        explicitly_addressed=explicitly_addressed,
        default_timeout=default_timeout,
        addressed_timeout=getattr(
            settings,
            "semantic_planner_addressed_timeout_seconds",
            default_timeout,
        ),
    )


def semantic_planner_circuit_is_open(
    *,
    lane: str = "unsolicited",
    now: float | None = None,
) -> bool:
    return semantic_planner_health.circuit_is_open(lane, now=now)


def reserve_semantic_planner_request(
    *,
    lane: str = "unsolicited",
    now: float | None = None,
) -> bool:
    return semantic_planner_health.reserve_request(
        lane,
        failure_threshold=getattr(settings, "semantic_planner_circuit_failures", 5),
        now=now,
    )


def semantic_planner_health_snapshot(*, now: float | None = None) -> dict[str, dict]:
    return semantic_planner_health.snapshot(now=now)


def record_semantic_planner_availability(
    available: bool,
    *,
    lane: str = "unsolicited",
    now: float | None = None,
) -> None:
    semantic_planner_health.record(
        available,
        lane,
        failure_threshold=getattr(settings, "semantic_planner_circuit_failures", 5),
        circuit_seconds=getattr(settings, "semantic_planner_circuit_seconds", 30),
        now=now,
    )


def semantic_plan_is_usable(plan: MessagePlan | None) -> bool:
    return semantic_routing.semantic_plan_is_usable(
        plan,
        min_confidence=getattr(settings, "semantic_planner_min_confidence", 0.68),
    )


def compact_scene_context(scene_context: str) -> str:
    return semantic_routing.compact_scene_context(scene_context)


def context_selected_by_plan(
    chat_context: Sequence[str],
    plan: MessagePlan | None,
) -> tuple[str, ...]:
    return semantic_routing.context_selected_by_plan(chat_context, plan)


def context_line_payload(line: str) -> dict:
    return semantic_routing.context_line_payload(line)


def context_line_message_id(line: str) -> str:
    return semantic_routing.context_line_message_id(line)


def context_revision(context: Sequence[str]) -> int:
    return semantic_routing.context_revision(context)


def context_message_ids_after_revision(
    context: Sequence[str],
    revision: int,
) -> tuple[str, ...]:
    return semantic_routing.context_message_ids_after_revision(context, revision)


def budget_recent_context(
    context: Sequence[str],
    *,
    max_messages: int,
    max_chars: int,
    required_message_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    return semantic_routing.budget_recent_context(
        context,
        max_messages=max_messages,
        max_chars=max_chars,
        required_message_ids=required_message_ids,
    )


def semantic_context_for_decision(decision: ProcessingDecision) -> str:
    return semantic_routing.semantic_context_for_decision(decision)


def semantic_relation_audit_fields(decision: ProcessingDecision) -> dict:
    return semantic_routing.semantic_relation_audit_fields(decision)


def derive_bot_reply_perspective(
    plan: MessagePlan,
    selected_context: Sequence[str],
) -> tuple[str, str]:
    return semantic_routing.derive_bot_reply_perspective(plan, selected_context)


def derive_participation_role(
    plan: MessagePlan,
    selected_context: Sequence[str],
    *,
    explicitly_addressed: bool,
) -> str:
    return semantic_routing.derive_participation_role(
        plan,
        selected_context,
        explicitly_addressed=explicitly_addressed,
    )


def apply_semantic_plan_metadata(
    decision: ProcessingDecision,
    plan: MessagePlan,
    *,
    explicitly_addressed: bool = False,
    context_revision: int = 0,
    scene_context: str = "",
    target_message_ids: Sequence[str] = (),
) -> ProcessingDecision:
    return semantic_routing.apply_semantic_plan_metadata(
        decision,
        plan,
        explicitly_addressed=explicitly_addressed,
        context_revision=context_revision,
        scene_context=scene_context,
        target_message_ids=target_message_ids,
    )


def refresh_semantic_decision_for_late_context(
    decision: ProcessingDecision,
    item: dict,
    latest_context: Sequence[str],
    *,
    scene_context: str,
    deadline: float,
) -> tuple[ProcessingDecision | None, str]:
    latest_context = tuple(latest_context)
    latest_revision = context_revision(latest_context)
    newer_ids = context_message_ids_after_revision(
        latest_context,
        decision.plan_context_revision,
    )
    if not newer_ids:
        return decision, "semantic context unchanged"
    if (
        not getattr(settings, "semantic_replan_enabled", True)
        or decision.semantic_replan_count >= 1
    ):
        return decision, "semantic replan already used or disabled"

    timeout = remaining_reply_timeout(
        deadline,
        cap=semantic_planner_timeout_cap(
            mentioned=bool(item.get("mentioned")),
            reply_target_user_id=str(item.get("reply_target_user_id") or ""),
            explicit_knowledge_command=bool(item.get("explicit_knowledge_command")),
        ),
        reserve=2,
    )
    if not timeout:
        return None, "reply deadline exhausted before semantic replan"
    plan = semantic_plan_for_message(
        str(item.get("question") or ""),
        latest_context,
        scene_context=scene_context,
        mentioned=bool(item.get("mentioned")),
        mentions_other=bool(item.get("mentions_other")),
        reply_target_user_id=str(item.get("reply_target_user_id") or ""),
        newer_message_ids=newer_ids,
        timeout=timeout,
    )
    if not semantic_plan_is_usable(plan):
        if item.get("mentioned"):
            decision.plan_context_revision = latest_revision
            return decision, "semantic replan unavailable; explicit address preserved"
        return None, "semantic replan unavailable; unsolicited reply fails closed"

    related_ids = set(plan.relevant_context_message_ids)
    related_ids.update(
        message_id
        for candidate in plan.topic_candidates
        for message_id in candidate.anchor_message_ids
    )
    related_ids.update(
        message_id
        for candidate in plan.subject_candidates
        for message_id in candidate.evidence_message_ids
    )
    related_new_ids = tuple(
        message_id for message_id in newer_ids if message_id in related_ids
    )
    if not related_new_ids:
        decision.plan_context_revision = latest_revision
        decision.semantic_replan_count += 1
        decision.semantic_replan_reason = "new messages were unrelated parallel context"
        return decision, decision.semantic_replan_reason

    decision.chat_context = context_selected_by_plan(latest_context, plan)
    apply_semantic_plan_metadata(
        decision,
        plan,
        explicitly_addressed=(
            bool(item.get("mentioned"))
            or str(item.get("reply_target_user_id") or "") == settings.bot_qq
        ),
        context_revision=latest_revision,
        scene_context=scene_context,
        target_message_ids=(*_message_ids(item), *related_new_ids),
    )
    decision.semantic_replan_count += 1
    decision.semantic_replan_reason = (
        "related late context replaced semantic decision: "
        + ",".join(related_new_ids)
    )
    if (
        not item.get("mentioned")
        and decision.participation_role in {"bystander", "uncertain"}
    ):
        return None, (
            "semantic replan removed bot participation: "
            f"{decision.participation_role}"
        )
    if decision.semantic_audience == "member" and not item.get("mentioned"):
        return None, "semantic replan redirected message to another member"
    if decision.reply_mode == "chat" and (
        plan.intent not in SEMANTIC_CHAT_INTENTS or not plan.reply_worthy
    ):
        return None, "semantic replan found no natural chat entry"
    return decision, decision.semantic_replan_reason


def chat_memory_enabled_for_group(group_id: int) -> bool:
    if not chat_memory_manager or not getattr(settings, "chat_memory_enabled", True):
        return False
    allowed = getattr(settings, "chat_memory_allowed_group_ids", ())
    return not allowed or str(group_id) in allowed


def probe_chat_memory(item: dict, query: str) -> MemoryProbeResult:
    group_id = int(item.get("group_id") or 0)
    if not chat_memory_enabled_for_group(group_id):
        return MemoryProbeResult(query=query, rejection_reason="memory disabled for group")
    normalized = "".join(re.findall(r"[a-z0-9\u3400-\u9fff]", str(query or "").lower()))
    if len(normalized) < 4:
        return MemoryProbeResult(query=query, rejection_reason="query too short for automatic probe")
    try:
        hits = chat_memory_manager.store.lexical_probe(
            group_id=group_id,
            query=query,
            exclude_message_id=str(item.get("message_id") or ""),
            limit=max(1, getattr(settings, "chat_memory_probe_max_hits", 8)),
            max_chars=max(200, getattr(settings, "chat_memory_probe_max_chars", 1600)),
        )
        context = chat_memory_manager.store.format_hits(hits)
        return MemoryProbeResult(
            query=query,
            hits=hits,
            context=context,
            attempted=True,
            rejection_reason="" if context else "no relevant memory candidate",
        )
    except Exception as exc:
        print("Chat memory probe failed:", type(exc).__name__, repr(exc))
        return MemoryProbeResult(
            query=query,
            attempted=True,
            rejection_reason=f"probe error: {type(exc).__name__}",
        )


def deduplicate_memory_context(
    memory_context: Sequence[str],
    recent_context: Sequence[str],
) -> tuple[tuple[str, ...], int]:
    recent_message_ids = {
        context_line_message_id(line) for line in recent_context if context_line_message_id(line)
    }
    result: list[str] = []
    dropped = 0
    for line in memory_context:
        payload = context_line_payload(line)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            result.append(line)
            continue
        kept = []
        for message in messages:
            message_id = str(message.get("message_id") or "") if isinstance(message, dict) else ""
            if message_id and message_id in recent_message_ids:
                dropped += 1
                continue
            kept.append(message)
        if not kept:
            continue
        payload["messages"] = kept
        result.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return tuple(result), dropped


def budget_memory_context(
    context: Sequence[str],
    *,
    max_hits: int,
    max_chars: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    used = 0
    for line in context:
        if not selected and len(line) > max_chars:
            continue
        if selected and used + len(line) > max(200, max_chars):
            continue
        selected.append(line)
        used += len(line)
        if len(selected) >= max(1, max_hits) or used >= max_chars:
            break
    return tuple(selected)


def apply_context_budget(decision: ProcessingDecision) -> None:
    recent_candidate_count = (
        decision.recent_context_candidate_count or len(decision.chat_context)
    )
    if decision.reply_mode == "chat":
        recent_limit = max(1, getattr(settings, "planned_chat_context_messages", 10))
        recent_chars = max(400, getattr(settings, "planned_chat_context_max_chars", 1600))
        memory_hits, memory_chars = 3, 1000
    elif decision.reply_mode == "knowledge":
        recent_limit, recent_chars = 8, 1200
        memory_hits, memory_chars = 2, 800
    else:
        recent_limit, recent_chars = 8, 1200
        memory_hits, memory_chars = 2, 800
    topic_anchor_ids = tuple(dict.fromkeys(
        message_id
        for candidate in decision.topic_candidates
        for message_id in candidate.anchor_message_ids
    ))
    decision.chat_context = budget_recent_context(
        decision.chat_context,
        max_messages=recent_limit,
        max_chars=recent_chars,
        required_message_ids=topic_anchor_ids,
    )
    decision.memory_context = budget_memory_context(
        decision.memory_context,
        max_hits=memory_hits,
        max_chars=memory_chars,
    )
    decision.recent_context_candidate_count = recent_candidate_count
    decision.recent_context_selected_count = len(decision.chat_context)
    decision.recent_context_chars = sum(len(line) for line in decision.chat_context)
    decision.memory_context_chars = sum(len(line) for line in decision.memory_context)
    decision.memory_hit_count = len(decision.memory_context)
    decision.recent_context_selected_ids = tuple(
        message_id
        for line in decision.chat_context
        if (message_id := context_line_message_id(line))
    )
    selected_chunk_ids: list[int] = []
    for line in decision.memory_context:
        try:
            chunk_id = int(context_line_payload(line).get("chunk_id"))
        except (TypeError, ValueError):
            continue
        if chunk_id not in selected_chunk_ids:
            selected_chunk_ids.append(chunk_id)
    decision.memory_selected_chunk_ids = tuple(selected_chunk_ids)


def enrich_decision_with_chat_memory(
    decision: ProcessingDecision,
    item: dict,
    plan: MessagePlan | None,
    probe: MemoryProbeResult | None = None,
) -> ProcessingDecision:
    group_id = int(item.get("group_id") or 0)
    reply_message_id = str(item.get("reply_message_id") or "")
    requested_by_plan = bool(plan and plan.memory_needed)
    hard_reply_retrieval = bool(reply_message_id and not requested_by_plan)
    query = (
        (plan.memory_query if plan else "")
        or decision.effective_question
        or str(item.get("question") or "")
    )
    probe = probe or probe_chat_memory(item, query)
    decision.memory_query = query
    decision.memory_retrieval_attempted = probe.attempted
    decision.memory_candidate_count = len(probe.context)
    decision.memory_rejection_reason = probe.rejection_reason
    decision.memory_selected_by_planner = bool(
        plan and (plan.selected_memory_chunk_ids or plan.memory_needed)
    )
    hits: Sequence[MemoryHit] = ()
    try:
        if not decision.should_reply:
            decision.memory_retrieval_mode = "probe_only"
            decision.memory_rejection_reason = "router decided no reply"
        elif not chat_memory_enabled_for_group(group_id):
            decision.memory_retrieval_mode = "disabled"
            decision.memory_rejection_reason = "memory disabled for group"
        elif requested_by_plan or hard_reply_retrieval:
            decision.memory_retrieval_mode = "planned" if requested_by_plan else "reply_chain"
            decision.memory_retrieval_attempted = True
            hits = chat_memory_manager.store.retrieve(
                group_id=group_id,
                query=query,
                speaker_id=stable_member_id(group_id, str(item.get("user_id") or "")),
                reply_message_id=reply_message_id,
                exclude_message_id=str(item.get("message_id") or ""),
                participant_scope=plan.participant_scope if requested_by_plan else "reply_chain",
                time_scope=plan.time_scope if plan else "",
                limit=max(1, getattr(settings, "chat_memory_max_hits", 6)),
                max_chars=max(200, getattr(settings, "chat_memory_max_chars", 2400)),
            )
        else:
            decision.memory_retrieval_mode = "lexical_probe"
            if plan:
                selected_ids = set(plan.selected_memory_chunk_ids)
                hits = tuple(hit for hit in probe.hits if hit.chunk_id in selected_ids)
                if probe.context and not selected_ids:
                    decision.memory_rejection_reason = "planner did not select memory candidate"
            else:
                hits = probe.hits
        formatted = chat_memory_manager.store.format_hits(hits) if hits else ()
        formatted, deduplicated = deduplicate_memory_context(formatted, decision.chat_context)
        decision.context_deduplicated_count = deduplicated
        decision.memory_hit_count = len(formatted)
        decision.memory_needed = bool(formatted)
        if not formatted:
            decision.memory_rejection_reason = decision.memory_rejection_reason or "no selected memory candidate"
        if getattr(settings, "chat_memory_shadow_mode", False):
            print("Chat memory shadow", group_id, len(formatted), query[:80])
            write_message_audit(
                decision="memory_shadow",
                reason="long-term chat retrieval shadow result",
                group_id=group_id,
                user_id=item.get("user_id"),
                question=str(item.get("question") or ""),
                memory_query=query,
                memory_hit_count=len(formatted),
                memory_retrieval_attempted=True,
                memory_retrieval_mode=decision.memory_retrieval_mode,
                memory_candidate_count=decision.memory_candidate_count,
                memory_rejection_reason=decision.memory_rejection_reason,
                event_time=item.get("time"),
            )
        else:
            decision.memory_context = formatted
            if formatted:
                decision.draft_reply = ""

        if decision.should_reply and chat_memory_enabled_for_group(group_id):
            related_message_ids = list(_message_ids(item))
            selected_recent_ids = tuple(
                message_id
                for line in decision.chat_context
                if (message_id := context_line_message_id(line))
            )
            for message_id in (
                reply_message_id,
                *selected_recent_ids,
                *(message_id for hit in hits for message_id in hit.message_ids),
            ):
                value = str(message_id or "").strip()
                if value and value not in related_message_ids:
                    related_message_ids.append(value)
            topic_ids_list = list(dict.fromkeys(hit.topic_id for hit in hits if hit.topic_id))
            for message_id in selected_recent_ids:
                for assignment in chat_memory_manager.store.topic_assignments_for_message(
                    group_id,
                    message_id,
                ):
                    if assignment.topic_id not in topic_ids_list:
                        topic_ids_list.append(assignment.topic_id)
            topic_ids = tuple(topic_ids_list)
            self_history = chat_memory_manager.store.format_self_history(
                group_id=group_id,
                related_message_ids=related_message_ids,
                topic_ids=topic_ids,
                limit=max(1, getattr(settings, "bot_self_history_max_turns", 3)),
                max_chars=max(200, getattr(settings, "bot_self_history_max_chars", 1200)),
            )
            decision.self_history_context = self_history
            decision.self_history_candidate_count = len(self_history)
            decision.self_history_selected_count = len(self_history)
            decision.self_history_chars = sum(len(line) for line in self_history)
            selected_bot_ids: list[str] = []
            for line in self_history:
                try:
                    message_id = str(
                        context_line_payload(line).get("bot_message", {}).get("message_id") or ""
                    )
                except (AttributeError, TypeError):
                    message_id = ""
                if message_id and message_id not in selected_bot_ids:
                    selected_bot_ids.append(message_id)
            decision.self_history_selected_message_ids = tuple(selected_bot_ids)
            if self_history:
                # Planner drafts are created before exact prior bot turns are loaded.
                decision.draft_reply = ""
            reasons: list[str] = []
            if reply_message_id:
                reasons.append("qq_reply")
            if selected_recent_ids:
                reasons.append("selected_recent_context")
            if hits:
                reasons.append("retrieved_topic")
            decision.self_history_reasons = tuple(reasons)
    except Exception as exc:
        decision.memory_rejection_reason = f"retrieval error: {type(exc).__name__}"
        print("Chat memory retrieval failed:", type(exc).__name__, repr(exc))
    apply_context_budget(decision)
    if not getattr(settings, "chat_memory_shadow_mode", False):
        write_message_audit(
            decision="memory_retrieval",
            reason=decision.memory_rejection_reason or "memory context injected",
            group_id=group_id,
            user_id=item.get("user_id"),
            question=str(item.get("question") or ""),
            memory_query=query,
            memory_hit_count=decision.memory_hit_count,
            memory_retrieval_attempted=decision.memory_retrieval_attempted,
            memory_retrieval_mode=decision.memory_retrieval_mode,
            memory_candidate_count=decision.memory_candidate_count,
            memory_rejection_reason=decision.memory_rejection_reason,
            recent_context_candidate_count=decision.recent_context_candidate_count,
            recent_context_selected_count=decision.recent_context_selected_count,
            recent_context_chars=decision.recent_context_chars,
            memory_context_chars=decision.memory_context_chars,
            context_deduplicated_count=decision.context_deduplicated_count,
            recent_context_selected_ids=decision.recent_context_selected_ids,
            memory_selected_chunk_ids=decision.memory_selected_chunk_ids,
            memory_selected_by_planner=decision.memory_selected_by_planner,
            self_history_candidate_count=decision.self_history_candidate_count,
            self_history_selected_count=decision.self_history_selected_count,
            self_history_chars=decision.self_history_chars,
            self_history_selected_message_ids=decision.self_history_selected_message_ids,
            self_history_reasons=decision.self_history_reasons,
            topic_candidates=decision.topic_candidates,
            subject_candidates=decision.subject_candidates,
            subject_ambiguity=decision.subject_ambiguity,
            bot_involvement=decision.bot_involvement,
            reply_perspective=decision.reply_perspective,
            semantic_audience=decision.semantic_audience,
            participation_role=decision.participation_role,
            plan_context_revision=decision.plan_context_revision,
            plan_scene_version=decision.plan_scene_version,
            related_message_ids=decision.related_message_ids,
            event_time=item.get("time"),
        )
    return decision


def answer_bot_meta(capability: str, *, admin: bool) -> str:
    if not admin:
        return "这类内部状态只对管理员开放。"
    knowledge_path = Path(settings.knowledge_dir)
    files = sorted(path.name for path in knowledge_path.glob("*.md"))
    if capability == "knowledge_files":
        if not files:
            return "当前没有发现可加载的知识库文件。"
        return "当前加载的知识库文件有：" + "、".join(files)
    if capability == "knowledge_status":
        return f"知识库已加载，当前有 {len(files)} 个文件、{len(kb.chunks)} 个片段。"
    if capability == "model_status":
        return f"知识问答使用 {settings.llm_model}，闲聊使用 {settings.chat_model}。"
    if capability in {"runtime_status", "health"}:
        queued = message_queue.qsize() + normal_message_queue.qsize() + chat_queue.qsize()
        return f"服务正在运行，知识片段 {len(kb.chunks)} 个，队列 {queued} 条。"
    return "可以查看知识库加载状态、知识库文件、当前模型和服务健康状态。"


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
    retrieval_question: str | None = None,
    allow_fallback: bool = True,
    chat_context: Sequence[str] = (),
    memory_context: Sequence[str] = (),
    self_history_context: Sequence[str] = (),
    semantic_context: str = "",
    knowledge_result: ContextResult | None = None,
    timeout: int | None = None,
) -> str:
    if is_identity_question(question):
        return (
            "叫我新兵营教官就行，主要给刚入坑 Squad 的兄弟答疑。"
            "HAB、FOB、医疗兵、反坦、搜不到服、卡三点、TS 设置这些都能问。"
            "要是问到本服规则，我不乱拍板，按群公告和管理员说法来。"
        )

    llm_question = effective_question or question
    result = knowledge_result or retrieve_knowledge(
        retrieval_question or llm_question,
        settings.max_context_chars,
    )
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
                context=tuple(chat_context[-8:]),
                memory_context=tuple(memory_context),
                self_history_context=tuple(self_history_context),
                semantic_context=semantic_context,
                candidate_knowledge_context=result.context,
                timeout=timeout or getattr(settings, "knowledge_generation_timeout_seconds", 10),
            )
            if unsupported_fallback_precise_facts(answer, result.context):
                return "这个具体数值我没有可靠依据，不能给你拍一个。"
            return finalize_model_answer(answer)
        return "这个我库里暂时没有准确信息。你可以换个更具体的问法，或者问一下小队长和管理员；涉及服务器规则的话，还是以本服公告为准。"

    answer = ask_llm(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        question=llm_question,
        context=result.context,
        chat_context=tuple(chat_context[-8:]),
        memory_context=tuple(memory_context),
        self_history_context=tuple(self_history_context),
        semantic_context=semantic_context,
        timeout=timeout or getattr(settings, "knowledge_generation_timeout_seconds", 10),
    )
    return finalize_model_answer(answer)


def fallback_grounding_issue(decision: ProcessingDecision, answer: str) -> str:
    if decision.reply_mode != "fallback":
        return ""
    candidate_context = (
        decision.knowledge_result.context
        if decision.knowledge_result is not None
        else ""
    )
    unsupported = unsupported_fallback_precise_facts(answer, candidate_context)
    if not unsupported:
        return ""
    rendered = ", ".join(f"{value}{unit}" for value, unit in sorted(unsupported))
    return f"unsupported precise fact in fallback reply: {rendered}"


def deterministic_review_failure_answer(
    decision: ProcessingDecision,
    candidate: str,
    *,
    mentioned: bool,
    context_changed: bool = False,
) -> str:
    explicitly_addressed = bool(
        mentioned
        or decision.participation_role == "addressed"
        or decision.semantic_audience == "bot"
    )
    if not explicitly_addressed:
        return ""
    if decision.semantic_intent == "control_attempt":
        return "这类操作不能通过普通聊天执行。"
    if (
        decision.reply_mode == "knowledge"
        and candidate
        and not context_changed
        and not decision.risk_flags
        and is_strong_knowledge_match(
            decision.retrieval_score,
            decision.retrieval_coverage,
        )
    ):
        knowledge_context = (
            decision.knowledge_result.context
            if decision.knowledge_result is not None
            else ""
        )
        if not unsupported_fallback_precise_facts(candidate, knowledge_context):
            return candidate
    if decision.semantic_intent == "knowledge":
        return "这个具体问题我暂时没有可靠信息，不能确定。"
    if decision.planner_status in {"unavailable", "circuit_open", "low_confidence"}:
        return "这次我没判断清楚你在问什么，换个说法再问我一次。"
    return ""


def answer_for_decision(
    question: str,
    decision: ProcessingDecision,
    generation_question: str,
    *,
    admin: bool = False,
    timeout: int | None = None,
) -> str:
    llm_question = generation_question or decision.effective_question or question
    semantic_context = semantic_context_for_decision(decision)
    if decision.reply_mode == "control_boundary":
        return "这类操作不能通过普通聊天执行。"
    if decision.reply_mode == "bot_meta":
        return answer_bot_meta(decision.capability, admin=admin)
    if decision.draft_reply and decision.reply_mode in {"fallback", "chat"}:
        return finalize_model_answer(
            decision.draft_reply,
            unsolicited=decision.reply_mode == "chat",
        )
    if decision.reply_mode == "fallback":
        candidate_knowledge_context = (
            decision.knowledge_result.context
            if decision.knowledge_result is not None
            else ""
        )
        answer = ask_fallback_llm(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            question=llm_question,
            context=decision.chat_context,
            memory_context=decision.memory_context,
            self_history_context=decision.self_history_context,
            semantic_context=semantic_context,
            candidate_knowledge_context=candidate_knowledge_context,
            timeout=timeout or getattr(settings, "knowledge_generation_timeout_seconds", 10),
        )
        if fallback_grounding_issue(decision, answer):
            return "这个具体数值我没有可靠依据，不能给你拍一个。"
        return finalize_model_answer(answer)
    if decision.reply_mode == "chat":
        answer = answer_chat(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.chat_model,
            message=question,
            context=decision.chat_context,
            memory_context=decision.memory_context,
            self_history_context=decision.self_history_context,
            semantic_context=semantic_context,
            timeout=timeout or getattr(settings, "chat_generation_timeout_seconds", 7),
        )
        return finalize_model_answer(answer, unsolicited=True)
    return answer_question(
        question,
        llm_question,
        retrieval_question=decision.effective_question or question,
        allow_fallback=False,
        chat_context=decision.chat_context,
        memory_context=decision.memory_context,
        self_history_context=decision.self_history_context,
        semantic_context=semantic_context,
        knowledge_result=(
            decision.knowledge_result
            if decision.knowledge_query == (decision.effective_question or question)
            else None
        ),
        timeout=timeout,
    )


def is_model_error_answer(answer: str) -> bool:
    return answer.startswith(("模型接口", "还没有配置模型 API Key")) or is_provider_refusal_text(answer)


def next_sequence() -> int:
    global sequence_number
    with sequence_lock:
        sequence_number += 1
        return sequence_number


def _pending_db_path(db_path: str | Path | None = None) -> str | Path:
    return db_path or settings.pending_queue_db


def open_pending_queue_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    return pending_store.open_pending_queue_db(_pending_db_path(db_path))


def persist_conversation_turn(
    group_id: int,
    state: ConversationState,
    *,
    db_path: str | Path | None = None,
) -> int:
    return pending_store.persist_conversation_turn(
        group_id,
        state,
        db_path=_pending_db_path(db_path),
    )


def _conversation_state_from_row(row) -> ConversationState:
    return pending_store.conversation_state_from_row(row)


def load_conversation_turn_by_bot_message_id(
    group_id: int,
    bot_message_id: str,
    *,
    db_path: str | Path | None = None,
) -> ConversationState | None:
    return pending_store.load_conversation_turn_by_bot_message_id(
        group_id,
        bot_message_id,
        db_path=_pending_db_path(db_path),
    )


def persist_pending_message(
    priority: int,
    sequence: int,
    item: dict,
    db_path: str | Path | None = None,
) -> int:
    return pending_store.persist_pending_message(
        priority,
        sequence,
        item,
        db_path=_pending_db_path(db_path),
    )


def load_pending_messages(
    db_path: str | Path | None = None,
    *,
    include_future: bool = False,
) -> list[tuple[int, int, dict]]:
    return pending_store.load_pending_messages(
        db_path=_pending_db_path(db_path),
        include_future=include_future,
    )


def mark_pending_failure(
    pending_id: int,
    error: str,
    *,
    db_path: str | Path | None = None,
    now: float | None = None,
    max_attempts: int | None = None,
) -> PendingFailureResult:
    retry_limit = (
        max_attempts
        if max_attempts is not None
        else getattr(settings, "pending_retry_max_attempts", 3)
    )
    return pending_store.mark_pending_failure(
        pending_id,
        error,
        db_path=_pending_db_path(db_path),
        now=now,
        max_attempts=retry_limit,
    )


def mark_pending_sent_unknown(
    pending_id: int,
    error: str,
    *,
    db_path: str | Path | None = None,
) -> None:
    pending_store.mark_pending_sent_unknown(
        pending_id,
        error,
        db_path=_pending_db_path(db_path),
    )


def mark_pending_dispatch_started(
    pending_id: int,
    dispatch_id: str,
    *,
    db_path: str | Path | None = None,
) -> None:
    pending_store.mark_pending_dispatch_started(
        pending_id,
        dispatch_id,
        db_path=_pending_db_path(db_path),
    )


def recover_incomplete_pending_dispatches(
    db_path: str | Path | None = None,
) -> int:
    return pending_store.recover_incomplete_pending_dispatches(
        db_path=_pending_db_path(db_path)
    )


def delete_pending_message(
    pending_id: int,
    db_path: str | Path | None = None,
) -> None:
    pending_store.delete_pending_message(
        pending_id,
        db_path=_pending_db_path(db_path),
    )


def pending_message_count(db_path: str | Path | None = None) -> int:
    return pending_store.pending_message_count(db_path=_pending_db_path(db_path))


def pending_status_counts(db_path: str | Path | None = None) -> dict[str, int]:
    return pending_store.pending_status_counts(db_path=_pending_db_path(db_path))


def enqueue_persistent_message(priority: int, item: dict) -> int:
    sequence = next_sequence()
    pending_id = persist_pending_message(priority, sequence, item)
    queued_item = dict(item)
    queued_item["_pending_id"] = pending_id
    queued_item["_queue_priority"] = priority
    queued_item["_queue_sequence"] = sequence
    target_queue = message_queue if priority == 0 else normal_message_queue
    target_queue.put((priority, sequence, queued_item))
    return pending_id


def _queue_pending_item(item: dict, *, delay: float = 0.0) -> None:
    raw_priority = item.get("_queue_priority")
    priority = int(
        raw_priority
        if raw_priority is not None
        else (0 if item.get("mentioned") or item.get("explicit_knowledge_command") else 1)
    )
    raw_sequence = item.get("_queue_sequence")
    sequence = int(raw_sequence if raw_sequence is not None else next_sequence())
    queued_item = dict(item)
    queued_item["_queue_priority"] = priority
    queued_item["_queue_sequence"] = sequence
    target_queue = message_queue if priority == 0 else normal_message_queue

    def enqueue() -> None:
        target_queue.put((priority, sequence, queued_item))

    if delay <= 0:
        enqueue()
        return
    timer = threading.Timer(delay, enqueue)
    timer.daemon = True
    timer.start()


def handle_pending_worker_failure(item: dict, error: str) -> str:
    pending_id = item.get("_pending_id")
    if pending_id is None:
        return "untracked"
    if item.get("_dispatch_completed"):
        return "delivered"
    if item.get("_dispatch_started"):
        mark_pending_sent_unknown(int(pending_id), error)
        return "sent_unknown"
    result = mark_pending_failure(int(pending_id), error)
    if result.status == "retry":
        _queue_pending_item(
            item,
            delay=max(0.0, result.next_attempt_at - time.time()),
        )
    return result.status


def begin_pending_dispatch(item: dict) -> None:
    pending_id = item.get("_pending_id")
    dispatch_id = f"{pending_id or 'untracked'}:{time.time_ns()}"
    if pending_id is not None:
        mark_pending_dispatch_started(int(pending_id), dispatch_id)
    item["_pending_dispatch_id"] = dispatch_id
    item["_dispatch_started"] = True


def classify_fragment_audience(item: dict) -> str:
    """Classify who a fragment addresses without guessing from general pronouns."""
    return classify_audience(item, bot_user_id=str(settings.bot_qq or ""))


def fragment_items_compatible(
    buffer: MessageFragmentBuffer,
    item: dict,
    audience: str,
) -> bool:
    return items_compatible(buffer, item, audience)


def _message_ids(item: dict) -> list[str]:
    return message_ids(item)


def bot_turn_metadata(item: dict, bot_message_id) -> tuple[tuple[str, ...], str]:
    trigger_ids = tuple(_message_ids(item))
    bot_id = str(bot_message_id or "").strip()
    turn_id = f"bot:{bot_id}" if bot_id else ""
    return trigger_ids, turn_id


def send_and_record_bot_turn(
    *,
    group_id: int,
    item: dict,
    answer: str,
    reply_mode: str,
    semantic_topic: str = "",
    mention_user_id: str = "",
    reply_to_trigger: bool = False,
) -> tuple[object, tuple[str, ...], str]:
    trigger_message_id = str(item.get("message_id") or "")
    user_id = str(item.get("user_id") or "")
    begin_pending_dispatch(item)
    bot_message_id = send_group_msg(
        settings.onebot_api_url,
        group_id,
        answer,
        settings.onebot_access_token,
        mention_user_id=mention_user_id,
        reply_to_message_id=trigger_message_id if reply_to_trigger else "",
    )
    item["_dispatch_completed"] = True
    item["_sent_message_id"] = str(bot_message_id or "")
    trigger_message_ids, turn_id = bot_turn_metadata(item, bot_message_id)
    record_group_chat_message(
        group_id,
        settings.bot_qq,
        answer,
        message_id=bot_message_id,
        reply_message_id=trigger_message_id if reply_to_trigger else "",
        reply_target_user_id=user_id if reply_to_trigger else "",
        reply_text=str(item.get("question") or "") if reply_to_trigger else "",
        generated_for_message_ids=trigger_message_ids,
        turn_id=turn_id,
        reply_mode=reply_mode,
        semantic_topic=semantic_topic,
    )
    save_chat_history()
    return bot_message_id, trigger_message_ids, turn_id


def message_already_covered_by_bot(group_id: int, message_id) -> bool:
    """Return whether a completed bot turn already covers this incoming message."""
    target = str(message_id or "").strip()
    if not target:
        return False
    with chat_history_lock:
        return any(
            entry.user_id == str(settings.bot_qq or "")
            and entry.message_status == "active"
            and target in entry.generated_for_message_ids
            for entry in group_chat_history.get(int(group_id), ())
        )


def _context_message_speakers(context: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in context:
        try:
            payload = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        message_id = str(payload.get("message_id") or "").strip()
        if message_id:
            speaker = payload.get("speaker") if isinstance(payload.get("speaker"), dict) else {}
            result[message_id] = str(speaker.get("id") or "").strip()
    return result


def merge_review_message_ids(item: dict | None, review, latest_context: Sequence[str]) -> None:
    """Attach only real context message IDs selected by the semantic reviewer."""
    if item is None:
        return
    existing = _message_ids(item)
    context_speakers = _context_message_speakers(latest_context)
    original_speakers = {
        context_speakers[message_id]
        for message_id in existing
        if context_speakers.get(message_id)
    }
    for message_id in review.related_message_ids:
        value = str(message_id or "").strip()
        same_sender = bool(
            original_speakers
            and context_speakers.get(value) in original_speakers
        )
        if value and same_sender and value not in existing:
            existing.append(value)
    item["message_ids"] = existing


def _new_fragment_buffer(item: dict, audience: str, now: float) -> MessageFragmentBuffer:
    return fragment_aggregator.new_buffer(
        item,
        audience,
        now,
        debounce_seconds=settings.message_fragment_debounce_seconds,
        max_wait_seconds=settings.message_fragment_max_wait_seconds,
    )


def _merge_fragment(buffer: MessageFragmentBuffer, item: dict, audience: str, now: float) -> None:
    fragment_aggregator.merge(
        buffer,
        item,
        audience,
        now,
        debounce_seconds=settings.message_fragment_debounce_seconds,
        max_wait_seconds=settings.message_fragment_max_wait_seconds,
    )


def _fragment_prefix_item(buffer: MessageFragmentBuffer, count: int) -> dict:
    return fragment_aggregator.prefix_item(buffer, count)


def semantic_bot_fragment_count(buffer: MessageFragmentBuffer) -> int:
    if buffer.audience != "bot" or len(buffer.fragments) < 2:
        return len(buffer.fragments)
    if not any(
        fragment.get("fragment_audience") == "unknown"
        for fragment in buffer.fragments[1:]
    ):
        return len(buffer.fragments)
    if not getattr(settings, "message_fragment_semantic_enabled", True):
        return 1
    decision = classify_bot_fragment_prefix(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=getattr(settings, "message_fragment_semantic_model", settings.llm_model),
        fragments=buffer.parts,
        context=tuple(buffer.item.get("chat_context") or ()),
        timeout=getattr(settings, "message_fragment_semantic_timeout_seconds", 8),
    )
    if not decision:
        return 1
    count, confidence = decision
    minimum = getattr(settings, "message_fragment_semantic_min_confidence", 0.75)
    return count if confidence >= minimum else 1


def _dispatch_fragment_buffer(buffer: MessageFragmentBuffer) -> int:
    count = semantic_bot_fragment_count(buffer)
    item = _fragment_prefix_item(buffer, count)
    if count < len(buffer.fragments):
        print(
            "Fragment audience split",
            buffer.group_id,
            buffer.user_id,
            f"bot={count}/{len(buffer.fragments)}",
        )
    priority = 0 if item.get("mentioned") else 1
    return enqueue_persistent_message(priority, item)


def _defer_fragment_buffers(buffers: Sequence[MessageFragmentBuffer]) -> None:
    fragment_aggregator.defer(buffers)


def clear_fragment_state() -> None:
    fragment_aggregator.clear()


def flush_group_fragment_buffer(group_id: int, *, defer_dispatch: bool = False) -> int | None:
    buffer = fragment_aggregator.pop_group(group_id)
    if not buffer:
        return None
    if defer_dispatch:
        _defer_fragment_buffers((buffer,))
        return None
    return _dispatch_fragment_buffer(buffer)


def flush_fragment_buffer_for_new_speaker(
    group_id: int,
    user_id,
    *,
    defer_dispatch: bool = False,
) -> int | None:
    buffer = fragment_aggregator.pop_for_new_speaker(group_id, user_id)
    if not buffer:
        return None
    if defer_dispatch:
        _defer_fragment_buffers((buffer,))
        return None
    return _dispatch_fragment_buffer(buffer)


def submit_message_fragment(
    item: dict,
    *,
    now: float | None = None,
    defer_dispatch: bool = False,
) -> list[int]:
    """Buffer a fragment, optionally deferring semantic dispatch to the worker."""
    current_time = time.monotonic() if now is None else float(now)
    audience = classify_fragment_audience(item)
    is_admin_command = bool(
        is_admin_user(item.get("user_id"), item.get("sender_role", ""))
        and get_admin_command(str(item.get("question") or ""))
    )

    displaced = fragment_aggregator.submit(
        item,
        audience,
        now=current_time,
        is_immediate=is_admin_command,
        max_parts=settings.message_fragment_max_parts,
        max_chars=settings.message_fragment_max_chars,
        debounce_seconds=settings.message_fragment_debounce_seconds,
        max_wait_seconds=settings.message_fragment_max_wait_seconds,
    )

    pending_ids: list[int] = []
    if defer_dispatch:
        _defer_fragment_buffers(displaced)
    else:
        pending_ids = [_dispatch_fragment_buffer(buffer) for buffer in displaced]
    if is_admin_command:
        pending_ids.append(enqueue_persistent_message(0 if item.get("mentioned") else 1, item))
    return pending_ids


def fragment_aggregation_worker() -> None:
    while True:
        due = fragment_aggregator.wait_for_due()
        for buffer in due:
            try:
                _dispatch_fragment_buffer(buffer)
            except Exception as exc:
                print("Fragment queue write failed:", repr(exc))


def restore_pending_messages() -> int:
    global sequence_number
    recovered = recover_incomplete_pending_dispatches()
    if recovered:
        print("Marked interrupted message dispatches as sent_unknown", recovered)
    pending = load_pending_messages(include_future=True)
    if not pending:
        return 0
    with sequence_lock:
        sequence_number = max(sequence_number, max(sequence for _priority, sequence, _item in pending))
    for priority, sequence, item in pending:
        item["_queue_priority"] = priority
        item["_queue_sequence"] = sequence
        _queue_pending_item(
            item,
            delay=max(
                0.0,
                float(item.get("_pending_next_attempt_at") or 0) - time.time(),
            ),
        )
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


def acquire_reply_slot(
    *,
    block: bool = True,
    reserve_slots: int = 0,
    deadline: float | None = None,
) -> bool:
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
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or sleep_for > remaining:
                return False
        time.sleep(sleep_for)


def wait_for_rate_limit(deadline: float | None = None) -> bool:
    return acquire_reply_slot(deadline=deadline)


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
    scene_context: str = "",
    memory_candidates: Sequence[str] = (),
    mentions_other: bool = False,
    reply_target_user_id: str = "",
    user_id: str = "",
    sender_role: str = "",
    planner_timeout: int | None = None,
    plan_out: list[MessagePlan] | None = None,
    explicit_knowledge_command: bool = False,
) -> ProcessingDecision:
    return message_router.should_process_message(
        runtime_dependencies,
        question,
        mentioned,
        effective_question,
        followup_of,
        followup_scope,
        group_id,
        chat_context,
        scene_context,
        memory_candidates,
        mentions_other,
        reply_target_user_id,
        user_id,
        sender_role,
        planner_timeout,
        plan_out,
        explicit_knowledge_command,
    )


def process_worker_item(
    item: dict,
    lane: str,
    lifecycle: PendingItemLifecycle | None = None,
) -> PendingItemLifecycle:
    return worker_handlers.process_worker_item(
        runtime_dependencies,
        item,
        lane,
        lifecycle,
    )



def worker(work_queue: queue.PriorityQueue, lane: str) -> None:
    while True:
        priority, seq, item = work_queue.get()
        if normal_lane_should_yield(
            lane,
            priority_pending=not message_queue.empty(),
        ):
            work_queue.put((priority, seq, item))
            work_queue.task_done()
            time.sleep(0.05)
            continue
        lifecycle = PendingItemLifecycle(item)
        try:
            process_worker_item(item, lane, lifecycle)
        finally:
            lifecycle.acknowledge(
                delete_pending_message,
                lambda exc: print("Pending queue acknowledge failed:", repr(exc)),
            )
            work_queue.task_done()


def process_chat_item(
    item: dict,
    decision: ProcessingDecision,
    lifecycle: PendingItemLifecycle | None = None,
) -> PendingItemLifecycle:
    return worker_handlers.process_chat_item(
        runtime_dependencies,
        item,
        decision,
        lifecycle,
    )



def chat_worker() -> None:
    while True:
        item, decision = chat_queue.get()
        lifecycle = PendingItemLifecycle(item)
        try:
            process_chat_item(item, decision, lifecycle)
        finally:
            lifecycle.acknowledge(
                delete_pending_message,
                lambda exc: print(
                    "Chat pending queue acknowledge failed:",
                    repr(exc),
                ),
            )
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
            scene_groups, scene_updating = chat_scene_state.counts()
            memory_status = chat_memory_manager.status() if chat_memory_manager else {}
            pending_counts = pending_status_counts()
            planner_health = semantic_planner_health_snapshot()
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
                    "fragment_buffered": fragment_aggregator.buffered_count(),
                    "pending": pending_counts.get("queued", 0) + pending_counts.get("retry", 0),
                    "pending_retry": pending_counts.get("retry", 0),
                    "pending_dispatching": pending_counts.get("dispatching", 0),
                    "pending_dead_letter": pending_counts.get("dead_letter", 0),
                    "pending_sent_unknown": pending_counts.get("sent_unknown", 0),
                    "scene_groups": scene_groups,
                    "scene_updating": scene_updating,
                    "semantic_planner": planner_health,
                    "memory_messages": memory_status.get("messages", 0),
                    "memory_chunks": memory_status.get("chunks", 0),
                    "memory_topic_relations": memory_status.get("topic_relations", 0),
                    "memory_queued": memory_status.get("queued", 0),
                    "memory_paused": memory_status.get("paused", False),
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

        if event.get("post_type") == "notice" and event.get("notice_type") == "group_recall":
            group_id = event.get("group_id")
            if not settings.allowed_group_ids or str(group_id) in settings.allowed_group_ids:
                try:
                    recall_group_chat_message(int(group_id), str(event.get("message_id") or ""))
                    schedule_chat_history_save()
                except (TypeError, ValueError):
                    pass
            self._json(200, {"ok": True, "recalled": True})
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
        received_time = time.time()
        text = extract_plain_text(raw_message)
        context_text = extract_context_text(raw_message)
        content_segments = extract_content_segments(raw_message)
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
            reply_target_user_id, reply_text = resolve_reply_message_context(
                numeric_group_id,
                reply_message_id,
            )

        chat_sequence = record_group_chat_message(
            numeric_group_id,
            user_id,
            context_text,
            event.get("time"),
            message_id=str(event.get("message_id") or ""),
            reply_message_id=reply_message_id,
            reply_target_user_id=reply_target_user_id,
            reply_text=reply_text,
            mentioned_bot=mentioned,
            mentioned_user_ids=mentioned_user_ids,
            display_name=(
                event.get("sender", {}).get("card")
                or event.get("sender", {}).get("nickname")
                or ""
            ),
            received_time=received_time,
            content_segments=content_segments,
        )
        schedule_chat_history_save()
        schedule_chat_scene_update(numeric_group_id, chat_sequence)

        try:
            flush_fragment_buffer_for_new_speaker(
                numeric_group_id,
                user_id,
                defer_dispatch=True,
            )
        except Exception as exc:
            print("Fragment queue write failed:", repr(exc))
            self._json(503, {"ok": False, "error": "queue unavailable"})
            return

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
            flush_group_fragment_buffer(numeric_group_id, defer_dispatch=True)
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
        explicit_knowledge_command = bool(
            settings.command_prefix
            and text.strip().startswith(settings.command_prefix)
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
            "reply_text": reply_text,
            "message_id": str(event.get("message_id") or ""),
            "chat_sequence": chat_sequence,
            "received_time": received_time,
            "content_segments": list(content_segments),
            "message_status": "active",
            "explicit_knowledge_command": explicit_knowledge_command,
        }
        fragment_audience = classify_fragment_audience(item)
        try:
            pending_ids = submit_message_fragment(item, defer_dispatch=True)
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

        if fragment_audience == "human":
            write_message_audit(
                decision="ignored",
                reason="fragment directed at another member",
                group_id=group_id,
                user_id=user_id,
                question=question,
                mentioned=False,
                event_time=event.get("time"),
            )
            self._json(200, {"ok": True, "ignored": "directed at another member"})
            return

        self._json(
            200,
            {
                "ok": True,
                "buffered": not pending_ids,
                "queued": bool(pending_ids),
                "mentioned": mentioned,
            },
        )


def main() -> None:
    memory_started = initialize_chat_memory()
    restored = restore_pending_messages()
    if restored:
        print(f"Restored pending messages: {restored}")
    loaded = load_chat_history()
    if loaded:
        print(f"Loaded chat history: {loaded} entries")
        migrated = migrate_loaded_chat_history_to_memory()
        if migrated:
            print(f"Queued chat history migration: {migrated} entries")
    threading.Thread(
        target=worker,
        args=(message_queue, "priority"),
        daemon=True,
    ).start()
    threading.Thread(
        target=chat_history_save_worker,
        daemon=True,
        name="chat-history-saver",
    ).start()
    threading.Thread(
        target=fragment_aggregation_worker,
        daemon=True,
        name="message-fragment-aggregator",
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
    print(f"Chat memory: {'enabled' if memory_started else 'disabled'}")
    print(
        f"Allowed groups: {','.join(settings.allowed_group_ids) or 'all'}, "
        f"max replies/min: {settings.max_replies_per_minute}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
