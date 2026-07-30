from __future__ import annotations

import json
import queue
import re
import sqlite3
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer
from typing import Sequence

from . import pending_store
from . import chat_history as chat_history_service
from .message_fragments import (
    classify_audience,
    items_compatible,
    message_ids,
)
from . import admin as admin_service
from . import message_router, semantic_routing, worker_handlers
from .delivery import policies as delivery_policies
from .delivery import replies as reply_delivery
from .delivery import review as delivery_review
from .ingress import events as ingress_events
from .ingress import fragments as fragment_ingress
from .observability import audit as audit_observability
from .queueing import dispatcher as queue_dispatcher
from .queueing import store as queue_store
from .transport import http as http_transport
from .config import settings
from .chat_memory import ChatMemoryManager, ChatMemoryStore, MemoryHit, MemoryMessage, redact_for_model
from .embedding import build_embedding_provider
from .fact_guard import (
    candidate_knowledge_segments,
    normalize_chinese_number,
    precise_fact_tokens,
    unsupported_fallback_precise_facts,
)
from .knowledge import ContextResult
from .knowledge_routing import KnowledgeRoutingService, attach_result
from .runtime import BotRuntime
from .runtime_dependencies import RuntimeDependencies
from .worker_runtime import (
    PendingItemLifecycle,
    normal_lane_should_yield,
    run_chat_worker,
    run_worker,
)
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


bot_runtime = BotRuntime(settings.knowledge_dir)
kb = bot_runtime.knowledge.base
message_queue = bot_runtime.queues.priority
normal_message_queue = bot_runtime.queues.normal
chat_queue = bot_runtime.queues.chat
reply_timestamps = bot_runtime.queues.reply_timestamps
rate_limit_lock = bot_runtime.queues.rate_limit_lock
kb_lock = bot_runtime.knowledge.lock
sequence_lock = bot_runtime.queues.sequence_lock
sequence_number = 0
audit_lock = bot_runtime.audit_lock
auto_reply_enabled = settings.auto_reply_enabled
topic_cooldown_lock = bot_runtime.conversation.topic_cooldown_lock
recent_reply_topics = bot_runtime.conversation.recent_reply_topics
chat_history_state = bot_runtime.conversation.history
chat_history_lock = chat_history_state.lock
group_chat_history = chat_history_state.messages
chat_message_sequence = 0
chat_reply_lock = bot_runtime.conversation.chat_reply_lock
group_send_locks_lock = bot_runtime.conversation.group_send_locks_lock
group_send_locks = bot_runtime.conversation.group_send_locks
chat_scene_state = bot_runtime.conversation.scene
chat_scene_lock = chat_scene_state.lock
group_chat_scenes = chat_scene_state.scenes
chat_scene_requested_sequence = chat_scene_state.requested_sequences
chat_scene_pending_messages = chat_scene_state.pending_messages
chat_scene_running = chat_scene_state.running
hostile_reply_lock = bot_runtime.conversation.hostile_reply_lock
hostile_reply_history = bot_runtime.conversation.hostile_reply_history
memory_clear_confirmations = bot_runtime.conversation.memory_clear_confirmations
fragment_aggregator = bot_runtime.conversation.fragments
fragment_condition = fragment_aggregator.condition
group_fragment_buffers = fragment_aggregator.buffers
ready_fragment_buffers = fragment_aggregator.ready
chat_memory_manager: ChatMemoryManager | None = None
chat_history_save_event = bot_runtime.conversation.history_save_event
knowledge_gap_lock = bot_runtime.knowledge.gap_lock
recent_knowledge_gap_queries = bot_runtime.knowledge.recent_gap_queries
semantic_planner_health = bot_runtime.planner_health
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


def _rotate_log_if_needed(
    log_path: Path, max_bytes: int = 5 * 1024 * 1024, keep: int = 5
) -> None:
    return audit_observability._rotate_log_if_needed(log_path, max_bytes, keep)



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
    event_time=None
) -> None:
    return audit_observability.write_message_audit(
        runtime_dependencies,
        decision=decision,
        reason=reason,
        group_id=group_id,
        user_id=user_id,
        question=question,
        mentioned=mentioned,
        has_context=has_context,
        sources=sources,
        followup_of=followup_of,
        followup_scope=followup_scope,
        reply_mode=reply_mode,
        retrieval_score=retrieval_score,
        retrieval_coverage=retrieval_coverage,
        model_latency_ms=model_latency_ms,
        model_name=model_name,
        reply_message_id=reply_message_id,
        reply_target_user_id=reply_target_user_id,
        chat_context=chat_context,
        scene_context=scene_context,
        answer=answer,
        mention_user_id=mention_user_id,
        semantic_intent=semantic_intent,
        semantic_topic=semantic_topic,
        implicit_meaning=implicit_meaning,
        capability=capability,
        semantic_confidence=semantic_confidence,
        topic_candidates=topic_candidates,
        subject_candidates=subject_candidates,
        subject_ambiguity=subject_ambiguity,
        bot_involvement=bot_involvement,
        reply_perspective=reply_perspective,
        semantic_audience=semantic_audience,
        participation_role=participation_role,
        plan_context_revision=plan_context_revision,
        plan_scene_version=plan_scene_version,
        related_message_ids=related_message_ids,
        semantic_replan_count=semantic_replan_count,
        semantic_replan_reason=semantic_replan_reason,
        planner_status=planner_status,
        planner_latency_ms=planner_latency_ms,
        memory_query=memory_query,
        memory_hit_count=memory_hit_count,
        memory_retrieval_attempted=memory_retrieval_attempted,
        memory_retrieval_mode=memory_retrieval_mode,
        memory_candidate_count=memory_candidate_count,
        memory_rejection_reason=memory_rejection_reason,
        recent_context_candidate_count=recent_context_candidate_count,
        recent_context_selected_count=recent_context_selected_count,
        recent_context_chars=recent_context_chars,
        memory_context_chars=memory_context_chars,
        context_deduplicated_count=context_deduplicated_count,
        recent_context_selected_ids=recent_context_selected_ids,
        memory_selected_chunk_ids=memory_selected_chunk_ids,
        memory_selected_by_planner=memory_selected_by_planner,
        self_history_candidate_count=self_history_candidate_count,
        self_history_selected_count=self_history_selected_count,
        self_history_chars=self_history_chars,
        self_history_selected_message_ids=self_history_selected_message_ids,
        self_history_reasons=self_history_reasons,
        bot_message_id=bot_message_id,
        generated_for_message_ids=generated_for_message_ids,
        turn_id=turn_id,
        event_time=event_time,
    )



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
    return admin_service.normalize_command_text(runtime_dependencies, message)



def get_admin_command(message: str) -> str:
    return admin_service.get_admin_command(runtime_dependencies, message)



def is_restored_admin_command(item: dict) -> bool:
    return admin_service.is_restored_admin_command(runtime_dependencies, item)



def is_admin_user(user_id, sender_role: str = "") -> bool:
    return admin_service.is_admin_user(runtime_dependencies, user_id, sender_role)



def recent_audit_entries(limit: int = 5) -> list[dict]:
    return audit_observability.recent_audit_entries(runtime_dependencies, limit)



def answer_admin_command(command: str, *, group_id: int = 0, user_id: str = "") -> str:
    return admin_service.answer_admin_command(
        runtime_dependencies, command, group_id=group_id, user_id=user_id
    )



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
    return chat_history_service.record_group_chat_message(
        runtime_dependencies,
        group_id,
        user_id,
        text,
        event_time,
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


def find_group_chat_message(group_id: int, message_id: str) -> GroupChatMessage | None:
    return chat_history_service.find_group_chat_message(
        runtime_dependencies, group_id, message_id
    )


def resolve_reply_message_context(
    group_id: int,
    message_id: str,
    *,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    return chat_history_service.resolve_reply_message_context(
        runtime_dependencies, group_id, message_id, db_path=db_path
    )


def stable_member_id(group_id: int, user_id: str) -> str:
    return chat_history_service.stable_member_id(
        runtime_dependencies, group_id, user_id
    )


def _context_message_payload(
    group_id: int,
    item: GroupChatMessage,
    *,
    current: bool,
) -> dict:
    return chat_history_service.context_message_payload(
        runtime_dependencies, group_id, item, current=current
    )


def recent_group_chat_context(
    group_id: int,
    *,
    now: float | None = None,
    context_seconds: int | None = None,
    max_messages: int | None = None,
    focus_sequence: int = 0,
    through_sequence: int = 0,
) -> tuple[str, ...]:
    return chat_history_service.recent_group_chat_context(
        runtime_dependencies,
        group_id,
        now=now,
        context_seconds=context_seconds,
        max_messages=max_messages,
        focus_sequence=focus_sequence,
        through_sequence=through_sequence,
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
    return delivery_review.locked_send_context_change(
        runtime_dependencies, group_id, reviewed_revision, item
    )


def validate_locked_send(
    group_id: int,
    item: dict,
    reviewed_revision: int,
    *,
    check_context: bool = True,
) -> tuple[bool, str, str]:
    return delivery_review.validate_locked_send(
        runtime_dependencies,
        group_id,
        item,
        reviewed_revision,
        check_context=check_context,
    )


def unsafe_or_repeated_reply(group_id: int, answer: str, *, limit: int = 10) -> str:
    return delivery_review.unsafe_or_repeated_reply(
        runtime_dependencies, group_id, answer, limit=limit
    )


def is_recent_duplicate_group_message(
    group_id: int,
    text: str,
    *,
    focus_sequence: int,
    event_time=None,
    window_seconds: int = 60,
) -> bool:
    return delivery_review.is_recent_duplicate_group_message(
        runtime_dependencies,
        group_id,
        text,
        focus_sequence=focus_sequence,
        event_time=event_time,
        window_seconds=window_seconds,
    )


def reply_deadline(event_time, mentioned: bool) -> float:
    return delivery_review.reply_deadline(
        runtime_dependencies, event_time, mentioned
    )


def remaining_reply_timeout(
    deadline: float,
    *,
    cap: int,
    reserve: int = 0,
) -> int:
    return delivery_review.remaining_reply_timeout(
        deadline, cap=cap, reserve=reserve
    )


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
    return delivery_review.review_and_refresh_answer(
        runtime_dependencies,
        question=question,
        answer=answer,
        decision=decision,
        group_id=group_id,
        mentioned=mentioned,
        admin=admin,
        deadline=deadline,
        baseline_revision=baseline_revision,
        target_item=target_item,
    )


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
    return delivery_review.refresh_answer_for_late_context(
        runtime_dependencies,
        question=question,
        answer=answer,
        decision=decision,
        group_id=group_id,
        mentioned=mentioned,
        admin=admin,
        deadline=deadline,
        reviewed_revision=reviewed_revision,
        target_item=target_item,
    )


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
    return audit_observability.record_knowledge_gap(runtime_dependencies, query, result)



def recent_knowledge_gap_entries(limit: int = 5) -> list[dict]:
    return audit_observability.recent_knowledge_gap_entries(runtime_dependencies, limit)



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
    return queue_dispatcher.next_sequence(runtime_dependencies)



def _pending_db_path(db_path: str | Path | None = None) -> str | Path:
    return queue_store._pending_db_path(runtime_dependencies, db_path)



def open_pending_queue_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    return queue_store.open_pending_queue_db(runtime_dependencies, db_path)



def persist_conversation_turn(
    group_id: int, state: ConversationState, *, db_path: str | Path | None = None
) -> int:
    return queue_store.persist_conversation_turn(
        runtime_dependencies, group_id, state, db_path=db_path
    )



def _conversation_state_from_row(row) -> ConversationState:
    return queue_store._conversation_state_from_row(runtime_dependencies, row)



def load_conversation_turn_by_bot_message_id(
    group_id: int, bot_message_id: str, *, db_path: str | Path | None = None
) -> ConversationState | None:
    return queue_store.load_conversation_turn_by_bot_message_id(
        runtime_dependencies, group_id, bot_message_id, db_path=db_path
    )



def persist_pending_message(
    priority: int, sequence: int, item: dict, db_path: str | Path | None = None
) -> int:
    return queue_store.persist_pending_message(
        runtime_dependencies, priority, sequence, item, db_path
    )



def load_pending_messages(
    db_path: str | Path | None = None, *, include_future: bool = False
) -> list[tuple[int, int, dict]]:
    return queue_store.load_pending_messages(
        runtime_dependencies, db_path, include_future=include_future
    )



def mark_pending_failure(
    pending_id: int,
    error: str,
    *,
    db_path: str | Path | None = None,
    now: float | None = None,
    max_attempts: int | None = None
) -> PendingFailureResult:
    return queue_store.mark_pending_failure(
        runtime_dependencies,
        pending_id,
        error,
        db_path=db_path,
        now=now,
        max_attempts=max_attempts,
    )



def mark_pending_sent_unknown(
    pending_id: int, error: str, *, db_path: str | Path | None = None
) -> None:
    return queue_store.mark_pending_sent_unknown(
        runtime_dependencies, pending_id, error, db_path=db_path
    )



def mark_pending_dispatch_started(
    pending_id: int, dispatch_id: str, *, db_path: str | Path | None = None
) -> None:
    return queue_store.mark_pending_dispatch_started(
        runtime_dependencies, pending_id, dispatch_id, db_path=db_path
    )



def recover_incomplete_pending_dispatches(db_path: str | Path | None = None) -> int:
    return queue_store.recover_incomplete_pending_dispatches(
        runtime_dependencies, db_path
    )



def delete_pending_message(pending_id: int, db_path: str | Path | None = None) -> None:
    return queue_store.delete_pending_message(runtime_dependencies, pending_id, db_path)



def pending_message_count(db_path: str | Path | None = None) -> int:
    return queue_store.pending_message_count(runtime_dependencies, db_path)



def pending_status_counts(db_path: str | Path | None = None) -> dict[str, int]:
    return queue_store.pending_status_counts(runtime_dependencies, db_path)



def enqueue_persistent_message(priority: int, item: dict) -> int:
    return queue_dispatcher.enqueue_persistent_message(
        runtime_dependencies, priority, item
    )



def _queue_pending_item(item: dict, *, delay: float = 0.0) -> None:
    return queue_dispatcher._queue_pending_item(runtime_dependencies, item, delay=delay)



def handle_pending_worker_failure(item: dict, error: str) -> str:
    return queue_dispatcher.handle_pending_worker_failure(
        runtime_dependencies, item, error
    )



def begin_pending_dispatch(item: dict) -> None:
    return queue_dispatcher.begin_pending_dispatch(runtime_dependencies, item)



def classify_fragment_audience(item: dict) -> str:
    return fragment_ingress.classify_fragment_audience(runtime_dependencies, item)



def fragment_items_compatible(
    buffer: MessageFragmentBuffer, item: dict, audience: str
) -> bool:
    return fragment_ingress.fragment_items_compatible(
        runtime_dependencies, buffer, item, audience
    )



def _message_ids(item: dict) -> list[str]:
    return reply_delivery.message_ids(runtime_dependencies, item)


def bot_turn_metadata(item: dict, bot_message_id) -> tuple[tuple[str, ...], str]:
    return reply_delivery.bot_turn_metadata(
        runtime_dependencies, item, bot_message_id
    )


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
    return reply_delivery.send_and_record_bot_turn(
        runtime_dependencies,
        group_id=group_id,
        item=item,
        answer=answer,
        reply_mode=reply_mode,
        semantic_topic=semantic_topic,
        mention_user_id=mention_user_id,
        reply_to_trigger=reply_to_trigger,
    )


def message_already_covered_by_bot(group_id: int, message_id) -> bool:
    return reply_delivery.message_already_covered_by_bot(
        runtime_dependencies, group_id, message_id
    )


def _context_message_speakers(context: Sequence[str]) -> dict[str, str]:
    return reply_delivery.context_message_speakers(context)


def merge_review_message_ids(item: dict | None, review, latest_context: Sequence[str]) -> None:
    return reply_delivery.merge_review_message_ids(
        runtime_dependencies, item, review, latest_context
    )


def _new_fragment_buffer(
    item: dict, audience: str, now: float
) -> MessageFragmentBuffer:
    return fragment_ingress._new_fragment_buffer(
        runtime_dependencies, item, audience, now
    )



def _merge_fragment(
    buffer: MessageFragmentBuffer, item: dict, audience: str, now: float
) -> None:
    return fragment_ingress._merge_fragment(
        runtime_dependencies, buffer, item, audience, now
    )



def _fragment_prefix_item(buffer: MessageFragmentBuffer, count: int) -> dict:
    return fragment_ingress._fragment_prefix_item(runtime_dependencies, buffer, count)



def semantic_bot_fragment_count(buffer: MessageFragmentBuffer) -> int:
    return fragment_ingress.semantic_bot_fragment_count(runtime_dependencies, buffer)



def _dispatch_fragment_buffer(buffer: MessageFragmentBuffer) -> int:
    return fragment_ingress._dispatch_fragment_buffer(runtime_dependencies, buffer)



def _defer_fragment_buffers(buffers: Sequence[MessageFragmentBuffer]) -> None:
    return fragment_ingress._defer_fragment_buffers(runtime_dependencies, buffers)



def clear_fragment_state() -> None:
    return fragment_ingress.clear_fragment_state(runtime_dependencies)



def flush_group_fragment_buffer(
    group_id: int, *, defer_dispatch: bool = False
) -> int | None:
    return fragment_ingress.flush_group_fragment_buffer(
        runtime_dependencies, group_id, defer_dispatch=defer_dispatch
    )



def flush_fragment_buffer_for_new_speaker(
    group_id: int, user_id, *, defer_dispatch: bool = False
) -> int | None:
    return fragment_ingress.flush_fragment_buffer_for_new_speaker(
        runtime_dependencies, group_id, user_id, defer_dispatch=defer_dispatch
    )



def submit_message_fragment(
    item: dict, *, now: float | None = None, defer_dispatch: bool = False
) -> list[int]:
    return fragment_ingress.submit_message_fragment(
        runtime_dependencies, item, now=now, defer_dispatch=defer_dispatch
    )



def fragment_aggregation_worker() -> None:
    return fragment_ingress.fragment_aggregation_worker(runtime_dependencies)



def restore_pending_messages() -> int:
    return queue_dispatcher.restore_pending_messages(runtime_dependencies)



def message_max_age_seconds(mentioned: bool) -> int:
    return delivery_policies.message_max_age_seconds(runtime_dependencies, mentioned)


def is_message_too_old(
    event_time,
    mentioned: bool,
    *,
    fallback_time=None,
    now: float | None = None,
) -> bool:
    return delivery_policies.is_message_too_old(
        runtime_dependencies,
        event_time,
        mentioned,
        fallback_time=fallback_time,
        now=now,
    )


def is_event_too_old(event: dict) -> bool:
    return delivery_policies.is_event_too_old(runtime_dependencies, event)


def acquire_reply_slot(
    *,
    block: bool = True,
    reserve_slots: int = 0,
    deadline: float | None = None,
) -> bool:
    return delivery_policies.acquire_reply_slot(
        runtime_dependencies,
        block=block,
        reserve_slots=reserve_slots,
        deadline=deadline,
    )


def wait_for_rate_limit(deadline: float | None = None) -> bool:
    return delivery_policies.wait_for_rate_limit(runtime_dependencies, deadline)


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
    return run_worker(runtime_dependencies, work_queue, lane)


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
    return run_chat_worker(runtime_dependencies)


def handle_onebot_event(event: dict) -> tuple[int, dict]:
    return ingress_events.handle_onebot_event(runtime_dependencies, event)


Handler = http_transport.create_handler(runtime_dependencies)



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
