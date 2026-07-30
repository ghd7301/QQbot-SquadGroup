from __future__ import annotations

import queue
import threading
from pathlib import Path

from .chat_history import ChatHistoryState
from .chat_scene import ChatSceneState
from .knowledge import KnowledgeBase
from .message_fragments import FragmentAggregator
from .planner_health import SemanticPlannerHealth


class QueueRuntime:
    def __init__(self) -> None:
        self.priority: queue.PriorityQueue = queue.PriorityQueue()
        self.normal: queue.PriorityQueue = queue.PriorityQueue()
        self.chat: queue.Queue = queue.Queue()
        self.reply_timestamps: list[float] = []
        self.rate_limit_lock = threading.Lock()
        self.sequence_lock = threading.Lock()


class ConversationRuntime:
    def __init__(self) -> None:
        self.topic_cooldown_lock = threading.RLock()
        self.recent_reply_topics: dict[tuple[int, str], float] = {}

        self.history = ChatHistoryState()
        self.chat_reply_lock = threading.RLock()
        self.group_send_locks_lock = threading.Lock()
        self.group_send_locks: dict[int, threading.Lock] = {}

        self.scene = ChatSceneState()
        self.hostile_reply_lock = threading.Lock()
        self.hostile_reply_history: dict[tuple[int, str], list[float]] = {}
        self.memory_clear_confirmations: dict[tuple[int, str], float] = {}

        self.fragments = FragmentAggregator()
        self.history_save_event = threading.Event()


class KnowledgeRuntime:
    def __init__(self, knowledge_dir: str | Path) -> None:
        self.base = KnowledgeBase(knowledge_dir)
        self.lock = threading.RLock()
        self.gap_lock = threading.Lock()
        self.recent_gap_queries: dict[str, float] = {}


class BotRuntime:
    def __init__(self, knowledge_dir: str | Path) -> None:
        self.queues = QueueRuntime()
        self.conversation = ConversationRuntime()
        self.knowledge = KnowledgeRuntime(knowledge_dir)
        self.audit_lock = threading.Lock()
        self.planner_health = SemanticPlannerHealth()
