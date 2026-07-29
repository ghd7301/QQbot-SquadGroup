import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("BOT_HOST", "127.0.0.1")
    port: int = int(os.getenv("BOT_PORT", "8088"))
    command_prefix: str = os.getenv("COMMAND_PREFIX", "/问")
    bot_qq: str = os.getenv("BOT_QQ", "")
    member_id_secret: str = os.getenv("MEMBER_ID_SECRET", "")
    admin_qq_ids: tuple[str, ...] = tuple(
        qq_id.strip()
        for qq_id in os.getenv("ADMIN_QQ_IDS", "").split(",")
        if qq_id.strip()
    )
    onebot_access_token: str = os.getenv("ONEBOT_ACCESS_TOKEN", "")
    onebot_api_url: str = os.getenv("ONEBOT_API_URL", "http://127.0.0.1:3000")
    onebot_message_lookup_timeout_seconds: float = float(
        os.getenv("ONEBOT_MESSAGE_LOOKUP_TIMEOUT_SECONDS", "3")
    )
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    chat_model: str = os.getenv("CHAT_MODEL") or os.getenv("LLM_MODEL", "deepseek-chat")
    knowledge_dir: str = os.getenv("KNOWLEDGE_DIR", "knowledge")
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "4500"))
    knowledge_strong_min_score: float = float(os.getenv("KNOWLEDGE_STRONG_MIN_SCORE", "0.18"))
    knowledge_strong_min_coverage: float = float(os.getenv("KNOWLEDGE_STRONG_MIN_COVERAGE", "0.6"))
    knowledge_gap_log_enabled: bool = os.getenv(
        "KNOWLEDGE_GAP_LOG_ENABLED", "false"
    ).lower() in {"1", "true", "yes", "on"}
    knowledge_gap_log: str = os.getenv("KNOWLEDGE_GAP_LOG", "work/knowledge_gaps.jsonl")
    knowledge_gap_dedupe_seconds: int = int(os.getenv("KNOWLEDGE_GAP_DEDUPE_SECONDS", "3600"))
    max_answer_chars: int = int(os.getenv("MAX_ANSWER_CHARS", "500"))
    dry_run: bool = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
    max_replies_per_minute: int = int(os.getenv("MAX_REPLIES_PER_MINUTE", "8"))
    normal_message_max_age_seconds: int = int(
        os.getenv("NORMAL_MESSAGE_MAX_AGE_SECONDS", os.getenv("MAX_MESSAGE_AGE_SECONDS", "60"))
    )
    mentioned_message_max_age_seconds: int = int(
        os.getenv("MENTIONED_MESSAGE_MAX_AGE_SECONDS", "300")
    )
    normal_reply_delay_seconds: float = float(os.getenv("NORMAL_REPLY_DELAY_SECONDS", "0"))
    message_fragment_debounce_seconds: float = float(os.getenv("MESSAGE_FRAGMENT_DEBOUNCE_SECONDS", "3"))
    message_fragment_max_wait_seconds: float = float(os.getenv("MESSAGE_FRAGMENT_MAX_WAIT_SECONDS", "8"))
    message_fragment_max_parts: int = int(os.getenv("MESSAGE_FRAGMENT_MAX_PARTS", "6"))
    message_fragment_max_chars: int = int(os.getenv("MESSAGE_FRAGMENT_MAX_CHARS", "800"))
    message_fragment_semantic_enabled: bool = os.getenv(
        "MESSAGE_FRAGMENT_SEMANTIC_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    message_fragment_semantic_model: str = os.getenv("MESSAGE_FRAGMENT_SEMANTIC_MODEL") or os.getenv(
        "LLM_MODEL", "deepseek-chat"
    )
    message_fragment_semantic_timeout_seconds: int = int(
        os.getenv("MESSAGE_FRAGMENT_SEMANTIC_TIMEOUT_SECONDS", "8")
    )
    message_fragment_semantic_min_confidence: float = float(
        os.getenv("MESSAGE_FRAGMENT_SEMANTIC_MIN_CONFIDENCE", "0.75")
    )
    contextual_query_enabled: bool = os.getenv(
        "CONTEXTUAL_QUERY_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    contextual_query_model: str = os.getenv("CONTEXTUAL_QUERY_MODEL") or os.getenv(
        "LLM_MODEL", "deepseek-chat"
    )
    contextual_query_timeout_seconds: int = int(
        os.getenv("CONTEXTUAL_QUERY_TIMEOUT_SECONDS", "8")
    )
    contextual_query_min_confidence: float = float(
        os.getenv("CONTEXTUAL_QUERY_MIN_CONFIDENCE", "0.75")
    )
    semantic_planner_enabled: bool = os.getenv(
        "SEMANTIC_PLANNER_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    semantic_planner_model: str = os.getenv("SEMANTIC_PLANNER_MODEL") or os.getenv(
        "CHAT_MODEL"
    ) or os.getenv("LLM_MODEL", "deepseek-chat")
    semantic_planner_timeout_seconds: int = int(
        os.getenv("SEMANTIC_PLANNER_TIMEOUT_SECONDS", "5")
    )
    semantic_planner_addressed_timeout_seconds: int = int(
        os.getenv("SEMANTIC_PLANNER_ADDRESSED_TIMEOUT_SECONDS", "6")
    )
    semantic_planner_min_confidence: float = float(
        os.getenv("SEMANTIC_PLANNER_MIN_CONFIDENCE", "0.68")
    )
    semantic_planner_context_messages: int = int(
        os.getenv("SEMANTIC_PLANNER_CONTEXT_MESSAGES", "8")
    )
    semantic_planner_context_max_chars: int = int(
        os.getenv("SEMANTIC_PLANNER_CONTEXT_MAX_CHARS", "2400")
    )
    semantic_planner_memory_max_chars: int = int(
        os.getenv("SEMANTIC_PLANNER_MEMORY_MAX_CHARS", "800")
    )
    semantic_replan_enabled: bool = os.getenv(
        "SEMANTIC_REPLAN_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    semantic_planner_circuit_failures: int = int(
        os.getenv("SEMANTIC_PLANNER_CIRCUIT_FAILURES", "5")
    )
    semantic_planner_circuit_seconds: int = int(
        os.getenv("SEMANTIC_PLANNER_CIRCUIT_SECONDS", "30")
    )
    chat_relevance_check_enabled: bool = os.getenv(
        "CHAT_RELEVANCE_CHECK_ENABLED", "true"
    ).lower() in {"1", "true", "yes", "on"}
    chat_relevance_check_model: str = os.getenv("CHAT_RELEVANCE_CHECK_MODEL") or os.getenv(
        "CHAT_MODEL"
    ) or os.getenv("LLM_MODEL", "deepseek-chat")
    chat_relevance_check_timeout_seconds: int = int(
        os.getenv("CHAT_RELEVANCE_CHECK_TIMEOUT_SECONDS", "10")
    )
    final_reply_review_model: str = os.getenv("FINAL_REPLY_REVIEW_MODEL") or os.getenv(
        "CHAT_MODEL"
    ) or os.getenv("LLM_MODEL", "deepseek-chat")
    final_reply_review_timeout_seconds: int = int(
        os.getenv("FINAL_REPLY_REVIEW_TIMEOUT_SECONDS", "4")
    )
    final_reply_review_mode: str = os.getenv("FINAL_REPLY_REVIEW_MODE", "adaptive").strip().lower()
    mentioned_reply_total_timeout_seconds: int = int(
        os.getenv("MENTIONED_REPLY_TOTAL_TIMEOUT_SECONDS", "15")
    )
    normal_reply_total_timeout_seconds: int = int(
        os.getenv("NORMAL_REPLY_TOTAL_TIMEOUT_SECONDS", "15")
    )
    knowledge_generation_timeout_seconds: int = int(
        os.getenv("KNOWLEDGE_GENERATION_TIMEOUT_SECONDS", "10")
    )
    chat_generation_timeout_seconds: int = int(
        os.getenv("CHAT_GENERATION_TIMEOUT_SECONDS", "7")
    )
    same_topic_cooldown_seconds: int = int(os.getenv("SAME_TOPIC_COOLDOWN_SECONDS", "60"))
    message_audit_log: str = os.getenv("MESSAGE_AUDIT_LOG", "work/message_audit.jsonl")
    pending_queue_db: str = os.getenv("PENDING_QUEUE_DB", "work/pending_queue.sqlite3")
    pending_retry_max_attempts: int = int(os.getenv("PENDING_RETRY_MAX_ATTEMPTS", "3"))
    chat_memory_enabled: bool = os.getenv("CHAT_MEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    chat_memory_db: str = os.getenv("CHAT_MEMORY_DB", "work/chat_memory.sqlite3")
    chat_memory_shadow_mode: bool = os.getenv("CHAT_MEMORY_SHADOW_MODE", "false").lower() in {"1", "true", "yes", "on"}
    chat_memory_allowed_group_ids: tuple[str, ...] = tuple(
        group_id.strip()
        for group_id in os.getenv("CHAT_MEMORY_ALLOWED_GROUP_IDS", "").split(",")
        if group_id.strip()
    )
    chat_memory_max_hits: int = int(os.getenv("CHAT_MEMORY_MAX_HITS", "6"))
    chat_memory_max_chars: int = int(os.getenv("CHAT_MEMORY_MAX_CHARS", "2400"))
    bot_self_history_max_turns: int = int(os.getenv("BOT_SELF_HISTORY_MAX_TURNS", "3"))
    bot_self_history_max_chars: int = int(os.getenv("BOT_SELF_HISTORY_MAX_CHARS", "1200"))
    chat_memory_probe_max_hits: int = int(os.getenv("CHAT_MEMORY_PROBE_MAX_HITS", "8"))
    chat_memory_probe_max_chars: int = int(os.getenv("CHAT_MEMORY_PROBE_MAX_CHARS", "1600"))
    planned_chat_context_messages: int = int(os.getenv("PLANNED_CHAT_CONTEXT_MESSAGES", "10"))
    planned_chat_context_max_chars: int = int(os.getenv("PLANNED_CHAT_CONTEXT_MAX_CHARS", "1600"))
    chat_memory_retention_days: int = int(os.getenv("CHAT_MEMORY_RETENTION_DAYS", "90"))
    chat_memory_embedding_provider: str = os.getenv("CHAT_MEMORY_EMBEDDING_PROVIDER", "hashed")
    chat_memory_embedding_base_url: str = os.getenv("CHAT_MEMORY_EMBEDDING_BASE_URL", "")
    chat_memory_embedding_api_key: str = os.getenv("CHAT_MEMORY_EMBEDDING_API_KEY", "")
    chat_memory_embedding_model: str = os.getenv("CHAT_MEMORY_EMBEDDING_MODEL", "")
    chat_memory_embedding_dimensions: int = int(os.getenv("CHAT_MEMORY_EMBEDDING_DIMENSIONS", "384"))
    chat_memory_embedding_timeout_seconds: int = int(os.getenv("CHAT_MEMORY_EMBEDDING_TIMEOUT_SECONDS", "15"))
    auto_reply_enabled: bool = os.getenv("AUTO_REPLY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    llm_fallback_enabled: bool = os.getenv("LLM_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    fallback_only_when_mentioned: bool = os.getenv("FALLBACK_ONLY_WHEN_MENTIONED", "true").lower() in {"1", "true", "yes", "on"}
    chat_reply_enabled: bool = os.getenv("CHAT_REPLY_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    chat_reply_cooldown_seconds: int = int(os.getenv("CHAT_REPLY_COOLDOWN_SECONDS", "60"))
    max_chat_replies_per_hour: int = int(os.getenv("MAX_CHAT_REPLIES_PER_HOUR", "20"))
    chat_context_seconds: int = int(os.getenv("CHAT_CONTEXT_SECONDS", "300"))
    chat_context_messages: int = int(os.getenv("CHAT_CONTEXT_MESSAGES", "12"))
    chat_reply_debounce_seconds: float = float(os.getenv("CHAT_REPLY_DEBOUNCE_SECONDS", "2"))
    chat_scene_enabled: bool = os.getenv("CHAT_SCENE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    chat_scene_model: str = os.getenv("CHAT_SCENE_MODEL") or os.getenv("CHAT_MODEL") or os.getenv(
        "LLM_MODEL", "deepseek-chat"
    )
    chat_scene_debounce_seconds: float = float(os.getenv("CHAT_SCENE_DEBOUNCE_SECONDS", "3"))
    chat_scene_update_interval_seconds: float = float(os.getenv("CHAT_SCENE_UPDATE_INTERVAL_SECONDS", "30"))
    chat_scene_stale_seconds: int = int(os.getenv("CHAT_SCENE_STALE_SECONDS", "600"))
    chat_scene_min_messages: int = int(os.getenv("CHAT_SCENE_MIN_MESSAGES", "3"))
    chat_scene_timeout_seconds: int = int(os.getenv("CHAT_SCENE_TIMEOUT_SECONDS", "30"))
    chat_allowed_group_ids: tuple[str, ...] = tuple(
        group_id.strip()
        for group_id in os.getenv("CHAT_ALLOWED_GROUP_IDS", "").split(",")
        if group_id.strip()
    )
    allowed_group_ids: tuple[str, ...] = tuple(
        group_id.strip()
        for group_id in os.getenv("ALLOWED_GROUP_IDS", "").split(",")
        if group_id.strip()
    )


settings = Settings()
