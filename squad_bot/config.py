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
    same_topic_cooldown_seconds: int = int(os.getenv("SAME_TOPIC_COOLDOWN_SECONDS", "60"))
    followup_same_user_seconds: int = int(
        os.getenv("FOLLOWUP_SAME_USER_SECONDS", os.getenv("FOLLOWUP_CONTEXT_SECONDS", "120"))
    )
    followup_group_seconds: int = int(os.getenv("FOLLOWUP_GROUP_SECONDS", "30"))
    followup_mention_seconds: int = int(os.getenv("FOLLOWUP_MENTION_SECONDS", "180"))
    message_audit_log: str = os.getenv("MESSAGE_AUDIT_LOG", "work/message_audit.jsonl")
    pending_queue_db: str = os.getenv("PENDING_QUEUE_DB", "work/pending_queue.sqlite3")
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
    chat_scene_model: str = os.getenv("CHAT_SCENE_MODEL") or os.getenv("LLM_MODEL", "deepseek-chat")
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
