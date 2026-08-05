"""Configuration loaded from environment / .env.

All runtime knobs live here so the rest of the code reads plain attributes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional in some envs
    pass


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _int(v: str | None, default: int) -> int:
    try:
        return int(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


def _float(v: str | None, default: float) -> float:
    try:
        return float(v) if v is not None and v.strip() != "" else default
    except ValueError:
        return default


def _list_int(v: str | None) -> list[int]:
    if not v:
        return []
    out: list[int] = []
    for part in v.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


@dataclass
class Config:
    # --- LLM (OpenAI-compatible chat completions) ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_timeout_sec: float = 30.0
    llm_max_retries: int = 2

    # --- Embedding (OpenAI-compatible /embeddings). None => hash fallback ---
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None

    # --- Bot identity & groups ---
    bot_qq: str = ""
    whitelisted_groups: list[int] = field(default_factory=list)

    # --- OneBot v11 transport (we host a reverse websocket server) ---
    onebot_ws_host: str = "0.0.0.0"
    onebot_ws_port: int = 8081

    # --- Knowledge base ---
    knowledge_dir: str = "knowledge"
    knowledge_top_k: int = 5
    knowledge_threshold: float = 0.30
    lexical_weight: float = 0.5
    vector_weight: float = 0.5

    # --- Context window ---
    agg_window_sec: float = 3.0
    short_ctx_msgs: int = 16
    short_ctx_chars: int = 4000

    # --- Reply constraints ---
    max_reply_chars: int = 500
    rate_limit_per_min: int = 20

    # --- Safety ---
    dedup_window_sec: float = 30.0
    end_to_end_sec: float = 15.0

    # --- Storage ---
    db_path: str = "qqbot.db"

    # --- Runtime ---
    dry_run: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=os.getenv("LLM_API_KEY", cls.llm_api_key),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            llm_temperature=_float(os.getenv("LLM_TEMPERATURE"), cls.llm_temperature),
            llm_timeout_sec=_float(os.getenv("LLM_TIMEOUT_SEC"), cls.llm_timeout_sec),
            llm_max_retries=_int(os.getenv("LLM_MAX_RETRIES"), cls.llm_max_retries),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or None,
            embedding_api_key=os.getenv("EMBEDDING_API_KEY") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL") or None,
            bot_qq=os.getenv("BOT_QQ", cls.bot_qq),
            whitelisted_groups=_list_int(os.getenv("WHITELISTED_GROUPS")),
            onebot_ws_host=os.getenv("ONE_BOT_WS_HOST", cls.onebot_ws_host),
            onebot_ws_port=_int(os.getenv("ONE_BOT_WS_PORT"), cls.onebot_ws_port),
            knowledge_dir=os.getenv("KNOWLEDGE_DIR", cls.knowledge_dir),
            knowledge_top_k=_int(os.getenv("KNOWLEDGE_TOP_K"), cls.knowledge_top_k),
            knowledge_threshold=_float(os.getenv("KNOWLEDGE_THRESHOLD"), cls.knowledge_threshold),
            short_ctx_msgs=_int(os.getenv("SHORT_CTX_MSGS"), cls.short_ctx_msgs),
            short_ctx_chars=_int(os.getenv("SHORT_CTX_CHARS"), cls.short_ctx_chars),
            max_reply_chars=_int(os.getenv("MAX_REPLY_CHARS"), cls.max_reply_chars),
            rate_limit_per_min=_int(os.getenv("RATE_LIMIT_PER_MIN"), cls.rate_limit_per_min),
            dedup_window_sec=_float(os.getenv("DEDUP_WINDOW_SEC"), cls.dedup_window_sec),
            end_to_end_sec=_float(os.getenv("END_TO_END_SEC"), cls.end_to_end_sec),
            db_path=os.getenv("DB_PATH", cls.db_path),
            dry_run=_bool(os.getenv("DRY_RUN"), cls.dry_run),
        )
