"""JSONL audit log (design doc §13)."""
from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger("qqbot.audit")


class Auditor:
    def __init__(self, path: str = "audit.jsonl") -> None:
        self.path = path

    def log(self, entry: dict) -> None:
        entry = dict(entry)
        entry.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:  # pragma: no cover - disk failure only
            logger.warning("audit write failed: %s", e)
