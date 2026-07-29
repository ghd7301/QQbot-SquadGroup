import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from squad_bot.observability.audit import (
    _rotate_log_if_needed,
    recent_audit_entries,
    write_message_audit,
)


class AuditTests(unittest.TestCase):
    def test_message_audit_round_trip_and_recent_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "message_audit.jsonl"
            deps = SimpleNamespace(
                settings=SimpleNamespace(
                    message_audit_log=str(path),
                    knowledge_strong_min_score=0.18,
                    knowledge_strong_min_coverage=0.6,
                    bot_qq="999",
                ),
                audit_lock=threading.Lock(),
            )

            write_message_audit(
                deps,
                decision="answered",
                reason="sent",
                question="第一个问题",
            )
            write_message_audit(
                deps,
                decision="skipped",
                reason="no context",
                group_id=100,
                question="第二个问题",
                retrieval_score=0.2,
                retrieval_coverage=0.7,
                sources=("guide.md",),
            )

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            recent = recent_audit_entries(deps)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["knowledge_strength"], "strong")
        self.assertEqual(records[1]["group_id"], "100")
        self.assertEqual([entry["question"] for entry in recent], ["第二个问题"])

    def test_rotation_keeps_previous_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            path.write_text("old record\n", encoding="utf-8")

            _rotate_log_if_needed(path, max_bytes=1, keep=2)

            self.assertFalse(path.exists())
            self.assertEqual(
                path.with_suffix(".jsonl.1").read_text(encoding="utf-8"),
                "old record\n",
            )


if __name__ == "__main__":
    unittest.main()
