#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from squad_bot.config import settings
from squad_bot.llm import answer_chat, is_chat_no_reply, normalize_model_answer


def load_cases(path: Path, terms: tuple[str, ...], limit: int) -> list[dict]:
    cases: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if record.get("reply_mode") != "chat" or not record.get("chat_context"):
            continue
        question = str(record.get("question") or "").strip()
        if not question:
            continue
        if terms and not any(term in question for term in terms):
            continue
        cases.append(record)
    if limit > 0:
        cases = cases[-limit:]
    return cases


def replay_case(record: dict, model: str, timeout: int) -> dict:
    started = time.monotonic()
    raw_answer = answer_chat(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        message=str(record.get("question") or ""),
        context=tuple(record.get("chat_context") or ()),
        scene_context=str(record.get("scene_context") or ""),
        timeout=timeout,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    declined = is_chat_no_reply(raw_answer)
    return {
        "time": record.get("time"),
        "question": record.get("question"),
        "old_decision": record.get("decision"),
        "old_answer": record.get("answer") or "NO_REPLY",
        "old_latency_ms": record.get("model_latency_ms", 0),
        "replay_model": model,
        "replay_decision": "skipped" if declined else "answered",
        "replay_answer": "NO_REPLY" if declined else normalize_model_answer(raw_answer),
        "replay_latency_ms": latency_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay audited chat contexts without sending QQ messages.")
    parser.add_argument("--audit-log", type=Path, default=Path(settings.message_audit_log))
    parser.add_argument("--model", default=settings.chat_model)
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Keep cases whose current question contains this text; may be repeated.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.audit_log, tuple(args.contains), args.limit)
    if not cases:
        print("No matching chat audit cases found.", file=sys.stderr)
        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(replay_case, case, args.model, args.timeout) for case in cases]
        results = [future.result() for future in futures]

    output = "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
