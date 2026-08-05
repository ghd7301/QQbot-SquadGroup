"""Unified async LLM call layer (design doc §18 step 2).

OpenAI-compatible chat completions. Supports JSON mode with tolerant parsing and
bounded retries. A fake callable can be injected for tests / dry-run.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import aiohttp

from .config import Config

logger = logging.getLogger("qqbot.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    data: Optional[dict]  # parsed JSON when json_mode, else None


FakeFn = Callable[[str, str], Awaitable[str]]


def _robust_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"cannot parse JSON from LLM output: {raw[:200]}")


class LLMClient:
    def __init__(
        self,
        config: Config,
        session: Optional[aiohttp.ClientSession] = None,
        fake: Optional[FakeFn] = None,
    ) -> None:
        self.config = config
        self._session = session
        self._fake = fake

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.llm_timeout_sec)
            )
        return self._session

    async def chat(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        if self._fake is not None:
            raw = await self._fake(system, user)
            return LLMResponse(text=raw, data=_robust_json(raw) if json_mode else None)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: dict = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.llm_temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                session = await self._ensure_session()
                async with session.post(
                    f"{self.config.llm_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return LLMResponse(
                        text=content,
                        data=_robust_json(content) if json_mode else None,
                    )
            except Exception as e:  # noqa: BLE001 - retry on any failure
                last_err = e
                logger.warning("LLM call failed (attempt %d): %s", attempt, e)
                await asyncio.sleep(min(2**attempt, 4))
        raise LLMError(f"LLM call failed after retries: {last_err}")

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
