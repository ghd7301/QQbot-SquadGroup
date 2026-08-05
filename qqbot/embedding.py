"""Embedding providers (design doc §8.3).

Two strategies, matching the agreed design:
  - HashedNgramEmbedding: dependency-free, zero-cost n-gram hash vector used for
    lexical recall. Always available.
  - ApiEmbedding: OpenAI-compatible /embeddings endpoint, enabled only when
    EMBEDDING_MODEL is configured.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod

import aiohttp

from .config import Config

logger = logging.getLogger("qqbot.embedding")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    toks: list[str] = []
    for ch in text:
        if ch.strip():
            toks.append(ch)
    for w in _TOKEN_RE.findall(text):
        toks.append(w)
    for i in range(len(text) - 1):
        toks.append(text[i : i + 2])
    return toks


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingProvider(ABC):
    dim: int = 0

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashedNgramEmbedding(EmbeddingProvider):
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for t in tokenize(text):
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class ApiEmbedding(EmbeddingProvider):
    def __init__(self, config: Config, session: aiohttp.ClientSession | None = None) -> None:
        self.config = config
        self._session = session

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.llm_timeout_sec)
            )
        return self._session

    async def embed(self, texts: list[str]) -> list[list[float]]:
        session = await self._ensure_session()
        async with session.post(
            f"{(self.config.embedding_base_url or '').rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {self.config.embedding_api_key or ''}",
                "Content-Type": "application/json",
            },
            json={"model": self.config.embedding_model, "input": texts},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [item["embedding"] for item in data["data"]]


def build_embedding(config: Config, session: aiohttp.ClientSession | None = None) -> EmbeddingProvider:
    if config.embedding_model:
        logger.info("Using API embedding model: %s", config.embedding_model)
        return ApiEmbedding(config, session)
    logger.info("Using hashed n-gram embedding (lexical fallback, no API cost)")
    return HashedNgramEmbedding()
