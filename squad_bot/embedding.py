from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class HashedNgramEmbedding:
    """Dependency-free fallback. Useful for recall, weaker than a semantic model."""

    dimensions: int = 384
    name: str = "hashed-ngram-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        normalized = "".join(str(text or "").lower().split())
        vector = [0.0] * self.dimensions
        if not normalized:
            return vector
        grams: list[str] = []
        for size in (1, 2, 3):
            grams.extend(normalized[index : index + size] for index in range(len(normalized) - size + 1))
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@dataclass
class OpenAICompatibleEmbedding:
    base_url: str
    api_key: str
    model: str
    dimensions: int = 0
    timeout: int = 15

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.api_key or not self.model:
            raise RuntimeError("embedding API is not configured")
        payload = {"model": self.model, "input": list(texts)}
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        rows = result.get("data") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("embedding API returned no data")
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        vectors = [list(map(float, row.get("embedding") or ())) for row in ordered]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise RuntimeError("embedding API returned invalid vectors")
        if not self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors


def build_embedding_provider(settings) -> EmbeddingProvider:
    provider = str(getattr(settings, "chat_memory_embedding_provider", "hashed") or "hashed").lower()
    if provider in {"api", "openai", "openai-compatible"}:
        return OpenAICompatibleEmbedding(
            base_url=getattr(settings, "chat_memory_embedding_base_url", "") or settings.llm_base_url,
            api_key=getattr(settings, "chat_memory_embedding_api_key", "") or settings.llm_api_key,
            model=getattr(settings, "chat_memory_embedding_model", ""),
            dimensions=max(0, int(getattr(settings, "chat_memory_embedding_dimensions", 0))),
            timeout=max(1, int(getattr(settings, "chat_memory_embedding_timeout_seconds", 15))),
        )
    return HashedNgramEmbedding(
        dimensions=max(64, int(getattr(settings, "chat_memory_embedding_dimensions", 384) or 384))
    )
