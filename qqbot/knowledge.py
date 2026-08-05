"""Knowledge base: load -> chunk -> index -> hybrid retrieve (design doc §8).

Chunks are split per Markdown heading. Lexical recall uses BM25 over the chunk
tokens; vector recall uses the configured embedding provider. The two are merged,
min-max normalised, weighted, reranked, thresholded and truncated to top-K.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field

from .embedding import EmbeddingProvider, cosine, tokenize

logger = logging.getLogger("qqbot.knowledge")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

K1 = 1.5
B = 0.75


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    section_path: list[str]
    vector: list[float] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.source}#{'/'.join(self.section_path)}"


class KnowledgeBase:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.chunks: list[Chunk] = []
        self._df: dict[str, int] = {}
        self._avgdl = 1.0

    # ---------- loading ----------
    def load(self, knowledge_dir: str) -> int:
        self.chunks = []
        for name in sorted(os.listdir(knowledge_dir)):
            if not name.lower().endswith((".md", ".markdown")):
                continue
            path = os.path.join(knowledge_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                self._ingest_file(name, f.read())
        self._build_lexical()
        return len(self.chunks)

    def _ingest_file(self, name: str, content: str) -> None:
        title = name
        stack: list[str] = []
        buf: list[str] = []
        lines = content.splitlines()

        def flush(level: int, heading: str, body: str) -> None:
            while stack and len(stack) > level:
                stack.pop()
            if heading:
                stack.append(heading)
            text = (heading + "\n" if heading else "") + body.strip()
            if text.strip():
                self.chunks.append(
                    Chunk(
                        text=text,
                        source=name,
                        title=title,
                        section_path=list(stack),
                    )
                )

        cur_level = 0
        cur_heading = ""
        for line in lines:
            m = _HEADING_RE.match(line)
            if m:
                flush(cur_level, cur_heading, "\n".join(buf))
                buf = []
                cur_level = len(m.group(1))
                cur_heading = m.group(2).strip()
            else:
                buf.append(line)
        flush(cur_level, cur_heading, "\n".join(buf))

    def _build_lexical(self) -> None:
        self._df = {}
        lens = []
        for c in self.chunks:
            c.tokens = tokenize(c.text)
            lens.append(len(c.tokens))
            for t in set(c.tokens):
                self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = (sum(lens) / len(lens)) if lens else 1.0

    async def embed_all(self) -> None:
        if not self.chunks:
            return
        texts = [c.text for c in self.chunks]
        vecs = await self.provider.embed(texts)
        for c, v in zip(self.chunks, vecs):
            c.vector = v

    # ---------- retrieval ----------
    def _bm25(self, query_tokens: list[str], chunk: Chunk) -> float:
        if not query_tokens:
            return 0.0
        n = len(self.chunks)
        score = 0.0
        dl = len(chunk.tokens)
        tf_counts: dict[str, int] = {}
        for t in chunk.tokens:
            tf_counts[t] = tf_counts.get(t, 0) + 1
        for qt in set(query_tokens):
            df = self._df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            f = tf_counts.get(qt, 0)
            score += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self._avgdl))
        return score

    async def retrieve(
        self,
        knowledge_query: str,
        user_text: str = "",
        *,
        top_k: int = 5,
        threshold: float = 0.30,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> list[Chunk]:
        if not self.chunks:
            return []
        q_tokens = tokenize(knowledge_query + " " + user_text)
        u_tokens = tokenize(user_text)
        lex_scores = [self._bm25(q_tokens, c) + self._bm25(u_tokens, c) * 0.5 for c in self.chunks]
        vec_scores = [0.0] * len(self.chunks)
        has_vec = any(len(c.vector) > 0 for c in self.chunks)
        if has_vec:
            q_vec = (await self.provider.embed([knowledge_query + " " + user_text]))[0]
            vec_scores = [cosine(q_vec, c.vector) for c in self.chunks]

        combined = self._combine(lex_scores, vec_scores, lexical_weight, vector_weight)
        ranked = sorted(
            ((s, c) for s, c in zip(combined, self.chunks) if s >= threshold),
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in ranked[:top_k]]

    def _combine(self, a: list[float], b: list[float], wa: float, wb: float) -> list[float]:
        na = self._minmax(a)
        nb = self._minmax(b)
        return [wa * x + wb * y for x, y in zip(na, nb)]

    @staticmethod
    def _minmax(vals: list[float]) -> list[float]:
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]
