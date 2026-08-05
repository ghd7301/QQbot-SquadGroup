import asyncio
import os

from qqbot.embedding import HashedNgramEmbedding
from qqbot.knowledge import KnowledgeBase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_and_retrieve():
    async def run():
        kb = KnowledgeBase(HashedNgramEmbedding())
        n = kb.load(os.path.join(ROOT, "knowledge"))
        assert n > 0
        await kb.embed_all()
        # threshold 0.0 so we always get ranked results to inspect
        chunks = await kb.retrieve("医疗兵 玩法", "医疗兵要咋玩", top_k=5, threshold=0.0)
        assert chunks, "should retrieve something"
        assert any("医疗兵" in c.text for c in chunks), "top results should mention 医疗兵"

    asyncio.run(run())


def test_retrieve_empty_on_no_knowledge():
    async def run():
        kb = KnowledgeBase(HashedNgramEmbedding())
        await kb.embed_all()
        chunks = await kb.retrieve("foo", "", top_k=5, threshold=0.0)
        assert chunks == []

    asyncio.run(run())
