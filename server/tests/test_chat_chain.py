"""多轮 RAG 查询改写和上下文组装测试。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chain import _history_text, rag_answer_stream, rewrite_retrieval_query
from app.rag.retriever import ScoredChunk


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []

    async def chat_stream(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        yield response


class FakeRetriever:
    def __init__(self, hits_by_query: dict[str, list[ScoredChunk]]):
        self.hits_by_query = hits_by_query
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int):
        self.queries.append(query)
        return self.hits_by_query.get(query, [])[:top_k]


def test_history_text_keeps_recent_messages():
    history = [{"role": "user", "content": f"问题{i}"} for i in range(10)]
    text = _history_text(history, max_messages=3)
    assert "问题6" not in text
    assert "问题7" in text
    assert "问题9" in text


def test_rewrite_query_uses_history():
    async def run():
        client = FakeClient(["项目为什么选择 LanceDB 作为向量数据库 向量存储 选型"])
        rewritten = await rewrite_retrieval_query(
            "为什么选择它？",
            [
                {"role": "user", "content": "项目使用什么向量数据库？"},
                {"role": "assistant", "content": "使用 LanceDB。"},
            ],
            client,
        )
        assert "LanceDB" in rewritten
        assert "最近对话" in client.calls[0]["user"]

    asyncio.run(run())


def test_rag_uses_rewritten_query_and_preserves_original_question():
    asyncio.run(_run_rag_uses_rewritten_query())


async def _run_rag_uses_rewritten_query():
    chunk = ScoredChunk(content="LanceDB 是嵌入式向量库，无需独立服务。", chunk_id="c1", source_path="README.md")
    rewritten = "项目为什么选择 LanceDB 作为向量数据库"
    retriever = FakeRetriever({rewritten: [chunk]})
    client = FakeClient([rewritten, "因为它无需独立服务。[1]"])

    events = [
        event
        async for event in rag_answer_stream(
            question="为什么选择它？",
            client=client,
            retriever=retriever,
            history=[{"role": "assistant", "content": "项目使用 LanceDB。"}],
        )
    ]

    assert retriever.queries == [rewritten]
    assert any(kind == "citation" for kind, _ in events)
    assert "【最近对话】" in client.calls[1]["user"]
    assert "【当前问题】\n为什么选择它？" in client.calls[1]["user"]


def test_rewrite_no_hit_falls_back_to_original_question():
    asyncio.run(_run_rewrite_no_hit_fallback())


async def _run_rewrite_no_hit_fallback():
    chunk = ScoredChunk(content="部署需要 Docker。", chunk_id="c1", source_path="deploy.md")
    retriever = FakeRetriever({"怎么部署？": [chunk]})
    client = FakeClient(["错误改写", "使用 Docker 部署。[1]"])

    events = [
        event
        async for event in rag_answer_stream(
            question="怎么部署？",
            client=client,
            retriever=retriever,
            history=[{"role": "user", "content": "换个问题"}],
        )
    ]

    assert retriever.queries == ["错误改写", "怎么部署？"]
    assert events[-1][0] == "done"
