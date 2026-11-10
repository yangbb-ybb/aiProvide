"""端到端编排：问题 → 召回 → 重排 → 组上下文 → 流式生成。

M0 阶段 chat.py 直接调 Ollama；M1 起这条链路接管。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..llm.ollama import OllamaClient
from .retriever import ScoredChunk


async def rag_answer_stream(
    question: str,
    client: OllamaClient,
    retriever,
    reranker,
    top_k: int = 5,
) -> AsyncIterator[tuple[str, object]]:
    """流式产出 (event_type, payload)。

    event_type 取值：delta（增量文本）/ citation（引用列表）/ done
    """
    raise NotImplementedError("M1 阶段实现：RAG 问答链路")
