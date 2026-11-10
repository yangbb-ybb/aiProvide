"""端到端编排：问题 → 召回 → 重排 → 组上下文 → 流式生成。

M0 阶段 chat.py 直接调 Ollama；M1 起这条链路接管。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from ..llm.ollama import OllamaClient
from .prompt import SYSTEM_PROMPT, build_context
from .retriever import ScoredChunk

logger = logging.getLogger(__name__)

_NO_HIT_ANSWER = "知识库里没有检索到与这个问题相关的内容，暂时无法回答。请换个说法，或先把相关文档导入知识库。"


async def rag_answer_stream(
    question: str,
    client: OllamaClient,
    retriever,
    top_k: int = 5,
    context_chunk_chars: int = 800,
) -> AsyncIterator[tuple[str, object]]:
    """流式产出 (event_type, payload)。

    event_type 取值：delta（增量文本）/ citation（引用列表）/ done

    事件顺序：先 citation 再 delta。规划文档 §3.3 的示例把 citation 放在
    delta 之后，但前端是按事件类型渲染的，引用先到能让来源在生成过程中
    就显示出来，而不是等答案出完才出现。

    重排不在这里做 —— retriever 内部已经装了 reranker，retrieve() 返回的就是精排结果。
    """
    # 召回 + 重排是 CPU 密集（向量检索、BM25、cross-encoder 都在本地跑），
    # 必须挪到线程里，不能堵住事件循环。
    chunks: list[ScoredChunk] = await asyncio.to_thread(retriever.retrieve, question, top_k)

    if not chunks:
        logger.info("召回为空：%s", question)
        yield ("delta", _NO_HIT_ANSWER)
        yield ("done", {})
        return

    yield ("citation", [to_citation(i, c) for i, c in enumerate(chunks, start=1)])

    context = build_context(chunks, max_chars=context_chunk_chars)
    user = f"【参考资料】\n{context}\n\n【问题】\n{question}"

    async for piece in client.chat_stream(system=SYSTEM_PROMPT, user=user):
        yield ("delta", piece)

    yield ("done", {})


def to_citation(index: int, chunk: ScoredChunk) -> dict:
    """把一个块转成前端要的引用结构。"""
    return {
        "index": index,
        "source_path": chunk.source_path,
        "heading_path": list(chunk.heading_path),
        "page": chunk.page,
        "score": round(float(chunk.score), 4),
        "preview": chunk.content[:200],
        "title": chunk.title,
        "chunk_id": chunk.chunk_id,
        "scores": {k: round(float(v), 4) for k, v in chunk.scores.items()},
    }
