"""端到端编排：问题 → 召回 → 重排 → 组上下文 → 流式生成。

M0 阶段 chat.py 直接调 Ollama；M1 起这条链路接管。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from ..llm.ollama import OllamaClient
from .prompt import SYSTEM_PROMPT, build_context
from .retriever import ScoredChunk

logger = logging.getLogger(__name__)

_NO_HIT_ANSWER = "知识库里没有检索到与这个问题相关的内容，暂时无法回答。请换个说法，或先把相关文档导入知识库。"
_QUERY_REWRITE_SYSTEM = """你是知识库检索查询改写器。请结合最近对话，把用户的最新问题改写成一条可以独立检索的中文查询。
要求：
1. 补全“它、这个、上述方案、那为什么”等指代，但不要编造对话里没有的事实。
2. 保留产品名、技术名、数字、时间和限制条件。
3. 同时加入常见同义表达，使用空格分隔关键词，例如“部署 上线 发布”。
4. 只输出一行改写后的查询，不要解释，不要回答问题。
5. 如果最新问题本身已经完整，原样或轻微扩展即可。"""


def _history_text(history: list[dict], max_messages: int = 6, max_chars: int = 3000) -> str:
    """压缩最近对话，避免查询改写占用过多上下文。"""
    lines = []
    for item in history[-max_messages:]:
        role = "用户" if item.get("role") == "user" else "助手"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}：{content}")
    text = "\n".join(lines)
    return text[-max_chars:]


async def rewrite_retrieval_query(
    question: str,
    history: list[dict],
    client: OllamaClient,
) -> str:
    """把依赖上下文的追问改写为可独立检索的查询；失败时安全回退原问题。"""
    if not history:
        return question
    user = f"最近对话：\n{_history_text(history)}\n\n最新问题：\n{question}"
    pieces = []
    try:
        async for piece in client.chat_stream(
            system=_QUERY_REWRITE_SYSTEM,
            user=user,
            temperature=0.0,
            top_p=0.2,
            num_ctx=2048,
        ):
            pieces.append(piece)
    except Exception:
        logger.exception("检索查询改写失败，回退原问题")
        return question
    rewritten = "".join(pieces).strip().strip('"“”')
    if not rewritten or len(rewritten) > 500:
        return question
    logger.info("检索查询改写：%s -> %s", question, rewritten)
    return rewritten


async def rag_answer_stream(
    question: str,
    client: OllamaClient,
    retriever,
    top_k: int = 5,
    context_chunk_chars: int = 800,
    history: list[dict] | None = None,
    trace_id: str | None = None,
) -> AsyncIterator[tuple[str, object]]:
    """流式产出 (event_type, payload)。

    event_type 取值：delta（增量文本）/ citation（引用列表）/ done

    事件顺序：先 citation 再 delta。规划文档 §3.3 的示例把 citation 放在
    delta 之后，但前端是按事件类型渲染的，引用先到能让来源在生成过程中
    就显示出来，而不是等答案出完才出现。

    重排不在这里做 —— retriever 内部已经装了 reranker，retrieve() 返回的就是精排结果。
    """
    conversation = history or []
    rewrite_started = time.perf_counter()
    retrieval_query = await rewrite_retrieval_query(question, conversation, client)
    logger.info(
        "chat 阶段完成",
        extra={
            "request_id": trace_id,
            "phase": "query_rewrite",
            "duration_ms": round((time.perf_counter() - rewrite_started) * 1000, 2),
            "query_chars": len(retrieval_query),
            "history_messages": len(conversation),
        },
    )

    # 召回 + 重排是 CPU 密集（向量检索、BM25、cross-encoder 都在本地跑），
    # 必须挪到线程里，不能堵住事件循环。
    retrieval_started = time.perf_counter()
    chunks: list[ScoredChunk] = await asyncio.to_thread(
        retriever.retrieve,
        retrieval_query,
        top_k,
    )
    logger.info(
        "chat 阶段完成",
        extra={
            "request_id": trace_id,
            "phase": "retrieve_and_rerank",
            "duration_ms": round((time.perf_counter() - retrieval_started) * 1000, 2),
            "candidate_count": len(chunks),
            "result_count": len(chunks),
        },
    )

    # 查询改写偶尔会过度扩展。没有命中时再用原问题兜底一次，
    # 避免“改写失败”反而让原本能回答的问题失效。
    if not chunks and retrieval_query != question:
        chunks = await asyncio.to_thread(retriever.retrieve, question, top_k)

    if not chunks:
        logger.info("召回为空：%s", question)
        yield ("delta", _NO_HIT_ANSWER)
        yield ("done", {})
        return

    yield ("citation", [to_citation(i, c) for i, c in enumerate(chunks, start=1)])

    context = build_context(chunks, max_chars=context_chunk_chars)
    conversation_text = _history_text(conversation)
    history_section = f"【最近对话】\n{conversation_text}\n\n" if conversation_text else ""
    user = f"{history_section}【参考资料】\n{context}\n\n【当前问题】\n{question}"

    generation_started = time.perf_counter()
    async for piece in client.chat_stream(system=SYSTEM_PROMPT, user=user):
        yield ("delta", piece)
    logger.info(
        "chat 阶段完成",
        extra={
            "request_id": trace_id,
            "phase": "generation",
            "duration_ms": round((time.perf_counter() - generation_started) * 1000, 2),
        },
    )

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
