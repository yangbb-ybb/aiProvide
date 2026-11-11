"""端到端编排：问题 → 召回 → 重排 → 组上下文 → 流式生成。

这个模块是 RAG 链路的大脑。`api/chat.py` 只负责会话落库和 SSE 封装，
真正「问题 → 答案」的活都在 `rag_answer_stream()` 里。

    ┌─ rag_answer_stream(question, client, retriever, top_k, history)
    │
    │ ① 查询改写：结合 history，把「那它为什么慢」补成一条能独立检索的查询
    │      失败 / 结果为空 / 超长 → 回退原问题
    │
    │ ② 召回 + 重排：一次 retriever.retrieve() 调用全包了
    │      内部：向量检索 ∥ BM25 → RRF 融合 → 邻居扩展 → cross-encoder 重排
    │      CPU 密集（三个模型都在本地跑），整个丢进 asyncio.to_thread，不堵事件循环
    │
    │ ③ 兜底重试：查询改写偶尔会过度扩展
    │      没命中 且 问题被改写过 → 用原问题再检索一次
    │
    │ ④ 没命中 → 不拒答，降级成通用回答
    │      换一套提示词，要求模型开口就说明「本地知识库未命中」
    │
    │ ⑤ 命中 → 先发 citation（引用），再发 delta（增量文本）
    │      组上下文：每块加 [n] 编号 + 来源标签，正文裁到 800 字
    │      拼上【最近对话】和【当前问题】，交给 Ollama 流式生成
    │
    │ ⑥ done：整轮结束
    │
    └─ 结束

事件类型（消费方：`api/chat.py` 和 `eval/run.py`）：

    delta    —— 增量文本，yield 一次推一段
    citation —— 引用列表，命中时**只发一次**
    done     —— 结束信号
    trace    —— 预留通道，检索调试面板要接实时召回详情时启用

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

# 没有命中本地资料时用的提示词 —— 降级成通用回答，而不是拒答。
# 关键约束是「开头先说明未命中」：不说明的话，用户会以为这是从资料里查出来的，
# 等于把模型的通用知识伪装成有出处的答案。
_GENERAL_SYSTEM = """你是一个有帮助的中文 AI 助手。
本地知识库没有找到与用户问题直接相关的资料，请使用你的通用知识回答。
要求：
1. 直接回答问题，不要编造“本地资料”或不存在的引用编号。
2. 如果问题需要实时信息、个人数据或你无法确认的事实，要明确说明限制。
3. 简洁、清楚地给出结论和必要解释。
4. 开头先说明“本地知识库未命中，以下为通用回答”，然后再回答。"""

# 查询改写用的提示词。放在这里而不是 prompt.py，是因为它不是「回答问题」的提示词，
# 而是「检索器的前置处理器」—— 输出要的是一条查询，不是一段答案。
# 第 3 条「加同义词、用空格分隔」是给 BM25 用的：jieba 按空格切词，
# 补上「部署 上线 发布」这类同义表达，关键词路才捞得到。
_QUERY_REWRITE_SYSTEM = """你是知识库检索查询改写器。请结合最近对话，把用户的最新问题改写成一条可以独立检索的中文查询。
要求：
1. 补全“它、这个、上述方案、那为什么”等指代，但不要编造对话里没有的事实。
2. 保留产品名、技术名、数字、时间和限制条件。
3. 同时加入常见同义表达，使用空格分隔关键词，例如“部署 上线 发布”。
4. 只输出一行改写后的查询，不要解释，不要回答问题。
5. 如果最新问题本身已经完整，原样或轻微扩展即可。"""


def _history_text(history: list[dict], max_messages: int = 6, max_chars: int = 3000) -> str:
    """压缩最近对话，避免查询改写占用过多上下文。

    两道限制：
    - 只取最近 max_messages 条，再多对「指代补全」没帮助，只吃 token
    - 再按 max_chars 从**尾部**截断。留最新的对话比留最早的重要 ——
      指代指向的是刚才说的东西，不是十轮前说的。

    返回格式是「用户：xxx」/「助手：xxx」，改写提示词要读得懂这个格式。
    """
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
    """把依赖上下文的追问改写为可独立检索的查询；失败时安全回退原问题。

    为什么需要改写：向量库和 BM25 都是「单条查询 → 召回」，
    检索不到「它」「上面那个」这种没有实体名词的追问。
    检索层不懂对话上下文，改写这一步就是把对话上下文**补回查询里**。

    三条降级路径，任何一条触发都回退原问题：
    1. 没有 history —— 没什么可补的，直接用原问题（省一次模型调用）
    2. 调用抛异常 —— 改写是锦上添花，不能让它拖垮整轮问答
    3. 结果为空或超过 500 字 —— 模型跑偏了（解释了一大段，或开始编造）
    """
    # ① 没有历史对话就不改写，省掉一次模型往返
    if not history:
        return question

    user = f"最近对话：\n{_history_text(history)}\n\n最新问题：\n{question}"
    pieces = []
    try:
        # temperature=0：改写是「抽取」不是「创作」，要的是可复现的改写，
        # 不是多样性。top_p=0.2 同理，num_ctx 压到 2048 也够用。
        async for piece in client.chat_stream(
            system=_QUERY_REWRITE_SYSTEM,
            user=user,
            temperature=0.0,
            top_p=0.2,
            num_ctx=2048,
        ):
            pieces.append(piece)
    except Exception:
        # 改写失败不是致命错误：拿原问题去检索，大不了召回差一点
        logger.exception("检索查询改写失败，回退原问题")
        return question

    # strip 掉模型可能加的引号（中英文都试），只留查询本身
    rewritten = "".join(pieces).strip().strip('"“”')
    # 空结果或超长都说明模型跑偏了，回退
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
    （trace 是预留通道，见模块 docstring）

    事件顺序：先 citation 再 delta。规划文档 §3.3 的示例把 citation 放在
    delta 之后，但前端是按事件类型渲染的，引用先到能让来源在生成过程中
    就显示出来，而不是等答案出完才出现。

    重排不在这里做 —— retriever 内部已经装了 reranker，retrieve() 返回的就是精排结果。

    top_k 是**最终进上下文的块数**（不是召回条数）：召回要宽（50 条候选池），
    进模型的要窄（5 块 × 800 字 ≈ 4000 token），两件事由检索层和这里分别控制。
    """
    conversation = history or []

    # ── ① 查询改写 ──────────────────────────────────────────────────────
    # 把追问补成能独立检索的查询，失败回退原问题（细节见 rewrite_retrieval_query）
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

    # ── ② 召回 + 重排 ───────────────────────────────────────────────────
    # 召回+融合+重排是 CPU 密集（向量检索、BM25、cross-encoder 都在本地跑），
    # 必须挪到线程里，不能堵住事件循环 —— 堵住了 SSE 就推不出去。
    # 重排不在这一层做：retriever 内部已经装了 reranker，
    # retrieve() 返回的就是精排后的结果。
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

    # ── ③ 兜底重试 ──────────────────────────────────────────────────────
    # 查询改写偶尔会过度扩展（把「它」补成了错误的实体）。
    # 没命中时再用原问题兜底一次，避免「改写失败」反而让原本能回答的问题失效。
    if not chunks and retrieval_query != question:
        chunks = await asyncio.to_thread(retriever.retrieve, question, top_k)

    # ── ④ 没命中：降级为通用回答 ─────────────────────────────────────────
    # 注意这里**不拒答**。没召回到资料 ≠ 问题无法回答，
    # 更可能只是资料没覆盖这个话题。硬拒答会让用户以为系统坏了。
    # 但也不能让模型装作在查资料 —— 所以换一套提示词，
    # 要求它开头就说「本地知识库未命中」，别把通用知识伪装成有出处的答案。
    # 区分：知识库**为空**（还没导入资料）走 chat.py 的直通对话；
    # 知识库**非空但没召回到**才走这里。两者不是一回事。
    if not chunks:
        logger.info("本地知识库未命中，降级为通用回答：%s", question)
        async for piece in client.chat_stream(system=_GENERAL_SYSTEM, user=question):
            yield ("delta", piece)
        yield ("done", {})
        return

    # ── ⑤ 命中：先发引用，再生成正文 ────────────────────────────────────
    # 引用排在正文之前：前端按事件类型渲染，来源先显示出来，
    # 用户在答案还在打字时就能看到「这是从哪来的」。
    yield ("citation", [to_citation(i, c) for i, c in enumerate(chunks, start=1)])

    # 组上下文：每块前面加 [n] 编号和「来源：文件 > 标题路径」标签，
    # 正文裁到 context_chunk_chars（默认 800）字。
    context = build_context(chunks, max_chars=context_chunk_chars)

    # 最近对话也拼进去，让模型能理解「它」「上面那个」这类指代。
    # 但 SYSTEM_PROMPT 第 2 条钉死了边界：最近对话只能用来理解指代，
    # 不能当事实来源 —— 事实只能来自【参考资料】。
    conversation_text = _history_text(conversation)
    history_section = f"【最近对话】\n{conversation_text}\n\n" if conversation_text else ""
    user = f"{history_section}【参考资料】\n{context}\n\n【当前问题】\n{question}"

    # ── ⑥ 流式生成 ──────────────────────────────────────────────────────
    # 每个 piece 一段增量文本，yield 一次推一次。
    # 这里 yield 的时机决定了体感速度：模型吐一段，用户就看到一段。
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

    # ── ⑦ 结束 ──────────────────────────────────────────────────────────
    yield ("done", {})


def to_citation(index: int, chunk: ScoredChunk) -> dict:
    """把一个块转成前端要的引用结构。

    index 就是正文里那个 [n] 的 n —— 从 1 开始，和 build_context 里的编号对齐。

    scores 里保留四路分数（rrf / dense / bm25 / rerank）：
    检索调试面板靠它回答「这块为什么排第一」——
    是稠密路捞到的还是关键词路捞到的，重排又把它提了多少名次。
    """
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
