"""POST /api/chat —— SSE 流式问答。

这个接口收到请求后到底干了什么，按时间顺序拆成 8 步：

    ┌─ 收到 POST /api/chat {question, history?, session_id?, top_k?}
    │
    │ ① 依赖注入：拿到 4 个进程级单例
    │      client（Ollama 客户端）/ retriever（召回器）
    │      vector_store（向量库）/ conversation_store（会话库）
    │      —— 模型只加载一次，不能每个请求重来
    │
    │ ② 取配置，决定 top_k（送进上下文几块）
    │
    │ ③ 拿 request_id：中间件塞的，串起这一次请求的所有日志
    │
    │ ④ ensure_session：没有 session_id 就生成一个，写进 sessions 表
    │
    │ ⑤ 先把「用户这轮问了什么」落库 —— 在生成之前
    │
    │ ⑥ 返回 StreamingResponse，真正的工作在 gen() 里随流推进
    │
    │ ⑦ gen()：查知识库是否为空
    │      空   → 直通对话（不检索，模型自由回答）
    │      非空 → rag_answer_stream()：
    │               查询改写（结合 history）
    │               召回 → RRF 融合 → 重排（CPU 密集，放线程里跑）
    │               命中 → 先发 citation，再发 delta 增量文本
    │               没命中 → 用原问题再试一次，还不行就说「无法回答」
    │
    │ ⑧ finally：把「助手答了什么」落库（带引用、状态、错误信息）
    │
    └─ 结束
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..config import get_settings
from ..deps import get_conversation_store, get_ollama, get_retriever, get_vector_store
from ..llm.ollama import OllamaClient, OllamaError
from ..rag.chain import rag_answer_stream
from ..schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# 知识库是空的时候用这条提示词退化成普通对话。
# 注意：知识库非空但没召回到相关内容时**不能**用这条 —— 那时要老实说「无法回答」。
# 两者要分清：「还没导入资料」和「资料里没有」不是一回事。
_PLAIN_SYSTEM = "你是一个乐于助人的中文 AI 助手。回答简洁、直接。"


def _sse(payload: dict) -> str:
    """把一个事件序列化成 SSE 帧。

    SSE 帧的格式是 `data: {json}\\n\\n`，末尾的空行是帧分隔符，
    前端按空行切帧（见 web/src/api/client.ts 的 chatStream）。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    client: OllamaClient = Depends(get_ollama),
    retriever=Depends(get_retriever),
    vector_store=Depends(get_vector_store),
    conversation_store=Depends(get_conversation_store),
) -> StreamingResponse:
    """流式问答。SSE 事件类型：delta（增量文本）/ citation（引用）/ trace（检索日志）/ done / error。

    知识库里有文档时：先检索 → 组装带引用编号的上下文 → 生成 → 推送 citation。
    知识库还是空的：退化成普通对话，让 M0 阶段也能正常聊天。
    """
    # ── ② 配置 ──────────────────────────────────────────────────────────
    settings = get_settings()
    # top_k = 最终送进 LLM 上下文的块数。没传就用 RERANK_TOP_K（默认 5）。
    top_k = req.top_k or settings.rerank_top_k

    # ── ③ 请求标识 ──────────────────────────────────────────────────────
    # request_logging_middleware（app/main.py）会给每个请求塞一个 request_id，
    # 优先用它，这样前端传上来的 X-Request-ID 和后端日志能对上；
    # 万一没有（比如直接调函数测试），就自己生成一个。
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    # 计时起点。中间件那边记的 duration_ms 只到「响应头发出」为止
    # （SSE 的正文是响应返回之后才慢慢推的），所以这个接口的真实耗时
    # 要看下面几个阶段日志，不能看中间件的接口耗时。
    request_started = time.perf_counter()

    # ── ④ 会话 ──────────────────────────────────────────────────────────
    # 没传 session_id 就新建一个；传了就沿用（并刷新 updated_at）。
    # ensure_session 内部写 sqlite，是同步阻塞调用，必须挪到线程里，别堵事件循环。
    session_started = time.perf_counter()
    session_id = await asyncio.to_thread(
        conversation_store.ensure_session,
        req.session_id,
        req.question,
    )
    session_ms = round((time.perf_counter() - session_started) * 1000, 2)
    turn_id = uuid4().hex
    # 写回 request.state，后面的中间件（响应日志）也能拿到 session_id
    request.state.session_id = session_id
    logger.info(
        "chat 阶段完成",
        extra={
            "request_id": request_id,
            "session_id": session_id,
            "phase": "ensure_session",
            "duration_ms": session_ms,
        },
    )

    # ── ⑤ 先落用户消息 ──────────────────────────────────────────────────
    # 为什么在生成之前就写：万一生成中途失败/断开，
    # 历史里至少还留着「用户问了什么」，重试时上下文不丢。
    await asyncio.to_thread(
        conversation_store.add_message,
        session_id,
        turn_id,
        "user",
        req.question,
    )

    # ── ⑥ 流式生成体 ────────────────────────────────────────────────────
    # 下面这个 async 生成器是响应体。StreamingResponse 一返回，HTTP 响应就开始了，
    # 后续每 yield 一段就往线上推一帧 —— 所以耗时的活儿必须放在 yield 之间，
    # 不能先做完再返回，否则前端就等不到「首字」。
    async def gen():
        # 累积本轮的答案和引用，finally 里要落库
        answer_parts: list[str] = []
        citations: list[dict] = []
        error_message: str | None = None
        first_token_started: float | None = None
        try:
            # ── ⑦a 知识库是不是空的 ─────────────────────────────────────
            # count() 要查向量库，也是同步阻塞调用。
            phase_started = time.perf_counter()
            kb_empty = await asyncio.to_thread(vector_store.count) == 0
            logger.info(
                "chat 阶段完成",
                extra={
                    "request_id": request_id,
                    "session_id": session_id,
                    "phase": "kb_count",
                    "duration_ms": round((time.perf_counter() - phase_started) * 1000, 2),
                },
            )

            if kb_empty:
                # ── ⑦b 空库：直通对话 ────────────────────────────────────
                # 不检索、不带引用，模型拿用户问题直接答。
                # 这是「用户还没导入任何资料」的状态，不是「资料不足」。
                logger.info("知识库为空，走直通对话")
                generation_started = time.perf_counter()
                async for piece in client.chat_stream(system=_PLAIN_SYSTEM, user=req.question):
                    if first_token_started is None:
                        first_token_started = time.perf_counter()
                        logger.info(
                            "chat 首 token 到达",
                            extra={
                                "request_id": request_id,
                                "session_id": session_id,
                                "phase": "generation_first_token",
                                "first_token_ms": round(
                                    (first_token_started - request_started) * 1000,
                                    2,
                                ),
                            },
                        )
                    answer_parts.append(piece)
                    yield _sse({"type": "delta", "text": piece})
                logger.info(
                    "chat 阶段完成",
                    extra={
                        "request_id": request_id,
                        "session_id": session_id,
                        "phase": "generation",
                        "duration_ms": round((time.perf_counter() - generation_started) * 1000, 2),
                    },
                )
                yield _sse({"type": "done", "session_id": session_id})
                return

            # ── ⑦c 有资料：走完整 RAG 链路 ─────────────────────────────
            # rag_answer_stream 内部依次做：
            #   查询改写（结合 history，把「它」「上面那个」补全成能独立检索的问句）
            #   → 召回 → RRF 融合 → 邻居扩展 → 重排（CPU 密集，它自己会挪到线程）
            #   → 组装带 [n] 编号的上下文 → 交给 Ollama 流式生成
            async for event_type, payload in rag_answer_stream(
                question=req.question,
                client=client,
                retriever=retriever,
                top_k=top_k,
                context_chunk_chars=settings.context_chunk_chars,
                history=[item.model_dump() for item in req.history],
                trace_id=request_id,
            ):
                if event_type == "delta":
                    # 首 token 的到达时间 = 用户从发出请求到看到第一个字等了多久，
                    # 是体感最直接的指标，单独记一次。
                    if first_token_started is None:
                        first_token_started = time.perf_counter()
                        logger.info(
                            "chat 首 token 到达",
                            extra={
                                "request_id": request_id,
                                "session_id": session_id,
                                "phase": "generation_first_token",
                                "first_token_ms": round(
                                    (first_token_started - request_started) * 1000,
                                    2,
                                ),
                            },
                        )
                    answer_parts.append(str(payload))
                    yield _sse({"type": "delta", "text": payload})
                elif event_type == "trace":
                    # 预留通道：检索阶段的详细信息（召回了什么、分数多少）。
                    # rag/chain.py 目前还没产出 trace 事件，检索细节由
                    # rag/retriever.py 自己直接打日志；等检索调试面板要接
                    # 实时召回详情时，从这里转出去。
                    logger.info(
                        "chat 检索阶段完成：%s",
                        payload,
                        extra={
                            "request_id": request_id,
                            "session_id": session_id,
                            "phase": "retrieval",
                        },
                    )
                elif event_type == "citation":
                    # 引用先于正文到达：前端拿到就能先把来源显示出来，
                    # 不用等答案生成完
                    citations = list(payload)
                    yield _sse({"type": "citation", "citations": payload})
                elif event_type == "done":
                    yield _sse({"type": "done", "session_id": session_id})
        except OllamaError as exc:
            # Ollama 侧的明确错误（连不上、模型没拉），原样透给前端
            error_message = str(exc)
            yield _sse({"type": "error", "message": error_message, "session_id": session_id})
        except Exception:  # 兜底，别让连接挂着
            # 其它异常不把细节往外抛（可能含路径等内部信息），细节进日志
            # logger.exception 会把完整堆栈写进 data/logs/app.jsonl
            logger.exception("问答链路异常")
            error_message = "处理失败，请查看服务日志"
            yield _sse({"type": "error", "message": error_message, "session_id": session_id})
        finally:
            # ── ⑧ 落助手消息 ───────────────────────────────────────────
            # 无论成功、报错还是客户端提前断开，都要把这一轮记下来，
            # 带上引用和状态（complete / error），历史才有连续性。
            await asyncio.to_thread(
                conversation_store.add_message,
                session_id,
                turn_id,
                "assistant",
                "".join(answer_parts),
                citations=citations,
                status="error" if error_message else "complete",
                error=error_message,
            )

    # ── 返回响应 ─────────────────────────────────────────────────────────
    # X-Accel-Buffering: no —— 告诉 nginx 别缓冲，否则流式会被攒成一坨最后一次性吐出来。
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
