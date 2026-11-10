"""提示词。

全文与设计说明见规划文档 §4.6。要点：
- 强制只依据【参考资料】作答，不要用先验知识补充
- 强制标注来源编号 [n]，否则答案无法溯源
- 资料不足时明说「无法回答」，不要编造
- temperature 0.1：提取任务不是创作任务
"""

from __future__ import annotations

from .retriever import ScoredChunk

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。你只能依据【参考资料】中的内容回答问题。

规则：
1. 只使用【参考资料】中出现的信息，不要用自己的先验知识补充或推断。
2. 每条结论后面必须标注来源编号，格式如 [1] 或 [1][3]，编号对应参考资料前的 [n]。
3. 如果参考资料不足以回答，直接说"现有资料无法回答这个问题"，并说明还缺少什么信息。不要猜测、不要编造。
4. 用简体中文回答，简洁直接：先给结论，再给必要的展开。
5. 参考资料之间出现矛盾信息时，把矛盾点指出来，不要擅自选一个。
6. 不要复述问题，不要写"好的""根据参考资料"这类开场白。"""

USER_TEMPLATE = """【参考资料】
{context}

【问题】
{question}"""


def build_context(chunks: list[ScoredChunk], max_chars: int = 800) -> str:
    """把块拼成带 [n] 编号和来源标签的上下文。

    每块正文裁到 max_chars 字（规划文档 §4.5）：
    再长信息密度下降，而且会挤占上下文预算（5 块 × 800 字 ≈ 4000 token）。
    裁剪点优先落在句读边界，避免把句子截断成半句。
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = c.source_path
        if c.heading_path:
            src += " > " + " > ".join(c.heading_path)
        if c.page is not None:
            src += f" 第{c.page}页"
        parts.append(f"[{i}] (来源：{src})\n{_truncate(c.content, max_chars)}")
    return "\n\n".join(parts)


def _truncate(text: str, max_chars: int) -> str:
    """裁到 max_chars 字，尽量落在句读边界上。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for sep in ("。", "！", "？", "；", "\n", "。", "!", "?", ";"):
        idx = window.rfind(sep)
        if idx >= max_chars * 0.5:
            return window[: idx + 1]
    return window + "…"


def build_messages(question: str, chunks: list[ScoredChunk], max_chars: int = 800) -> list[dict]:
    """组装成 Ollama /api/chat 的 messages 数组。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=build_context(chunks, max_chars=max_chars), question=question
            ),
        },
    ]
