"""提示词。

全文与设计说明见规划文档 §4.6。要点：
- 强制只依据【参考资料】作答，不要用先验知识补充
- 强制标注来源编号 [n]，否则答案无法溯源
- 资料不足时明说「无法回答」，不要编造
- temperature 0.1：提取任务不是创作任务

第 5 条规则用反斜杠续行拆成两行（源码单行超 120 字符的 lint 上限），
反斜杠在字符串里是续行符、不进字符串内容，所以送进模型的提示词逐字节不变。
"""

from __future__ import annotations

from .retriever import ScoredChunk

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。你只能依据【参考资料】中的内容回答问题。

规则：
1. 事实、结论和数据只能来自【参考资料】，不要用自己的先验知识补充或编造。
2. 可以使用【最近对话】理解“它、这个、上面那个、为什么”等指代，但最近对话不能替代参考资料成为事实来源。
3. 用户不必与资料原文使用一模一样的措辞；应识别同义表达、口语说法和合理的语义对应，再基于资料回答。
4. 每条结论后面必须标注来源编号，格式如 [1] 或 [1][3]，编号对应参考资料前的 [n]。
5. 如果资料能回答核心问题但缺少次要细节，先回答能确认的部分，再说明哪些细节缺失；\
不要因为措辞不同或局部信息不足就整体拒答。
6. 只有参考资料确实不包含问题所需的核心事实时，才说“现有资料无法回答这个问题”，并说明还缺少什么信息。
7. 用简体中文回答，先给结论，再给必要的展开。参考资料矛盾时明确指出矛盾。
8. 不要复述问题，不要写“好的”“根据参考资料”这类开场白。"""

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
