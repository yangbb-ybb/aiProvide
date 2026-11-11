"""结构感知分块。

算法（规划文档 §4.2）：
1. 正则扫标题 `^(#{1,6})\\s+(.+)$`，维护 heading_path 栈
2. 每个标题下的正文 = 一个 section
3. section 内按空行切段落；代码块和表格整体不切
4. 累积到 target_chars 切一刀，切点优先落在句读边界
5. 相邻块重叠 overlap_chars，按句边界回退，不切断句子

参数依据：
- target=500：<300 语义不完整；>800 超出嵌入模型 512-token 训练窗口
- overlap=80：约 16%，够补断句又不至于让相邻块挤占 top-k 名额
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 标题行：`# xxx` 到 `###### xxx`，末尾允许有闭合的 #（Markdown 风格）
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# 页码标记，由 PDF loader 注入：`<!-- page: 12 -->`
_PAGE_RE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->$")

# 句末标点。中文句读 + 英文句号（英文句号要求后面跟空白，避免切坏 "3.2"）
_SENT_ENDS = "。！？；!?;：\n"
_HARD_LIMIT_RATIO = 1.5
"""单个单元超过 target 的这个倍数时，用硬切兜底（一般是超长无标点的代码块）。"""


@dataclass
class Chunk:
    """一个文本块。字段含义见规划文档 §3.3 的字段表。

    chunk_id / doc_id / title 等由 indexer 在入库时补全，
    分块阶段只负责正文和它自带的结构信息。
    """

    content: str
    heading_path: list[str] = field(default_factory=list)
    chunk_idx: int = 0
    char_len: int = 0
    page: int | None = None
    # ---- 以下由 indexer 补全 ----
    chunk_id: str = ""
    doc_id: str = ""
    source_path: str = ""
    source_type: str = ""
    title: str = ""
    content_hash: str = ""
    mtime: float = 0.0
    url: str | None = None


@dataclass
class _Unit:
    """分块过程中的最小单位。一段文字、一张表的一部分、或一个代码块。"""

    text: str
    page: int | None = None
    is_table: bool = False


def chunk_markdown(
    md: str,
    source_meta: dict,
    target_chars: int = 500,
    overlap_chars: int = 80,
) -> list[Chunk]:
    """把 Markdown 切成带 heading_path 的块列表。

    source_meta 目前用于承接未来可能的内容级过滤（如按目录裁剪），
    现阶段不参与切块逻辑，保留参数是为了调用方语义完整。
    """
    chunks: list[Chunk] = []
    for heading_path, body in _split_sections(md):
        units = _split_units(body, target_chars)
        for text, page in _pack(units, target_chars, overlap_chars):
            content = text.strip()
            if not content:
                continue
            chunks.append(
                Chunk(
                    content=content,
                    heading_path=list(heading_path),
                    chunk_idx=len(chunks),
                    char_len=len(content),
                    page=page,
                )
            )
    return chunks


# --------------------------------------------------------------------------
# 第一步：按标题切 section
# --------------------------------------------------------------------------


def _split_sections(md: str) -> list[tuple[list[str], list[str]]]:
    """把 Markdown 切成 [(heading_path, 行列表)]。

    代码块（``` 围栏）里的 # 不是标题，要跳过。
    """
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []
    headings: list[str] = []
    lines: list[str] = []
    in_fence = False
    fence = ""

    for line in md.splitlines():
        stripped = line.strip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence = stripped[:3]
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            if stripped.startswith(fence):
                in_fence = False
            continue

        m = _HEADING_RE.match(line)
        if m:
            if lines:
                sections.append((list(headings), lines))
                lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # 栈里 level >= 当前 level 的都出栈，剩下的就是父级标题
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            headings = [t for _, t in stack]
            continue

        lines.append(line)

    if lines:
        sections.append((list(headings), lines))
    return sections


# --------------------------------------------------------------------------
# 第二步：section 正文切成原子单元
# --------------------------------------------------------------------------


def _split_units(lines: list[str], target_chars: int) -> list[_Unit]:
    """把 section 正文切成不可再分的单元。

    - 段落（空行分隔）整体为一个单元
    - 代码块（``` 围栏）整体为一个单元
    - 表格按行切，但每段都重复表头，方便脱离上下文也能读懂
    """
    units: list[_Unit] = []
    page: int | None = None
    buf: list[str] = []
    in_fence = False
    fence = ""

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        if _is_table(text):
            for part in _split_table(text, target_chars):
                units.append(_Unit(part, page=page, is_table=True))
        elif len(text) > target_chars:
            # 超长段落（没有空行的一整段）→ 按句读再切
            for part in _split_long_text(text, target_chars):
                units.append(_Unit(part, page=page))
        else:
            units.append(_Unit(text, page=page))

    for line in lines:
        stripped = line.strip()

        pm = _PAGE_RE.match(stripped)
        if pm and not in_fence:
            flush()
            page = int(pm.group(1))
            continue

        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            flush()
            in_fence = True
            fence = stripped[:3]
            buf.append(line)
            continue

        if in_fence:
            buf.append(line)
            if stripped.startswith(fence):
                in_fence = False
                flush()
            continue

        if not stripped:
            flush()
            continue

        buf.append(line)

    flush()
    return units


def _is_table(text: str) -> bool:
    """判断一段文本是不是 Markdown 表格（所有非空行都以 | 开头）。"""
    rows = [ln for ln in text.splitlines() if ln.strip()]
    if len(rows) < 2:
        return False
    return all(ln.strip().startswith("|") for ln in rows)


def _split_table(text: str, target_chars: int) -> list[str]:
    """把大表按行切开，每片都带上表头 + 分隔行。"""
    rows = [ln for ln in text.splitlines() if ln.strip()]
    if len(rows) < 3:
        return [text]
    header = rows[:2]
    body = rows[2:]
    out: list[str] = []
    cur: list[str] = []
    cur_len = len("\n".join(header))
    for row in body:
        if cur and cur_len + len(row) + 1 > target_chars:
            out.append("\n".join(header + cur))
            cur = []
            cur_len = len("\n".join(header))
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        out.append("\n".join(header + cur))
    return out


# --------------------------------------------------------------------------
# 第三步：句读切分与重叠
# --------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句，标点保留在句尾。"""
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _SENT_ENDS:
            j = i + 1
            # 吞掉紧跟的收尾引号 / 括号，避免标点被劈开
            while j < n and text[j] in "」』”’\"')）】》":
                j += 1
            sentences.append(text[start:j])
            start = j
            i = j
            continue
        if ch == "." and (i + 1 >= n or text[i + 1].isspace()):
            # 英文句号：后面必须是空白或结尾，否则是小数点 / 缩写
            sentences.append(text[start : i + 1])
            start = i + 1
        i += 1
    if start < n:
        sentences.append(text[start:])
    return [s for s in sentences if s.strip()]


def _split_long_text(text: str, target_chars: int) -> list[str]:
    """把超长文本按句读切成若干片，片与片之间带重叠。"""
    out: list[str] = []
    cur = ""
    for sentence in _split_sentences(text):
        if not cur:
            cur = sentence
        elif len(cur) + len(sentence) <= target_chars:
            cur += sentence
        else:
            out.append(cur)
            cur = sentence
    if cur:
        out.append(cur)
    # 兜底：仍有超过硬上限的片（无标点的长代码行），按字符硬切
    hard_limit = int(target_chars * _HARD_LIMIT_RATIO)
    result: list[str] = []
    for piece in out:
        while len(piece) > hard_limit:
            result.append(piece[:hard_limit])
            piece = piece[hard_limit:]
        if piece:
            result.append(piece)
    return result


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """取上一块尾部作为下一块的开头，尽量从句子开头切入，不切断句子。"""
    if overlap_chars <= 0 or not text:
        return ""
    tail = text[-overlap_chars:]
    for i, ch in enumerate(tail):
        if ch in _SENT_ENDS:
            return tail[i + 1 :].strip()
    return tail.strip()


# --------------------------------------------------------------------------
# 第四步：把单元打包成块
# --------------------------------------------------------------------------


def _pack(units: list[_Unit], target_chars: int, overlap_chars: int) -> list[tuple[str, int | None]]:
    """把单元累积成块，尽量接近 target_chars。返回 [(正文, 页码)]。

    三条硬规则：
    - 表格片之间不合并（避免重复表头糊在一起），表格片也不带重叠
    - 换页即断开（PDF 页是天然边界，跨页合并会让页码归属失真）
    - 其余单元按 target_chars 累积，块间保留 overlap_chars 的重叠
    """
    chunks: list[tuple[str, int | None]] = []
    cur = ""
    cur_page: int | None = None
    last_is_table = False

    for unit in units:
        text = unit.text.strip()
        if not text:
            continue

        page_changed = (
            cur
            and unit.page is not None
            and cur_page is not None
            and unit.page != cur_page
        )
        # 连续两张表 → 先把当前块收掉，别把两张表粘一起
        if page_changed or (cur and unit.is_table and last_is_table):
            chunks.append((cur, cur_page))
            cur = ""
            cur_page = None
            last_is_table = False

        if not cur:
            cur = text
            cur_page = unit.page
        elif len(cur) + len(text) + 2 <= target_chars:
            cur = cur + "\n\n" + text
        else:
            chunks.append((cur, cur_page))
            cur_page = unit.page
            if unit.is_table:
                cur = text
            else:
                tail = _overlap_tail(cur, overlap_chars)
                cur = f"{tail}\n\n{text}" if tail else text
        last_is_table = unit.is_table

    if cur:
        chunks.append((cur, cur_page))
    return chunks
