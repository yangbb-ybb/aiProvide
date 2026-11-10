"""结构感知分块。

算法（规划文档 §4.2）：
1. 正则扫标题 `^(#{1,6})\\s+(.+)$`，维护 heading_path 栈
2. 每个标题下的正文 = 一个 section
3. section 内按空行切段落；代码块和表格整体不切
4. 累积到 target_chars 切一刀，切点优先落在句读边界
5. 相邻块重叠 overlap_chars，按句边界回退，不切断句子

参数依据：
- target=500：<300 语义不完整；>800 超出 bge-m3 的 512-token 训练窗口
- overlap=80：约 16%，够补断句又不至于让相邻块挤占 top-k 名额
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """一个文本块。字段含义见规划文档 §3.3 的字段表。"""

    content: str
    heading_path: list[str] = field(default_factory=list)
    chunk_idx: int = 0
    char_len: int = 0
    page: int | None = None


def chunk_markdown(
    md: str,
    source_meta: dict,
    target_chars: int = 500,
    overlap_chars: int = 80,
) -> list[Chunk]:
    """把 Markdown 切成带 heading_path 的块列表。"""
    raise NotImplementedError("M1 阶段实现：结构感知分块")
