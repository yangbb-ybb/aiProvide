"""文档解析：各格式统一转成 Markdown。

设计要求：所有格式转出的 Markdown **必须保留标题层级**，
这是 chunker 结构感知分块的前提。

实现顺序建议：md/txt → pdf → docx → xlsx → pptx → url。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParsedDoc:
    """解析结果。"""

    title: str
    markdown: str
    source_path: str
    source_type: str
    meta: dict = field(default_factory=dict)
    needs_ocr: bool = False
    """PDF 某页文本过少（<50 字符）时置 True。不静默产出空块。"""


def load(source: str, source_type: str) -> ParsedDoc:
    """统一入口。source 是本地路径或 URL，source_type 见 schemas 里的约定。

    待实现（按类型分发到下面各 load_* 函数）。
    """
    raise NotImplementedError(f"M1 阶段实现：解析 {source_type}")


def load_markdown(path: str) -> ParsedDoc:
    """md：直读，保留原样。"""
    raise NotImplementedError


def load_text(path: str) -> ParsedDoc:
    """txt：直读，按空行分段。"""
    raise NotImplementedError


def load_pdf(path: str) -> ParsedDoc:
    """pdf：PyMuPDF 逐页取文本，记录页码映射。

    某页文本 <50 字符 → needs_ocr=True。
    """
    raise NotImplementedError


def load_docx(path: str) -> ParsedDoc:
    """docx：按 style.name 判断 Heading 级别转成 #，表格转 md 表格。"""
    raise NotImplementedError


def load_xlsx(path: str) -> ParsedDoc:
    """xlsx：每个 sheet 转成一个 md 表格 + 表名作为 heading，空 sheet 跳过。"""
    raise NotImplementedError


def load_pptx(path: str) -> ParsedDoc:
    """pptx：每页转 `## 第N页 {标题}` + 各文本框内容，跳过占位符。"""
    raise NotImplementedError


def load_url(url: str) -> ParsedDoc:
    """url：httpx + trafilatura 抽正文去广告，失败降级 BeautifulSoup 去标签。"""
    raise NotImplementedError
