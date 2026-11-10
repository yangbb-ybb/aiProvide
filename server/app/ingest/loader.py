"""文档解析：各格式统一转成 Markdown。

设计要求：所有格式转出的 Markdown **必须保留标题层级**，
这是 chunker 结构感知分块的前提。

实现顺序建议：md/txt → pdf → docx → xlsx → pptx → url。
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit

# PDF 每页文本少于这个字符数算「文本层稀薄」
_LOW_TEXT_CHARS = 50

# xlsx 读取上限，避免一个超大 sheet 把内存吃光
_XLSX_MAX_ROWS = 10000
_XLSX_MAX_COLS = 50


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
    url: str | None = None
    """网页来源的原始 URL（其它类型为 None）。"""


def load(source: str, source_type: str) -> ParsedDoc:
    """统一入口。source 是本地路径或 URL，source_type 见 schemas 里的约定。"""
    if source_type == "md":
        return load_markdown(source)
    if source_type == "txt":
        return load_text(source)
    if source_type == "pdf":
        return load_pdf(source)
    if source_type == "docx":
        return load_docx(source)
    if source_type == "xlsx":
        return load_xlsx(source)
    if source_type == "pptx":
        return load_pptx(source)
    if source_type == "url":
        return load_url(source)
    raise ValueError(
        f"不支持的文档类型：{source_type}（支持 md / txt / pdf / docx / xlsx / pptx / url）"
    )


def _read_text_file(path: str) -> str:
    """读文本文件，统一按 UTF-8，失败时退回 GB18030（Windows 导出的中文 txt 常见）。"""
    data = Path(path).read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件（既不是 UTF-8 也不是 GB18030）：{path}")


def _title_from_markdown(md: str, fallback: str) -> str:
    """取第一个一级标题，没有就用文件名。"""
    for line in md.splitlines():
        m = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if m:
            return m.group(1).strip()
    return fallback


def load_markdown(path: str) -> ParsedDoc:
    """md：直读，保留原样。"""
    md = _read_text_file(path)
    name = Path(path).stem
    return ParsedDoc(
        title=_title_from_markdown(md, name),
        markdown=md,
        source_path=str(path),
        source_type="md",
        meta={"chars": len(md)},
    )


def load_text(path: str) -> ParsedDoc:
    """txt：直读，按空行分段。

    txt 没有标题层级，只能整体当成一个 section。
    这里把首行单独拎出来当标题 —— 多数个人笔记第一行就是标题。
    """
    raw = _read_text_file(path)
    lines = raw.splitlines()
    title = Path(path).stem
    body_lines = lines
    if lines and lines[0].strip() and len(lines[0].strip()) <= 60:
        title = lines[0].strip()
        body_lines = lines[1:]

    # 按空行分段，段与段之间留一个空行（chunker 靠空行识别段落）
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(body_lines)) if p.strip()]
    md = f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n"
    return ParsedDoc(
        title=title,
        markdown=md,
        source_path=str(path),
        source_type="txt",
        meta={"chars": len(raw), "paragraphs": len(paragraphs)},
    )


def load_pdf(path: str) -> ParsedDoc:
    """pdf：PyMuPDF 逐页取文本，记录页码映射。

    某页文本 <50 字符 → needs_ocr=True。

    关于 needs_ocr 的判定：文档里写的是「某页 <50 字符就标记」，
    但实测下来 PDF 封面、目录页、章节扉页都容易踩这条线，
    按字面实现会让几乎每份 PDF 都被标成 needs_ocr，标记就失去意义了。
    现在的口径是：**超过一半的页面**文本稀薄，或者整份 PDF 提不出文字，才判定 needs_ocr；
    具体哪几页稀薄记录在 meta["low_text_pages"] 里，页数少的明细自己看。
    """
    # PyMuPDF 1.24.3+ 的正式模块名是 pymupdf；fitz 是历史别名，且容易和
    # 别的同名包撞车，所以优先用 pymupdf。
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover - 老版本兜底
        import fitz

    doc = fitz.open(path)
    parts: list[str] = []
    low_text_pages: list[int] = []
    total_chars = 0
    try:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            stripped = text.strip()
            total_chars += len(stripped)
            if len(stripped) < _LOW_TEXT_CHARS:
                low_text_pages.append(page_no)
            # 页码标记：chunker 靠它切块并给块打页码
            parts.append(f"<!-- page: {page_no} -->\n\n{stripped}")
    finally:
        doc.close()

    md = "\n\n".join(parts)
    title = Path(path).stem
    # 用第一页里最长的一行当标题（有 PDF 内部标题信息的除外）
    if parts:
        first_page = parts[0].split("<!-- page:", 1)[-1]
        candidates = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
        if candidates:
            title = max(candidates, key=len)[:80]

    page_count = len(parts)
    needs_ocr = bool(low_text_pages) and (
        len(low_text_pages) * 2 > page_count or total_chars == 0
    )

    return ParsedDoc(
        title=title,
        markdown=md,
        source_path=str(path),
        source_type="pdf",
        meta={
            "pages": page_count,
            "chars": total_chars,
            "low_text_pages": low_text_pages,
        },
        needs_ocr=needs_ocr,
    )


def _docx_heading_level(style_name: str) -> int | None:
    """从样式名判断标题级别，认得中英文两种写法。"""
    name = (style_name or "").strip()
    if not name:
        return None
    if name in ("Title", "标题"):
        return 1
    m = re.match(r"^(?:Heading|标题)\s*(\d+)$", name, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), 6)
    return None


def load_docx(path: str) -> ParsedDoc:
    """docx：按 style.name 判断 Heading 级别转成 #，表格转 md 表格。"""
    import docx  # python-docx

    document = docx.Document(path)
    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        level = _docx_heading_level(para.style.name if para.style else "")
        if level:
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)

    table_count = 0
    for table in document.tables:
        md_table = _rows_to_md_table([[c.text.strip() for c in row.cells] for row in table.rows])
        if md_table:
            lines.append(md_table)
            table_count += 1

    md = "\n\n".join(lines)
    title = (document.core_properties.title or "").strip() or Path(path).stem
    # 没有一级标题时补一个，保证 heading_path 至少有一层
    if not any(ln.startswith("# ") for ln in lines):
        md = f"# {title}\n\n{md}"

    return ParsedDoc(
        title=title,
        markdown=md,
        source_path=str(path),
        source_type="docx",
        meta={"paragraphs": len(document.paragraphs), "tables": table_count},
    )


def load_xlsx(path: str) -> ParsedDoc:
    """xlsx：每个 sheet 转成一个 md 表格 + 表名作为 heading，空 sheet 跳过。"""
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts: list[str] = []
    sheet_count = 0
    truncated = False
    try:
        for sheet in workbook.worksheets:
            rows: list[list[str]] = []
            for row in sheet.iter_rows(
                max_row=_XLSX_MAX_ROWS, max_col=_XLSX_MAX_COLS, values_only=True
            ):
                cells = ["" if v is None else str(v).strip() for v in row]
                if any(cells):
                    rows.append(cells)
                if len(rows) >= _XLSX_MAX_ROWS:
                    truncated = True
                    break
            if not rows:
                continue  # 空 sheet 跳过
            sheet_count += 1
            md_table = _rows_to_md_table(rows)
            parts.append(f"## {sheet.title}\n\n{md_table}")
    finally:
        workbook.close()

    if not parts:
        raise ValueError(f"这个 xlsx 里没有读到任何有内容的工作表：{path}")

    md = "\n\n".join(parts)
    title = Path(path).stem
    return ParsedDoc(
        title=title,
        markdown=md,
        source_path=str(path),
        source_type="xlsx",
        meta={"sheets": sheet_count, "truncated": truncated},
    )


def load_pptx(path: str) -> ParsedDoc:
    """pptx：每页转 `## 第N页 {标题}` + 各文本框内容，跳过占位符。"""
    from pptx import Presentation

    presentation = Presentation(path)
    parts: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        title_text = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title_text = slide.shapes.title.text.strip()

        body_lines: list[str] = []
        for shape in slide.shapes:
            # 标题已经单独处理
            if slide.shapes.title is not None and shape == slide.shapes.title:
                continue
            if getattr(shape, "has_table", False):
                md_table = _rows_to_md_table(
                    [[c.text.strip() for c in row.cells] for row in shape.table.rows]
                )
                if md_table:
                    body_lines.append(md_table)
                continue
            if not getattr(shape, "has_text_frame", False):
                continue
            text = shape.text_frame.text.strip()
            # 空占位符直接跳过，不产出空的 bullet
            if text:
                body_lines.append(text)

        if not body_lines and not title_text:
            continue

        heading = f"## 第{index}页"
        if title_text:
            heading += f" {title_text}"
        if body_lines:
            parts.append(f"{heading}\n\n" + "\n\n".join(body_lines))
        else:
            parts.append(heading)

    md = "\n\n".join(parts)
    title = Path(path).stem
    return ParsedDoc(
        title=title,
        markdown=md,
        source_path=str(path),
        source_type="pptx",
        meta={"slides": len(presentation.slides)},
    )


def load_url(url: str) -> ParsedDoc:
    """安全抓取网页，拒绝内网地址并限制重定向和响应大小。"""
    import httpx
    import trafilatura
    from bs4 import BeautifulSoup
    from ..config import get_settings

    def safe_url(candidate: str) -> None:
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("URL 必须是有效的 http/https 地址")
        try:
            addresses = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            raise ValueError("URL 主机名无法解析") from exc
        for item in addresses:
            address = ipaddress.ip_address(item[4][0])
            if any((address.is_private, address.is_loopback, address.is_link_local,
                    address.is_reserved, address.is_multicast, address.is_unspecified)):
                raise ValueError("URL 不允许访问内网或保留地址")

    settings = get_settings()
    safe_url(url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; local-rag/0.1)"}
    try:
        with httpx.Client(timeout=20.0, follow_redirects=False) as client:
            current = url
            for _ in range(settings.url_max_redirects + 1):
                safe_url(current)
                response = client.get(current, headers=headers)
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("网页重定向地址无效")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in {x.strip() for x in settings.url_allowed_content_types.split(",")}:
                    raise ValueError("网页内容类型不受支持")
                if len(response.content) > settings.url_max_bytes:
                    raise ValueError("网页响应超过大小限制")
                html = response.text
                break
            else:
                raise ValueError("网页重定向次数过多")
    except (httpx.HTTPError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("抓取网页失败") from exc

    soup = BeautifulSoup(html, "html.parser")
    page_title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    if not page_title:
        og = soup.find("meta", property="og:title")
        page_title = (og.get("content") or "").strip() if og else ""

    try:
        extracted = trafilatura.extract(
            html, include_links=False, include_tables=True, include_comments=False
        )
    except Exception:  # trafilatura 对畸形 HTML 偶尔会抛，交给下面的降级分支
        extracted = None

    if extracted and len(extracted.strip()) >= 100:
        body = extracted
        method = "trafilatura"
    else:
        # 降级：去掉 script/style/导航类标签，再取文本
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()
        body = soup.get_text(separator="\n", strip=True)
        method = "beautifulsoup"

    if not body.strip():
        raise ValueError(f"网页里没抽到正文内容：{url}")

    md = f"# {page_title or url}\n\n{body.strip()}\n"
    return ParsedDoc(
        title=page_title or url,
        markdown=md,
        source_path=url,
        source_type="url",
        meta={"method": method, "chars": len(body)},
        url=url,
    )


def _rows_to_md_table(rows: list[list[str]]) -> str:
    """把二维数组转成 Markdown 表格。第一行当表头。"""
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    header = rows[0] + [""] * (width - len(rows[0]))
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in header) + " |"]
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in rows[1:]:
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in padded) + " |")
    return "\n".join(lines)
