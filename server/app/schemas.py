"""请求 / 响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    """用于理解追问指代的短期会话记录。"""

    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""

    question: str = Field(min_length=1, description="用户问题")
    history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=12,
        description="最近几轮会话，仅用于理解追问和生成连贯回答",
    )
    session_id: str | None = Field(default=None, description="会话 id，v1 不做持久化")
    top_k: int | None = Field(default=None, ge=1, le=20, description="送进上下文的块数，默认取 RERANK_TOP_K")
    use_web: bool = Field(default=False, description="M4：是否允许联网搜索")


class Citation(BaseModel):
    """一条引用来源。index 对应正文里的 [n] 编号。"""

    index: int
    source_path: str
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    score: float = 0.0
    preview: str = ""


class SearchRequest(BaseModel):
    """POST /api/search 请求体。给检索调试面板用，不经过 LLM。"""

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class UrlIngestRequest(BaseModel):
    """POST /api/documents/url 请求体。"""

    url: str = Field(min_length=1)


class IngestResponse(BaseModel):
    """文档入库结果。"""

    doc_id: str
    source_path: str
    source_type: str
    chunks: int
    needs_ocr: bool = Field(default=False, description="PDF 无文本层，需人工 OCR")


class DocumentSummary(BaseModel):
    """文档列表项。"""

    doc_id: str
    source_path: str
    source_type: str
    title: str = ""
    chunks: int = 0
    mtime: float = 0.0
