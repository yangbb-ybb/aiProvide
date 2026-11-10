"""SearchProvider 抽象。

M4 阶段在这里接入真实供应商（Tavily / Brave 等）。M1~M3 只用 NullProvider 占位，
接口先定好，避免以后改 chain.py 的结构。

注意：搜索返回的是**别人的网页正文**，不是别人的 AI 总结。入库前必须用
本地模型清洗成要点（见规划文档 §5），不直接引用第三方 AI 生成的摘要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class SearchResult:
    """一条搜索结果。"""

    title: str
    url: str
    snippet: str = ""
    content: str = ""
    score: float = 0.0


@runtime_checkable
class SearchProvider(Protocol):
    """搜索供应商协议。"""

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """按查询返回 top_k 条结果。失败时返回空列表，不要抛异常打断问答。"""
        ...


class NullProvider:
    """占位实现，永远返回空。M4 之前用它。"""

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []
