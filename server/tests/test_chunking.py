"""分块与提示词的单元测试。

这些是纯 Python 逻辑，不依赖模型，跑得快，改分块参数时先跑它们：

    uv run pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest.chunker import chunk_markdown
from app.rag.prompt import _truncate, build_context
from app.rag.retriever import ScoredChunk

SAMPLE = """# 项目总览

开篇段落，说明这个项目是干什么的。

## 部署要求

生产环境必须设置 DATABASE_URL 和 REDIS_URL 两个环境变量，否则服务启动会失败。
另外还需要一台至少 4 核 16G 的机器。

### 环境变量清单

- DATABASE_URL：数据库连接串
- REDIS_URL：缓存连接串

| 变量 | 是否必填 | 默认值 |
|---|---|---|
| DATABASE_URL | 是 | 无 |
| REDIS_URL | 是 | 无 |

```bash
export DATABASE_URL=postgres://localhost/app
export REDIS_URL=redis://localhost:6379
```

## 常见问题

服务启动报 `Connection refused` 怎么办？先确认 Redis 进程是否还在。
"""


def test_heading_stack():
    """标题层级要维护成完整的祖先链。"""
    chunks = chunk_markdown(SAMPLE, {})
    paths = {tuple(c.heading_path) for c in chunks}
    assert ("项目总览",) in paths
    assert ("项目总览", "部署要求") in paths
    assert ("项目总览", "部署要求", "环境变量清单") in paths
    assert ("项目总览", "常见问题") in paths


def test_no_empty_chunks():
    """不能产出空块。"""
    chunks = chunk_markdown(SAMPLE, {})
    assert chunks
    for c in chunks:
        assert c.char_len > 0
        assert c.content == c.content.strip()


def test_code_block_kept_whole():
    """代码块不能被切开。"""
    chunks = chunk_markdown(SAMPLE, {})
    assert any("export REDIS_URL" in c.content for c in chunks)


def test_table_header_kept():
    """表格切分后每片都要带表头。"""
    rows = "\n".join(f"| 第{i}行 | 值{i} |" for i in range(80))
    md = f"# 表\n\n| 名称 | 数值 |\n|---|---|\n{rows}\n"
    chunks = chunk_markdown(md, {}, target_chars=500, overlap_chars=80)
    assert len(chunks) >= 3
    for c in chunks:
        assert "名称 | 数值" in c.content
    # 行不能丢
    assert sum(c.content.count("| 第") for c in chunks) == 80


def test_long_paragraph_split_with_overlap():
    """超长段落要切开，且相邻块之间有重叠。"""
    long_p = "这是一句很长的中文说明。" * 120
    chunks = chunk_markdown(f"# 长文\n\n{long_p}\n", {}, target_chars=500, overlap_chars=80)
    assert len(chunks) >= 3
    assert chunks[0].content[-20:] in chunks[1].content


def test_hash_in_code_block_is_not_heading():
    """代码块里的 # 不是标题。"""
    md = "# 真标题\n\n```python\n# 注释不是标题\nx = 1\n```\n"
    chunks = chunk_markdown(md, {})
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["真标题"]


def test_page_markers():
    """PDF 页码标记要切块并归属正确。"""
    md = "# 文档\n\n<!-- page: 1 -->\n\n第一页内容。\n\n<!-- page: 7 -->\n\n第七页内容。\n"
    chunks = chunk_markdown(md, {})
    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[1].page == 7


def test_empty_document():
    """空文档不能炸。"""
    assert chunk_markdown("", {}) == []
    assert chunk_markdown("# 只有标题\n", {}) == []


def test_context_numbering_and_truncation():
    """上下文要带 [n] 编号，来源标签要全。"""
    chunks = [
        ScoredChunk(
            content="正文内容" * 500,
            source_path="/tmp/a.md",
            heading_path=["第3章", "3.2 环境变量"],
            page=12,
        )
    ]
    context = build_context(chunks, max_chars=100)
    assert context.startswith("[1] (来源：/tmp/a.md > 第3章 > 3.2 环境变量 第12页)")
    # 正文被裁到 100 字左右
    body = context.split("\n", 1)[1]
    assert len(body) <= 110


def test_truncate_at_sentence_boundary():
    """裁剪要落在句读边界。"""
    text = "第一句话在这里。第二句话也在这里。第三句话更长一点。"
    out = _truncate(text, 12)
    assert out.endswith("。")
    assert len(out) <= 13
