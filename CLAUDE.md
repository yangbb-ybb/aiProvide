# CLAUDE.md

本文件给 AI 编码助手（Claude Code 等）阅读。**动手前先读本文件，再读 [AGENTS.md](AGENTS.md)。**

## 这是什么项目

本地运行的 RAG 知识库问答系统。**不调用任何外部 AI 服务**，全部推理在本机 Apple M1 上完成，模型用开源权重（Ollama + fastembed/ONNX）。

完整设计见 [docs/项目逻辑规划.md](docs/项目逻辑规划.md)。

## 铁律

1. **不许引入 LangChain / LlamaIndex / Spring AI / LangChain4j 等编排框架。** RAG 链路（解析 / 分块 / 向量化 / 召回 / 融合 / 重排 / 提示词）必须手写，这是项目的核心目的，不是偷懒的例外。
2. **不许调用任何云端 AI API。** 全部推理走本机：生成走 Ollama，向量化和重排走 fastembed（ONNX）。唯一允许的外部调用是 M4 阶段的**搜索**接口（且搜索结果必须用本地模型清洗，不引用别人的 AI 总结）。
3. **中文必须分词。** BM25 索引用 jieba 切词。LanceDB 内置 FTS 是英文分词器，不要用它做中文检索。
4. **代码注释、文档、提交信息用中文。**
5. **新增依赖要在 [AGENTS.md](AGENTS.md) 的依赖清单里说明理由**，并注明体积（磁盘只剩 18GB 是硬约束）。

## 目录职责

| 路径 | 职责 |
|---|---|
| `server/app/ingest/` | 各格式解析成 Markdown + 结构感知分块 + 入库 |
| `server/app/rag/` | 向量化 / 混合召回 / 重排 / 提示词 / 端到端编排 |
| `server/app/store/` | LanceDB 向量仓库 + bm25s 关键词索引 |
| `server/app/llm/` | Ollama 客户端（**唯一**与推理后端耦合的地方） |
| `server/app/search/` | `SearchProvider` 抽象，M4 才接真实供应商 |
| `server/app/eval/` | 评测集与评测脚本 |
| `server/data/` | 运行时数据，**不入 git** |
| `web/src/views/` | Chat / Library / RetrievalDebug 三个视图 |

## 常用命令

```bash
# 后端
cd server && uv sync
uv run uvicorn app.main:app --reload --port 8000
uv run python -m app.eval.run --k 5

# 前端
cd web && npm run dev

# 模型
ollama list && ollama ps
ollama pull qwen3:4b
```

## 改动纪律

- **改检索相关的参数（分块大小 / top-k / top-n / RRF K / 重排开关）后，跑一次 `uv run python -m app.eval.run`**，把指标记进提交信息。不跑评测就不算改完 —— 否则调参全靠感觉。
- 改提示词同样要跑评测（答案命中率会变）。
- 保持 `llm/ollama.py` 是唯一的后端耦合点，不要在业务代码里直接发 HTTP 请求。
