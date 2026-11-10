# 本地 RAG 知识库

> 用自己的代码 + 本地开源模型，做一个能问自己资料的 AI 助手。
>
> **不调用任何外部 AI 服务**，推理 100% 在本机（Apple M1）完成。

把 Markdown / PDF / Word / Excel / PPT / 网页链接喂进去，它基于**你的**资料回答问题，并标注引用来源。

---

## 快速开始

```bash
# 1. 装 Ollama（推理后端）
brew install ollama && brew services start ollama
ollama pull qwen3:4b

# 2. 起后端
cd server && uv sync
uv run uvicorn app.main:app --reload --port 8000

# 3. 起前端
cd web && npm install && npm run dev
```

打开 http://localhost:5173

---

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | Python 3.13 + uv + FastAPI |
| 前端 | Vue 3 + Vite + Pinia |
| 生成模型 | Ollama + `qwen3:4b`（开源权重，Metal 加速） |
| 向量模型 | `Qwen/Qwen3-Embedding-0.6B-Q`（ONNX int8，1024 维，无需 PyTorch） |
| 重排模型 | `BAAI/bge-reranker-v2-m3-int8`（ONNX int8） |
| 向量库 | LanceDB（嵌入式，无服务） |
| 关键词索引 | jieba + bm25s（自建，中文分词） |

RAG 全链路（解析 → 分块 → 向量化 → 混合召回 → 重排 → 提示词）**每一环都是自己写的**，不使用 LangChain / LlamaIndex 等编排框架。

---

## 文档

**完整规划见 [docs/项目逻辑规划.md](docs/项目逻辑规划.md)** —— 里面有全部的架构设计、参数取值依据、被否决的方案对照、风险对策和命令速查。

---

## 开发约定

见 [CLAUDE.md](CLAUDE.md) 与 [AGENTS.md](AGENTS.md)。
