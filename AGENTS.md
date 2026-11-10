# AGENTS.md

本地 RAG 知识库项目的开发规范。配套 [CLAUDE.md](CLAUDE.md)（给 AI 助手看的速查版）。

---

## 1. 项目边界

- **完全本地**。推理走本机 Ollama（生成）+ fastembed（向量化 / 重排，ONNX）。
- **不调用云端 AI API**。唯一的外部依赖是 M4 阶段的搜索接口，且搜索返回的正文必须由本地模型清洗后再进上下文。
- **RAG 链路全部手写**，不用 LangChain / LlamaIndex / Spring AI / LangChain4j。

## 2. 目录结构

```
server/app/
├── main.py          FastAPI 装配、CORS、路由注册、静态资源
├── config.py        pydantic-settings 读 .env
├── schemas.py       Pydantic 请求/响应模型
├── api/             只做参数校验和响应拼装，业务逻辑下沉到 rag/ ingest/
├── ingest/          loader / chunker / indexer
├── rag/             embedder / retriever / reranker / prompt / chain
├── llm/             ollama.py（唯一与推理后端耦合的文件）
├── search/          base.py（SearchProvider 协议 + NullProvider）
├── store/           vector_store.py（LanceDB）/ bm25_index.py（bm25s）
└── eval/            qa.jsonl（评测集）/ run.py（评测脚本）
```

**依赖方向**：`api → rag → store / llm`，`api → ingest → store`。不允许反向依赖，不允许 `rag` 互相 import 形成环。

## 3. 编码约定

- Python 3.13，`from __future__ import annotations`，全量类型注解。
- 异步边界清晰：FastAPI 路由与 Ollama 客户端用 `async`；fastembed、bm25s 是同步 CPU 密集调用，**用 `run_in_threadpool` / `asyncio.to_thread` 包起来**，不要阻塞事件循环。
- 注释和 docstring 用中文，写清楚**为什么**这么做（不只是做了什么）。
- 错误处理要给出**可操作的中文提示**，例如「Ollama 未启动，请执行 `ollama serve`」而不是「ConnectionError」。
- 不用 `print`，用 `logging`。

## 4. 依赖清单

新增依赖必须先在本表登记，说明理由和体积。磁盘只剩 18GB，不是随便加的。

| 依赖 | 用途 | 体积 |
|---|---|---|
| fastapi / uvicorn | Web 框架 | 小 |
| httpx | 调 Ollama | 小 |
| pydantic-settings | 配置 | 小 |
| lancedb + pyarrow | 向量仓库（嵌入式） | 中 |
| fastembed | Qwen3-Embedding-0.6B-Q 向量化 + bge-reranker-v2-m3-int8 重排（ONNX，不含 PyTorch） | ~150MB + 模型 1.7GB |
| jieba | 中文分词（BM25 用） | 小 |
| bm25s | BM25 检索 | 小 |
| numpy | 向量运算 | 小 |
| PyMuPDF | PDF 解析（AGPL 许可，日后商用需换 pdfplumber） | 中 |
| python-docx / openpyxl / python-pptx | Office 解析 | 小 |
| trafilatura + beautifulsoup4 | 网页正文抽取 | 中 |
| 前端：vue / vite / pinia | UI | — |
| pytest（dev 组） | 跑 `tests/` 下的单元测试 | 小（仅开发期，不进运行时） |

**刻意不引入**：torch、transformers、sentence-transformers、langchain、llama-index、chroma、milvus-client、faiss。

**Python 版本必须锁 3.13**：`server/.python-version` 写死 `3.13`，`pyproject.toml` 里 `requires-python = ">=3.13,<3.14"`。
原因：fastembed 的运行时是 `onnxruntime`，它目前没有 cp315 的 wheel；而 `brew install ollama` 会把 Homebrew 默认 python3 升到 3.15，
uv 若按 `>=3.13` 解析就会选中 3.15，导致 `uv sync` 直接失败（"doesn't have a source distribution or wheel for the current platform"）。
写成 `>=3.13` 是错的 —— 只能写死上界。

**模型下载要走镜像，缓存要放持久目录**（`server/app/hf_env.py`）：
- `HF_ENDPOINT=https://hf-mirror.com` —— 直连 huggingface.co 会一直超时挂住。
- `FASTEMBED_CACHE_PATH=~/.cache/fastembed` —— fastembed 默认把权重放系统临时目录
  （macOS 的 `/var/folders/.../T/`），重启或磁盘紧张时会被清掉，1.7GB 白下。
- 两个变量都必须在 `import fastembed` 之前设好，所以统一在 `app/config.py` 里设置。

## 5. 检索链路的硬性要求

1. **BM25 必须用 jieba 分词**，索引持久化到 `data/bm25/`。
2. **RRF 融合自己实现**：`score(d) = 1/(K + rank_dense(d)) + 1/(K + rank_bm25(d))`，K 默认 60。只出现在一路的，另一路贡献 0。
3. **LanceDB 只当向量仓库 + 元数据过滤**，不要用它的 hybrid search（内置 FTS 是英文分词器，中文会崩）。
4. 重排后每块 `content` 裁到 `CONTEXT_CHUNK_CHARS`（默认 800 字）。
5. 生成的上下文里每块必须带 `[n]` 编号和来源标签，否则答案无法溯源。

## 6. 参数与依据

改这些参数前先理解代价。默认值见 [docs/项目逻辑规划.md](docs/项目逻辑规划.md) §4。

| 参数 | 默认 | 依据 |
|---|---|---|
| `CHUNK_TARGET_CHARS` | 500 | <300 语义不完整；>800 超出嵌入模型 512-token 训练窗口 |
| `CHUNK_OVERLAP_CHARS` | 80 | 约 16%，够补断句又不至于让相邻块挤占 top-k 名额 |
| `RETRIEVE_TOP_N` | 50 | 召回宁多勿漏；50 条重排成本约 1~2s |
| `RRF_K` | 60 | 业界通用值，与 LanceDB 官方默认一致 |
| `RERANK_TOP_K` | 5 | 5×800 字 ≈ 4000 token，成本与质量平衡点 |
| `temperature` | 0.1 | RAG 是提取任务不是创作任务，低温度更忠于资料 |

## 7. 评测纪律

**改动检索参数或提示词后，必须跑评测并把指标写进提交信息。**

```bash
uv run python -m app.eval.run --k 5
uv run python -m app.eval.run --k 5 --no-rerank   # A/B 对照
```

指标定义（Recall@k / MRR@k / 答案命中率 / 端到端正确率）见规划文档 §4.8。结果自动存入 `server/app/eval/results/{timestamp}.json` 保留历史。

不跑评测就不算改完 —— 否则调参全靠感觉，项目就无法长期演进。

## 8. 提交信息

中文，格式：

```
<类型>: <简述>

<为什么这么改 / 评测指标变化>
```

类型：`feat` / `fix` / `refactor` / `docs` / `chore` / `perf`。

## 9. 已知边界

- **PDF OCR 不支持**。loader 检测到某页文本 <50 字符会标记 `needs_ocr` 并在接口返回里告知，不静默产出空块。
- **PyMuPDF 是 AGPL 许可**，个人项目无碍，若日后要闭源商用必须换成 pdfplumber。
- **M1 16GB 内存**：8B 模型常驻约 8GB + embedding 0.6GB + reranker 0.5GB + 系统，紧张时用 `qwen3:4b` 或关掉 `RERANK_ENABLED`。
- **磁盘只剩 18GB**，拉模型前先确认空间；必要时 `export OLLAMA_MODELS=/Volumes/<盘>/ollama`。
