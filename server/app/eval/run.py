"""评测脚本。

用法：
    uv run python -m app.eval.run --k 5
    uv run python -m app.eval.run --k 5 --no-rerank   # A/B 对照
    uv run python -m app.eval.run --answer            # 额外跑生成层，算答案命中率
    uv run python -m app.eval.run --inspect           # 打印每题召回的块，人工标注用

指标定义见规划文档 §4.8：
- Recall@k   期望 chunk_ids 出现在 top-k 召回结果中的比例（k=1/3/5/10）
- MRR@k      第一个期望 chunk 的倒数排名
- 答案命中率  生成答案包含 answer 全部关键词（子串匹配）
- 端到端正确率 命中且答案命中

结果写入 app/eval/results/{timestamp}.json 并保留历史，
这样每次改参数（分块大小 / top-k / rerank 开关 / 提示词）都能量化对比。

**改动检索参数或提示词后必须跑一次这个脚本，并把指标写进提交信息。**
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
QA_PATH = EVAL_DIR / "qa.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# 指标统一在这几个 k 上算，方便横向对比
METRIC_KS = (1, 3, 5, 10)


def load_qa(path: Path = QA_PATH) -> list[dict]:
    """读取评测集。"""
    if not path.exists():
        return []
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def expected_keys(item: dict) -> set[str]:
    """把一条评测题的「期望命中」归一成可比较的键集合。

    两种标注方式：
    - `chunk_ids`：绝对块 ID（`{doc_id}:{chunk_idx}`）。精确，但 doc_id 由
      「路径 + mtime」哈希而来，文档重新上传就会变，评测集立刻失效。
    - `chunk_idxs` + `source_name`：按「文件名 + 块序号」标注。重新入库也还对得上，
      代价是分块参数一改，块序号就全变了，标注需要重做。

    两种都写了就以 `chunk_ids` 为准。
    """
    ids = item.get("chunk_ids") or []
    if ids:
        return set(ids)
    idxs = item.get("chunk_idxs") or []
    name = item.get("source_name") or ""
    return {f"{name}#{i}" for i in idxs}


def match_target(chunk, item: dict) -> str | None:
    """这个召回块命中了哪一条期望；没命中返回 None。

    `chunk_idxs` 模式下用「文件名后缀 + 块序号」比对：
    上传的文件名前面会带时间戳（`20261110_154814_部署手册.md`），
    所以用 endswith 而不是等值比较，文档重新上传后标注依然有效。
    """
    ids = item.get("chunk_ids") or []
    if ids:
        return chunk.chunk_id if chunk.chunk_id in ids else None

    idxs = item.get("chunk_idxs") or []
    if not idxs:
        return None
    name = item.get("source_name") or ""
    if name and not Path(chunk.source_path).name.endswith(name):
        return None
    if chunk.chunk_idx not in idxs:
        return None
    return f"{name}#{chunk.chunk_idx}"


def recall_at_k(matched: list[str | None], expected: set[str], k: int) -> float:
    """期望块出现在 top-k 里的比例（分母是期望块总数，不是召回条数）。"""
    if not expected:
        return 0.0
    found = {m for m in matched[:k] if m}
    return len(found & expected) / len(expected)


def mrr_at_k(matched: list[str | None], expected: set[str], k: int) -> float:
    """第一个期望块的倒数排名。"""
    for i, m in enumerate(matched[:k], start=1):
        if m and m in expected:
            return 1.0 / i
    return 0.0


def answer_hit(answer: str, keywords: list[str]) -> bool:
    """答案是否包含全部关键词（子串匹配）。"""
    if not keywords:
        return False
    return all(k in answer for k in keywords)


def build_retriever(with_rerank: bool):
    """按需组装召回器。"""
    from app.config import get_settings
    from app.deps import get_bm25_index, get_embedder, get_reranker, get_vector_store
    from app.rag.retriever import Retriever

    settings = get_settings()
    return Retriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        bm25_index=get_bm25_index(),
        reranker=get_reranker() if with_rerank else None,
        top_n=settings.retrieve_top_n,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
    )


def run_retrieval(qa: list[dict], retriever, top_n: int = 10, rerank: bool = True) -> list[dict]:
    """只跑检索层，算 Recall@k 和 MRR@k。

    **量的是「最终进上下文的那批块」的排序**：开了重排就按重排后的顺序算。
    不这么做的话 Recall@k 反映的是 RRF 融合顺序，而重排的价值恰好体现在
    「把对的块顶到前面」—— 量融合顺序就永远看不出重排的贡献。
    """
    rows = []
    for item in qa:
        expected = expected_keys(item)
        chunks = retriever.retrieve_raw(item["question"], top_n)
        if rerank and retriever.reranker is not None and chunks:
            chunks = retriever.reranker.rerank(item["question"], chunks, top_k=len(chunks))
        matched = [match_target(c, item) for c in chunks]
        row = {
            "id": item.get("id", ""),
            "question": item["question"],
            "expected": sorted(expected),
            "retrieved": [c.chunk_id for c in chunks],
            "hits": [],
            "scores": {k: recall_at_k(matched, expected, k) for k in METRIC_KS},
            "mrr": {k: mrr_at_k(matched, expected, k) for k in METRIC_KS},
        }
        for chunk, m in zip(chunks, matched):
            row["hits"].append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_idx": chunk.chunk_idx,
                    "source_path": chunk.source_path,
                    "heading_path": chunk.heading_path,
                    "preview": chunk.content[:120],
                    "scores": {k: round(float(v), 6) for k, v in chunk.scores.items()},
                    "expected": m is not None,
                }
            )
        rows.append(row)
    return rows


async def run_answers(qa: list[dict], retriever, top_k: int) -> list[dict]:
    """跑生成层，算答案命中率和端到端正确率。"""
    from app.deps import get_ollama
    from app.rag.chain import rag_answer_stream

    client = get_ollama()
    rows = []
    for item in qa:
        text_parts: list[str] = []
        async for event_type, payload in rag_answer_stream(
            question=item["question"],
            client=client,
            retriever=retriever,
            top_k=top_k,
        ):
            if event_type == "delta":
                text_parts.append(str(payload))
        answer = "".join(text_parts)
        keywords = item.get("answer") or []
        rows.append(
            {
                "id": item.get("id", ""),
                "answer": answer,
                "keywords": keywords,
                "hit": answer_hit(answer, keywords),
            }
        )
    return rows


def inspect(qa: list[dict], retriever, top_n: int = 10) -> None:
    """打印每题的召回结果 —— 人工标注 chunk_idxs 时用这个。

    输出里的 `[序号]` 是块在文档内的编号（chunk_idx），
    直接把它填进 qa.jsonl 的 `chunk_idxs` 即可。
    """
    for item in qa:
        print(f"\n=== {item.get('id', '')} {item['question']}")
        chunks = retriever.retrieve_raw(item["question"], top_n)
        for i, c in enumerate(chunks, start=1):
            preview = c.content.replace("\n", " ")[:100]
            print(f"  [{i}] chunk_idx={c.chunk_idx}  {' > '.join(c.heading_path)}")
            print(f"       {preview}")


def summarize(rows: list[dict]) -> dict:
    """把逐题结果汇总成平均指标。"""
    if not rows:
        return {}
    summary = {"n": len(rows)}
    for k in METRIC_KS:
        summary[f"recall@{k}"] = round(sum(r["scores"][k] for r in rows) / len(rows), 4)
        summary[f"mrr@{k}"] = round(sum(r["mrr"][k] for r in rows) / len(rows), 4)
    return summary


def main() -> None:
    """评测入口。"""
    parser = argparse.ArgumentParser(description="本地 RAG 知识库评测")
    parser.add_argument("--k", type=int, default=5, help="送进上下文的块数（生成层）")
    parser.add_argument("--no-rerank", action="store_true", help="A/B：关掉重排")
    parser.add_argument("--qa", type=Path, default=QA_PATH, help="评测集路径")
    parser.add_argument("--answer", action="store_true", help="额外跑生成层，算答案命中率")
    parser.add_argument("--inspect", action="store_true", help="打印每题召回的块，人工标注用")
    parser.add_argument("--top-n", type=int, default=10, help="召回条数（算 Recall@k 用）")
    args = parser.parse_args()

    qa = load_qa(args.qa)
    if not qa:
        print(f"评测集为空或不存在：{args.qa}")
        print("请先在 qa.jsonl 里写几条问题（每条含 question / answer / chunk_ids）")
        return

    retriever = build_retriever(with_rerank=not args.no_rerank)

    if args.inspect:
        inspect(qa, retriever, args.top_n)
        return

    print(f"评测集 {len(qa)} 条，重排={'off' if args.no_rerank else 'on'}，召回 {args.top_n} 条")
    started = time.time()
    rows = run_retrieval(qa, retriever, args.top_n, rerank=not args.no_rerank)
    summary = summarize(rows)

    print("\n--- 检索层 ---")
    for k in METRIC_KS:
        print(f"  Recall@{k:<2} {summary[f'recall@{k}']:.4f}    MRR@{k:<2} {summary[f'mrr@{k}']:.4f}")

    if args.answer:
        print("\n--- 生成层 ---")
        answer_rows = asyncio.run(run_answers(qa, retriever, args.k))
        hit = sum(1 for r in answer_rows if r["hit"])
        # 端到端 = 检索命中（@k）且答案命中
        e2e = 0
        by_id = {r["id"]: r for r in rows}
        for r in answer_rows:
            row = by_id.get(r["id"])
            if row and row["scores"].get(args.k, 0) > 0 and r["hit"]:
                e2e += 1
        summary["answer_hit"] = round(hit / len(answer_rows), 4)
        summary["end_to_end"] = round(e2e / len(answer_rows), 4)
        for r in answer_rows:
            mark = "✓" if r["hit"] else "✗"
            print(f"  {mark} {r['id']}  {r['answer'][:60]}")
        print(f"\n  答案命中率 {summary['answer_hit']:.4f}    端到端正确率 {summary['end_to_end']:.4f}")

    summary["elapsed_sec"] = round(time.time() - started, 2)

    # 记下这次评测的配置，否则过几天回头看数字没法解释
    from app.config import get_settings

    settings = get_settings()
    summary["config"] = {
        "rerank": not args.no_rerank,
        "chunk_target_chars": settings.chunk_target_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "retrieve_top_n": settings.retrieve_top_n,
        "rrf_k": settings.rrf_k,
        "rerank_top_k": settings.rerank_top_k,
        "embed_model": settings.embed_model,
        "chat_model": settings.ollama_chat_model,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}.json"
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
