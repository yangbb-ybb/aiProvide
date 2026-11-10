"""评测脚本。

用法：
    uv run python -m app.eval.run --k 5
    uv run python -m app.eval.run --k 5 --no-rerank   # A/B 对照

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
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
QA_PATH = EVAL_DIR / "qa.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


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


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """期望块出现在 top-k 里的比例。"""
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def mrr_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """第一个期望块的倒数排名。"""
    expected_set = set(expected)
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in expected_set:
            return 1.0 / i
    return 0.0


def answer_hit(answer: str, keywords: list[str]) -> bool:
    """答案是否包含全部关键词（子串匹配）。"""
    return all(k in answer for k in keywords)


def main() -> None:
    """评测入口。"""
    parser = argparse.ArgumentParser(description="本地 RAG 知识库评测")
    parser.add_argument("--k", type=int, default=5, help="检索 top-k")
    parser.add_argument("--no-rerank", action="store_true", help="A/B：关掉重排")
    parser.add_argument("--qa", type=Path, default=QA_PATH, help="评测集路径")
    args = parser.parse_args()

    qa = load_qa(args.qa)
    if not qa:
        print(f"评测集为空或不存在：{args.qa}")
        print("请先在 qa.jsonl 里写几条问题（每条含 question / answer / chunk_ids）")
        return

    print(f"评测集 {len(qa)} 条，k={args.k}，rerank={'off' if args.no_rerank else 'on'}")
    print("检索与生成链路待 M1 实现后接入 —— 目前只验证评测框架本身可运行。")


if __name__ == "__main__":
    main()
