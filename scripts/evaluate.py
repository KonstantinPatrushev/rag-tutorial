"""Оценка качества retrieval на golden-set (SberQuAD вопрос→контекст).

Метрики (для каждого вопроса известен gold_doc_id — правильный документ):
  - hit@k   — доля вопросов, где правильный документ попал в top-k;
  - recall@k — то же (gold ровно один), удобная агрегатная метрика;
  - MRR     — средний обратный ранг первого правильного документа.

Скрипт сравнивает два режима — гибрид + RRF и гибрид + RRF + reranker —
и показывает прирост от cross-encoder reranking (это и есть измеримое
доказательство пользы улучшения).

Запуск:
    uv run python scripts/evaluate.py [--sample N] [--k K]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EVAL_SET
from app.index import HybridIndex
from app.retriever import Retriever

EVAL_K = 10


def load_eval() -> list[dict]:
    if not EVAL_SET.exists():
        raise FileNotFoundError(
            f"Нет golden-set: {EVAL_SET}. Запустите scripts/prepare_datasets.py"
        )
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))["eval"]


def rank_of_gold(results: list[dict], gold_doc_id) -> int | None:
    """Позиция (1-based) первого чанка нужного документа, иначе None."""
    gold = str(gold_doc_id)
    for rank, hit in enumerate(results, 1):
        if str(hit["doc_id"]) == gold:
            return rank
    return None


def evaluate(retriever: Retriever, eval_items: list[dict], k: int) -> dict:
    ranks: list[int | None] = []
    for item in eval_items:
        results = retriever.search(item["question"], k=k)
        ranks.append(rank_of_gold(results, item["gold_doc_id"]))

    n = len(ranks)

    def hit_at(kk: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= kk) / n

    mrr = sum((1.0 / r) for r in ranks if r is not None) / n
    return {
        "n": n,
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        f"recall@{k}": hit_at(k),
        "mrr": mrr,
    }


def fmt(metrics: dict) -> str:
    keys = [m for m in metrics if m != "n"]
    return "  ".join(f"{key}={metrics[key]:.3f}" for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    # reranker на CPU небыстрый: 100 вопросов — разумный баланс времени и точности
    parser.add_argument("--sample", type=int, default=100, help="сколько вопросов оценивать")
    parser.add_argument("--k", type=int, default=EVAL_K)
    args = parser.parse_args()

    eval_items = load_eval()[: args.sample]
    index = HybridIndex.load()
    print(f"Golden-set: {len(eval_items)} вопросов, k={args.k}\n")

    results = {}
    for label, use_rr in [("hybrid+RRF", False), ("hybrid+RRF+rerank", True)]:
        retriever = Retriever(index=index, use_reranker=use_rr)
        t0 = time.time()
        metrics = evaluate(retriever, eval_items, args.k)
        dt = time.time() - t0
        results[label] = metrics
        print(f"{label:22s} {fmt(metrics)}  ({dt:.1f}s)")

    report = ROOT / "data" / "index" / "eval_report.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён -> {report}")


if __name__ == "__main__":
    main()
