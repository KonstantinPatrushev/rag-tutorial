"""Тесты метрик оценки retrieval (scripts/evaluate.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as ev  # noqa: E402


class FakeRetriever:
    """Возвращает фиксированный ранжированный список doc_id."""

    def __init__(self, ranking):
        self._ranking = ranking

    def search(self, query, k=10):
        return [{"doc_id": d} for d in self._ranking[:k]]


def test_rank_of_gold_found():
    results = [{"doc_id": "3"}, {"doc_id": "7"}, {"doc_id": "1"}]
    assert ev.rank_of_gold(results, 7) == 2  # сравнение int gold со str doc_id


def test_rank_of_gold_missing():
    results = [{"doc_id": "3"}, {"doc_id": "1"}]
    assert ev.rank_of_gold(results, 99) is None


def test_evaluate_perfect_retriever():
    items = [{"question": "q1", "gold_doc_id": "5"}, {"question": "q2", "gold_doc_id": "9"}]
    # каждый вопрос: gold-документ первым
    r = FakeRetriever(["5", "9"])  # для q1 -> 5 первый; для q2 -> 5 первый, 9 второй

    # точнее зададим через подмену по вопросу
    class PerfectRetriever:
        def search(self, query, k=10):
            gold = "5" if query == "q1" else "9"
            return [{"doc_id": gold}, {"doc_id": "0"}]

    metrics = ev.evaluate(PerfectRetriever(), items, k=10)
    assert metrics["hit@1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_evaluate_miss_gives_zero():
    items = [{"question": "q", "gold_doc_id": "5"}]
    r = FakeRetriever(["1", "2", "3"])
    metrics = ev.evaluate(r, items, k=3)
    assert metrics["hit@1"] == 0.0
    assert metrics["mrr"] == 0.0
