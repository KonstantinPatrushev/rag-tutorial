"""Cross-encoder reranking — финальная пересортировка кандидатов.

Bi-encoder (dense retrieval) кодирует запрос и документ независимо — быстро,
но грубо. Cross-encoder подаёт пару (запрос, документ) в одну модель и выдаёт
прямую оценку релевантности — точнее, но дороже. Поэтому реранкером
пересортировываем только top-N кандидатов из гибридного retrieval (классический
SOTA-паттерн retrieve-then-rerank).

Модель грузится лениво и кэшируется. Если reranker отключён в конфиге или модель
недоступна — pipeline продолжает работать на RRF-фьюжне.
"""

from __future__ import annotations

from app.config import RERANK_MODEL

_RERANKER_CACHE: dict[str, object] = {}


def get_reranker(model_name: str = RERANK_MODEL):
    if model_name not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder

        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def rerank(
    query: str,
    candidates: list[dict],
    model_name: str = RERANK_MODEL,
) -> list[dict]:
    """Пересортировывает кандидатов по cross-encoder score (убыв.).

    Каждому кандидату добавляет поле `rerank_score`. Возвращает новый
    отсортированный список (исходный не мутируется по порядку).
    """
    if not candidates:
        return []
    model = get_reranker(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    scored = []
    for cand, score in zip(candidates, scores):
        item = dict(cand)
        item["rerank_score"] = float(score)
        scored.append(item)
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored
