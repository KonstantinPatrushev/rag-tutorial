"""Генерация ответа: retrieve → отбор релевантного → LLM (или extractive fallback).

Поведение grounded-RAG:
  - ответ строится ТОЛЬКО из найденных фрагментов;
  - каждый источник пронумерован, ответ содержит ссылки [n];
  - если ни один фрагмент не проходит порог релевантности — явный отказ.

Режимы:
  - llm        — внешняя LLM по OpenAI-совместимому API (если задан ключ);
  - extractive — офлайн-сборка ответа из топ-фрагментов (без ключа).
"""

from __future__ import annotations

from app.config import (
    MIN_FUSION_SCORE,
    MIN_RERANK_SCORE,
    TOP_K,
    llm_enabled,
)
from app.prompts import (
    REFUSAL_EMPTY_QUESTION,
    REFUSAL_NO_CONTEXT,
    SYSTEM_RULES,
    build_user_prompt,
)
from app.retriever import Retriever


def is_relevant(hit: dict) -> bool:
    """Порог релевантности зависит от того, что было финальным сигналом."""
    if hit.get("score_kind") == "rerank":
        return hit["score"] >= MIN_RERANK_SCORE
    return hit["score"] >= MIN_FUSION_SCORE


def format_sources(hits: list[dict]) -> list[dict]:
    """Источники для UI/ответа — со всеми диагностическими score."""
    out = []
    for hit in hits:
        out.append(
            {
                "doc_id": hit["doc_id"],
                "name": hit.get("name", ""),
                "text": hit["text"],
                "score": hit["score"],
                "score_kind": hit.get("score_kind", "fusion"),
                "fusion_score": hit.get("fusion_score"),
                "rerank_score": hit.get("rerank_score"),
                "dense_rank": hit.get("dense_rank"),
                "bm25_rank": hit.get("bm25_rank"),
            }
        )
    return out


def extractive_answer(question: str, relevant: list[dict]) -> str:
    """Офлайн-ответ без LLM: сводка топ-фрагментов с нумерованными ссылками."""
    parts = ["На основании найденных фрагментов (extractive-режим, без LLM):", ""]
    for i, hit in enumerate(relevant, 1):
        snippet = hit["text"].strip()
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + "…"
        parts.append(f"[{i}] {hit.get('name', '')}".rstrip())
        parts.append(snippet)
        parts.append("")
    parts.append("Источники: " + ", ".join(f"[{i}]" for i in range(1, len(relevant) + 1)))
    return "\n".join(parts).strip()


def llm_answer(question: str, relevant: list[dict]) -> str:
    """Ответ через внешнюю LLM по grounded-промпту с цитатами."""
    from app.llm import chat

    return chat(SYSTEM_RULES, build_user_prompt(question, relevant))


def ask(
    question: str,
    k: int = TOP_K,
    retriever: Retriever | None = None,
    use_llm: bool | None = None,
) -> dict:
    """Вопрос → {answer, sources, mode, refused}."""
    if not question.strip():
        return {"answer": REFUSAL_EMPTY_QUESTION, "sources": [], "mode": "none", "refused": True}

    r = retriever or Retriever()
    hits = r.search(question.strip(), k=k)
    sources = format_sources(hits)
    relevant = [h for h in hits if is_relevant(h)]

    if not relevant:
        return {
            "answer": REFUSAL_NO_CONTEXT,
            "sources": sources,
            "mode": "refused",
            "refused": True,
        }

    want_llm = llm_enabled() if use_llm is None else use_llm
    if want_llm:
        answer = llm_answer(question, relevant)
        mode = "llm"
    else:
        answer = extractive_answer(question, relevant)
        mode = "extractive"

    return {"answer": answer, "sources": sources, "mode": mode, "refused": False}
