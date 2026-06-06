"""Проверка generation: 3 рабочих вопроса + 1 negative (отказ)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import llm_enabled
from app.generator import ask
from app.index import HybridIndex
from app.retriever import Retriever


def show(label: str, question: str, retriever: Retriever) -> None:
    print(f"\n--- {label}: «{question}» ---")
    result = ask(question, retriever=retriever)
    print(f"режим: {result['mode']}, отказ: {result['refused']}")
    print(f"Ответ:\n{result['answer']}\n")
    for i, src in enumerate(result["sources"][:3], 1):
        print(f"  [{i}] doc_id={src['doc_id']}, score={src['score']:.3f} ({src['score_kind']}), {src['name'][:50]}")


if __name__ == "__main__":
    print(f"LLM включена: {llm_enabled()} (нет ключа -> extractive demo-режим)")
    r = Retriever(index=HybridIndex.load())
    show("Контекст есть", "В каком году истекает договор аренды Байконура?", r)
    show("Контекст есть", "чем представлены органические остатки?", r)
    show("Контекст есть", "из чего состоят белки?", r)
    show("Negative", "как приготовить борщ и какие нужны ингредиенты?", r)
