"""Тесты generation: grounded-ответ, отказ, формат источников.

Retriever подменяется заглушкой (FakeRetriever), поэтому тесты быстрые и не
грузят нейромодели — проверяется именно логика generator (порог релевантности,
отказ, цитаты в extractive-режиме).
"""

from app.generator import ask, is_relevant
from app.prompts import REFUSAL_EMPTY_QUESTION, REFUSAL_NO_CONTEXT, build_context_block


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, k=5):
        return self._hits[:k]


def _hit(doc_id, score, kind="rerank", text="Текст фрагмента про тему."):
    return {
        "doc_id": doc_id,
        "name": f"Документ {doc_id}",
        "text": text,
        "score": score,
        "score_kind": kind,
        "fusion_score": 0.03,
        "rerank_score": score if kind == "rerank" else None,
        "dense_rank": 1,
        "bm25_rank": 2,
    }


def test_is_relevant_threshold():
    assert is_relevant(_hit("1", 5.0, "rerank")) is True
    assert is_relevant(_hit("2", -8.0, "rerank")) is False


def test_empty_question_refuses():
    result = ask("   ", retriever=FakeRetriever([]))
    assert result["refused"] is True
    assert result["answer"] == REFUSAL_EMPTY_QUESTION


def test_refuses_when_all_below_threshold():
    retriever = FakeRetriever([_hit("1", -9.0), _hit("2", -10.0)])
    result = ask("вопрос вне базы", retriever=retriever, use_llm=False)
    assert result["refused"] is True
    assert result["answer"] == REFUSAL_NO_CONTEXT
    # источники всё равно возвращаются для диагностики
    assert len(result["sources"]) == 2


def test_extractive_answer_has_citations_and_sources():
    retriever = FakeRetriever([_hit("7", 6.0, text="Россия арендует космодром Байконур.")])
    result = ask("что арендует Россия?", retriever=retriever, use_llm=False)
    assert result["refused"] is False
    assert result["mode"] == "extractive"
    assert "[1]" in result["answer"]
    assert result["sources"][0]["doc_id"] == "7"


def test_context_block_numbering():
    block = build_context_block([_hit("1", 5.0), _hit("2", 4.0)])
    assert "[1]" in block and "[2]" in block
