"""Тесты hybrid retrieval на маленьком реальном индексе.

Строит крошечный гибридный индекс (4 чанка) настоящим эмбеддером + BM25 и
проверяет сквозной поиск. Модель e5-small кэшируется после первой загрузки,
поэтому тест выполняется быстро. Reranker отключён — отдельная тяжёлая модель.
"""

import pytest

from app.embedder import embed_passages
from app.index import HybridIndex, build_bm25, build_faiss
from app.retriever import Retriever

CHUNKS = [
    {"chunk_id": "0_0", "doc_id": "0", "name": "Байконур",
     "text": "Россия арендует у Казахстана космодром Байконур для космических запусков."},
    {"chunk_id": "1_0", "doc_id": "1", "name": "Белки",
     "text": "Белки — высокомолекулярные органические вещества из аминокислот."},
    {"chunk_id": "2_0", "doc_id": "2", "name": "Инфляция",
     "text": "Инфляция — это устойчивый рост общего уровня цен в экономике."},
    {"chunk_id": "3_0", "doc_id": "3", "name": "Водоросли",
     "text": "Органические остатки представлены известковыми выделениями водорослей."},
]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    texts = [c["text"] for c in CHUNKS]
    embeddings = embed_passages(texts)
    index = HybridIndex(
        chunks=CHUNKS,
        faiss_index=build_faiss(embeddings),
        bm25=build_bm25(texts),
        embed_model="intfloat/multilingual-e5-small",
    )
    return Retriever(index=index, use_reranker=False)


def test_returns_k_results(retriever):
    results = retriever.search("что арендует Россия в Казахстане?", k=2)
    assert len(results) == 2


def test_results_have_required_fields(retriever):
    results = retriever.search("из чего состоят белки?", k=3)
    assert results
    for hit in results:
        for field in ("doc_id", "text", "name", "score", "fusion_score", "score_kind"):
            assert field in hit
        assert isinstance(hit["score"], float)


def test_semantic_match_finds_right_doc(retriever):
    # запрос-перефразировка без точных слов из текста -> проверяем semantic-составляющую
    results = retriever.search("аренда космодрома у соседнего государства", k=1)
    assert results[0]["doc_id"] == "0"


def test_lexical_match_finds_right_doc(retriever):
    results = retriever.search("инфляция рост цен", k=1)
    assert results[0]["doc_id"] == "2"


def test_empty_query_returns_empty(retriever):
    assert retriever.search("") == []
    assert retriever.search("   ") == []
