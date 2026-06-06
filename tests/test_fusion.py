"""Тесты Reciprocal Rank Fusion (чистая логика, без моделей)."""

from app.retriever import reciprocal_rank_fusion


def test_rrf_rewards_top_ranks():
    # документ 0 первый в обоих списках -> наибольший score
    dense = [0, 1, 2]
    lexical = [0, 2, 1]
    scores = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    best = max(scores, key=scores.get)
    assert best == 0


def test_rrf_combines_both_lists():
    # документ есть только в одном списке, но всё равно получает score
    dense = [5, 6]
    lexical = [7, 8]
    scores = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    assert set(scores) == {5, 6, 7, 8}
    # верх каждого списка равноценен при симметрии
    assert scores[5] == scores[7]


def test_rrf_agreement_beats_single_list():
    # документ, согласованно высокий в обоих, обгоняет лидера лишь одного списка
    dense = [1, 0]
    lexical = [1, 2]
    scores = reciprocal_rank_fusion([dense, lexical], rrf_k=60)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}
