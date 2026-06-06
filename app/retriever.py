"""Hybrid retrieval: dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank.

Конвейер одного запроса:
  1. dense   — top-N ближайших чанков по эмбеддингам (FAISS, поиск по смыслу);
  2. lexical — top-N по BM25 (поиск по точным словам/терминам);
  3. RRF     — объединяем два ранжирования по позициям (Reciprocal Rank Fusion),
               это устойчивее, чем складывать «сырые» несопоставимые score;
  4. rerank  — cross-encoder пересортировывает top-кандидатов (опционально);
  5. top-k   — финальные источники с прозрачными score каждого этапа.
"""

from __future__ import annotations

import numpy as np

from app.config import (
    CANDIDATES_PER_RETRIEVER,
    RERANK_CANDIDATES,
    RRF_K,
    TOP_K,
    USE_RERANKER,
)
from app.embedder import embed_query
from app.index import HybridIndex, bm25_tokenize
from app.reranker import rerank as rerank_candidates


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    rrf_k: int = RRF_K,
) -> dict[int, float]:
    """RRF: score(d) = Σ_lists 1 / (rrf_k + rank(d)), rank — 1-based позиция.

    Принимает несколько ранжированных списков id (лучший — первый),
    возвращает {id: fused_score}. Не зависит от шкалы исходных score.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return scores


class Retriever:
    def __init__(
        self,
        index: HybridIndex | None = None,
        use_reranker: bool = USE_RERANKER,
    ) -> None:
        self.index = index or HybridIndex.load()
        self.use_reranker = use_reranker

    def _dense_ranking(self, query: str, n: int) -> list[int]:
        qvec = embed_query(query, model_name=self.index.embed_model).reshape(1, -1)
        n = min(n, self.index.faiss_index.ntotal)
        _, idxs = self.index.faiss_index.search(qvec, n)
        return [int(i) for i in idxs[0] if i != -1]

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.index.bm25.get_scores(bm25_tokenize(query))
        n = min(n, len(scores))
        top = np.argsort(scores)[::-1][:n]
        # отбрасываем нулевые BM25 (нет общих термов) — они не информативны
        return [int(i) for i in top if scores[i] > 0]

    def search(
        self,
        query: str,
        k: int = TOP_K,
        candidates: int = CANDIDATES_PER_RETRIEVER,
    ) -> list[dict]:
        if not query.strip():
            return []

        dense = self._dense_ranking(query, candidates)
        lexical = self._bm25_ranking(query, candidates)

        fused = reciprocal_rank_fusion([dense, lexical])
        if not fused:
            return []

        dense_pos = {idx: r for r, idx in enumerate(dense, 1)}
        lex_pos = {idx: r for r, idx in enumerate(lexical, 1)}

        ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)
        pool_ids = ranked_ids[:RERANK_CANDIDATES]

        pool: list[dict] = []
        for idx in pool_ids:
            chunk = self.index.chunks[idx]
            pool.append(
                {
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "name": chunk.get("name", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "fusion_score": float(fused[idx]),
                    "dense_rank": dense_pos.get(idx),
                    "bm25_rank": lex_pos.get(idx),
                }
            )

        if self.use_reranker:
            pool = rerank_candidates(query, pool)
            for item in pool:
                item["score"] = item["rerank_score"]
                item["score_kind"] = "rerank"
        else:
            for item in pool:
                item["score"] = item["fusion_score"]
                item["score_kind"] = "fusion"

        return pool[:k]
