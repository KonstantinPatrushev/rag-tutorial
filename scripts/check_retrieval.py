"""Проверка hybrid retrieval — наглядный вывод этапов dense/BM25/RRF/rerank."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.index import HybridIndex
from app.retriever import Retriever


def print_hit(i: int, hit: dict) -> None:
    preview = hit["text"][:110].replace("\n", " ")
    rr = hit.get("rerank_score")
    rr_str = f", rerank={rr:.3f}" if rr is not None else ""
    print(
        f"  [{i}] doc_id={hit['doc_id']} fusion={hit['fusion_score']:.4f}"
        f" (dense#{hit['dense_rank']}, bm25#{hit['bm25_rank']}){rr_str}"
    )
    print(f"      {preview}…")


def main() -> None:
    print("=== Проверка hybrid retrieval (dense + BM25 + RRF + rerank) ===\n")
    index = HybridIndex.load()
    r = Retriever(index=index)
    print(f"OK: индекс загружен ({index.faiss_index.ntotal} чанков, модель {index.embed_model})\n")

    queries = [
        "что арендует Россия в Казахстане?",
        "как приготовить борщ?",  # вне корпуса -> низкие score
    ]
    for query in queries:
        print(f"Запрос: «{query}»")
        for i, hit in enumerate(r.search(query, k=3), 1):
            print_hit(i, hit)
        print()


if __name__ == "__main__":
    main()
