"""Сборка гибридного индекса: ingest → chunk → embed → FAISS + BM25 → save.

Офлайн-шаг: всё тяжёлое (эмбеддинги, индексы) считается здесь и кладётся в
data/index/. Streamlit и retriever потом только загружают готовый индекс.

Запуск:
    uv run python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.chunker import load_jsonl, run as chunk_run
from app.config import CHUNKS_JSONL, DATA_INDEX, EMBED_MODEL
from app.embedder import embed_passages
from app.index import build_bm25, build_faiss, save_index
from ingest import run as ingest_run


def run() -> int:
    t0 = time.time()
    doc_count = ingest_run()
    chunk_count = chunk_run()
    chunks = load_jsonl(CHUNKS_JSONL)
    texts = [c["text"] for c in chunks]

    if not texts:
        raise ValueError("Нет чанков для индексации")

    print(f"Документов: {doc_count}, чанков: {chunk_count}")
    print(f"Эмбеддинги ({EMBED_MODEL}) …")
    embeddings = embed_passages(texts, show_progress=True)

    print("FAISS (dense) + BM25 (lexical) …")
    faiss_index = build_faiss(embeddings)
    bm25 = build_bm25(texts)

    n = save_index(embeddings, faiss_index, bm25)
    print(
        f"Индекс сохранён -> {DATA_INDEX} "
        f"(чанков: {n}, dim: {embeddings.shape[1]}, {time.time() - t0:.1f}s)"
    )
    return chunk_count


def main() -> None:
    run()


if __name__ == "__main__":
    main()
