"""Сборка и загрузка гибридного индекса: dense (FAISS) + lexical (BM25).

Два взаимодополняющих представления одного корпуса чанков:
  - dense  — FAISS IndexFlatIP по нормированным эмбеддингам (поиск по смыслу);
  - lexical — BM25 по токенизированным текстам (поиск по точным словам/терминам).

Гибрид сильнее любого из них по отдельности: dense ловит синонимы и
перефразировки, BM25 — редкие термины, имена, числа. Объединяются в retriever.py.
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunker import load_jsonl
from app.config import (
    BM25_PKL,
    CHUNKS_JSONL,
    DATA_INDEX,
    EMBED_MODEL,
    EMBEDDINGS_NPY,
    FAISS_INDEX,
    INDEX_CHUNKS_JSONL,
    META_JSON,
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    """Токенизация для BM25: нижний регистр + слова/числа (RU/EN)."""
    return _WORD_RE.findall(text.lower())


def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    """IndexFlatIP: внутреннее произведение = cosine для нормированных векторов."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_bm25(texts: list[str]) -> BM25Okapi:
    return BM25Okapi([bm25_tokenize(t) for t in texts])


def save_index(
    embeddings: np.ndarray,
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    chunks_path: Path = CHUNKS_JSONL,
    embed_model: str = EMBED_MODEL,
) -> int:
    DATA_INDEX.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, embeddings)
    faiss.write_index(faiss_index, str(FAISS_INDEX))
    with BM25_PKL.open("wb") as f:
        pickle.dump(bm25, f)
    shutil.copy2(chunks_path, INDEX_CHUNKS_JSONL)
    META_JSON.write_text(
        json.dumps(
            {
                "embed_model": embed_model,
                "n_chunks": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings.shape[0]


@dataclass
class HybridIndex:
    """Загруженный гибридный индекс: всё, что нужно retriever'у."""

    chunks: list[dict]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    embed_model: str

    @classmethod
    def load(
        cls,
        faiss_path: Path = FAISS_INDEX,
        bm25_path: Path = BM25_PKL,
        chunks_path: Path = INDEX_CHUNKS_JSONL,
        meta_path: Path = META_JSON,
    ) -> "HybridIndex":
        missing = [p for p in (faiss_path, bm25_path, chunks_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Индекс не собран ("
                + ", ".join(str(p.name) for p in missing)
                + "). Запустите: uv run python scripts/build_index.py"
            )
        chunks = load_jsonl(chunks_path)
        faiss_index = faiss.read_index(str(faiss_path))
        with bm25_path.open("rb") as f:
            bm25 = pickle.load(f)
        embed_model = EMBED_MODEL
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            embed_model = meta.get("embed_model", EMBED_MODEL)

        if faiss_index.ntotal != len(chunks):
            raise ValueError("FAISS-индекс и chunks.jsonl рассинхронизированы")
        return cls(chunks=chunks, faiss_index=faiss_index, bm25=bm25, embed_model=embed_model)
