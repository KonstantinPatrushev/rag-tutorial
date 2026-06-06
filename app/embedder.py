"""Плотные (semantic) эмбеддинги через sentence-transformers.

Модель multilingual-e5 обучена с инструктивными префиксами:
  - "query: ..."   для запросов;
  - "passage: ..." для документов/чанков.
Их обязательно проставлять, иначе качество падает. Векторы L2-нормируем,
тогда внутреннее произведение (FAISS IndexFlatIP) равно косинусной близости.

Модель грузится лениво и кэшируется на процесс — первый вызов скачивает
веса с HuggingFace (документировано в README).
"""

from __future__ import annotations

import numpy as np

from app.config import EMBED_BATCH_SIZE, EMBED_MODEL

_MODEL_CACHE: dict[str, object] = {}


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def get_model(model_name: str = EMBED_MODEL):
    """Ленивая загрузка модели с кэшем по имени."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _prefix(texts: list[str], kind: str, model_name: str) -> list[str]:
    """Проставляет e5-префиксы query:/passage: там, где это нужно."""
    if not _is_e5(model_name):
        return texts
    tag = "query: " if kind == "query" else "passage: "
    return [tag + t for t in texts]


def embed_passages(
    texts: list[str],
    model_name: str = EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    """Эмбеддинги документов/чанков. Возвращает float32 [N, dim], L2-нормированные."""
    model = get_model(model_name)
    vecs = model.encode(
        _prefix(texts, "passage", model_name),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vecs.astype("float32")


def embed_query(query: str, model_name: str = EMBED_MODEL) -> np.ndarray:
    """Эмбеддинг одного запроса. Возвращает float32 [dim], L2-нормированный."""
    model = get_model(model_name)
    vec = model.encode(
        _prefix([query], "query", model_name),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.astype("float32")
