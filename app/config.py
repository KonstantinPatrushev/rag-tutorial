"""Единый конфиг проекта: пути, модели, параметры retrieval/generation.

Все параметры пайплайна собраны здесь (KISS): чтобы поменять модель,
top_k или порог отказа — правится один файл. Значения можно переопределять
через переменные окружения (см. helpers ниже) для гибкости без правки кода.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Пути к данным и артефактам ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INDEX = ROOT / "data" / "index"

RAW_DATASETS = DATA_RAW / "datasets.json"        # корпус: [{id, name, text}]
EVAL_SET = DATA_RAW / "eval.json"                # golden-set: [{question, answer, gold_doc_id}]

DOCUMENTS_JSONL = DATA_PROCESSED / "documents.jsonl"
CHUNKS_JSONL = DATA_PROCESSED / "chunks.jsonl"

# Артефакты индекса (генерируются build_index.py, не коммитятся)
INDEX_CHUNKS_JSONL = DATA_INDEX / "chunks.jsonl"
EMBEDDINGS_NPY = DATA_INDEX / "embeddings.npy"
FAISS_INDEX = DATA_INDEX / "faiss.index"
BM25_PKL = DATA_INDEX / "bm25.pkl"
META_JSON = DATA_INDEX / "meta.json"             # имена моделей/параметры сборки

# --- Чанкинг (token-aware) ---
CHUNK_MAX_TOKENS = 220        # целевой размер чанка в токенах эмбеддера
CHUNK_OVERLAP_TOKENS = 40     # перекрытие между соседними чанками

# --- Модели ---
# Плотный (semantic) эмбеддер. multilingual-e5 требует префиксы query:/passage:.
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH", "64"))

# Cross-encoder reranker (опционально, можно отключить флагом ниже).
# Русскоязычная модель: даёт чистую шкалу релевантности [0,1] на русском
# (релевантные пары ~0.2–0.95, нерелевантные <0.01), что важно и для отказа.
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "DiTy/cross-encoder-russian-msmarco")
USE_RERANKER = os.getenv("RAG_USE_RERANKER", "1") not in ("0", "false", "False")

# --- Retrieval ---
TOP_K = int(os.getenv("RAG_TOP_K", "5"))          # сколько источников показать/передать в LLM
CANDIDATES_PER_RETRIEVER = int(os.getenv("RAG_CANDIDATES", "30"))  # глубина каждого ретривера
RRF_K = int(os.getenv("RAG_RRF_K", "60"))         # сглаживающая константа Reciprocal Rank Fusion
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))  # сколько кандидатов реранкить

# Порог релевантности после reranking/фьюжна: ниже — считаем, что контекста нет.
# Для cross-encoder (логиты) и для RRF-score пороги разные — храним оба.
MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.1"))
MIN_FUSION_SCORE = float(os.getenv("RAG_MIN_FUSION_SCORE", "0.01"))

# --- LLM (генерация) ---
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL")        # None -> api.openai.com
LLM_API_KEY = os.getenv("OPENAI_API_KEY")          # None -> offline extractive fallback
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "600"))


def llm_enabled() -> bool:
    """Есть ли ключ для внешней LLM. Если нет — работает extractive demo-режим."""
    return bool(LLM_API_KEY)
