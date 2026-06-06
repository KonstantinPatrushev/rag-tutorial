# Conventions — правила для разработки

Источник истины по стеку и pipeline: @vision.md
Идея продукта: @00_project_idea.md

---

## Архитектура кода

- **Offline / online:** всё тяжёлое (эмбеддинги, индексы) считает
  `scripts/build_index.py`; Streamlit и `Retriever` только **загружают** готовый
  индекс из `data/index/`.
- **Один модуль — одна роль:**
  - `app/chunker.py` — token-aware нарезка;
  - `app/embedder.py` — плотные эмбеддинги (sentence-transformers);
  - `app/index.py` — сборка/загрузка FAISS + BM25 (`HybridIndex`);
  - `app/retriever.py` — hybrid + RRF + rerank;
  - `app/reranker.py` — cross-encoder;
  - `app/llm.py` — клиент OpenAI-совместимого API;
  - `app/generator.py` — grounded-ответ + отказ;
  - `app/prompts.py` — system-правила, формат цитат, тексты отказа;
  - `app/main.py` — Streamlit UI.
- **Конфиг в одном месте** (`app/config.py`): пути, модели, top_k, RRF_K, пороги.
  Значения переопределяются через переменные окружения (модели, ключи).
- Пути — через `pathlib.Path`.

## RAG-поведение (обязательно)

- Каждый ответ содержит **источники**: `doc_id`, фрагмент, score (с разбивкой по
  этапам: dense-ранг, BM25-ранг, RRF, rerank).
- Ответ строится **только из найденных фрагментов**; при LLM-режиме — с цитатами
  `[n]`; если нет фрагментов выше порога — **явный отказ** без выдумок.
- Retrieval всегда **hybrid** (dense + BM25), объединение через **RRF**;
  reranking — опционально (флаг в конфиге/UI).
- Без ключа LLM проект обязан работать (extractive-режим).

## Качество и оценка

- Любое изменение retrieval проверяется на golden-set (`scripts/evaluate.py`):
  recall@k / hit@k / MRR. Улучшение должно быть **измеримым**.

## Стиль

- KISS: простые функции, dataclass/dict, без лишних абстракций.
- Код и идентификаторы — на английском; docstring и пользовательские сообщения —
  на русском (учебность).
- Обработка ошибок — там, где без неё ломается UX (нет индекса, пустой вопрос,
  недоступна модель reranker → откат на RRF).

## Зависимости и окружение

- **Python 3.10+**, менеджер — **uv**; venv в `.venv/` (не коммитить).
- Зависимости — в `pyproject.toml`; установка `uv sync`.
- Секреты — только через `.env` (не коммитить); `.env.example` — шаблон.
- `data/index/` и промежуточные `*.jsonl` — не коммитить; `datasets.json` и
  `eval.json` — коммитим (воспроизводимость без интернета).
