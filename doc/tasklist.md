# Tasklist — итерационный план разработки

Опирается на: @vision.md · @conventions.md · @00_project_idea.md · @workflow.md

**Правило:** одна итерация = один проверяемый результат.

---

## 📊 Прогресс

| Итерация | Название | Статус | Проверка |
|:--------:|----------|:------:|----------|
| 0 | Каркас проекта | ✅ | `uv sync` без ошибок |
| 1 | Данные из HuggingFace (SberQuAD) | ✅ | `datasets.json` 2000 + `eval.json` 200 |
| 2 | Ingestion | ✅ | `documents.jsonl` создан |
| 3 | Token-aware chunking | ✅ | `chunks.jsonl`, тест chunking |
| 4 | Эмбеддинги + гибридный индекс (FAISS + BM25) | ✅ | файлы в `data/index/` |
| 5 | Hybrid retrieval + RRF | ✅ | top-k со score этапов |
| 6 | Cross-encoder reranking | ✅ | rerank_score в выдаче |
| 7 | Generation (LLM + extractive fallback) + отказ | ✅ | ответ с цитатами / отказ |
| 8 | Streamlit UI | ✅ | вопрос → этапы → ответ → источники |
| 9 | Оценка качества (recall@k / MRR) | ✅ | `evaluate.py`, отчёт по метрикам |
| 10 | Тесты и README | ✅ | `pytest` green, README воспроизводим |
| 11 | Документ о данных | ✅ | `doc/DATA.md` |

**Готовность:** 12 / 12

---

## Итерация 0 — Каркас

- [x] `pyproject.toml`: streamlit, sentence-transformers, faiss-cpu, rank-bm25, datasets, openai, python-dotenv, scikit-learn, pytest
- [x] `.gitignore`: `.venv/`, `data/index/`, `data/processed/*.jsonl`, `.env`
- [x] `app/config.py` — пути, модели, top_k, RRF_K, пороги

## Итерация 1 — Данные (HuggingFace SberQuAD)

- [x] `scripts/prepare_datasets.py` — SberQuAD → `datasets.json` (2000 контекстов) + `eval.json` (200 вопросов с gold_doc_id)
- [x] Артефакты коммитятся (воспроизводимость без интернета)

**Проверка:** `uv run python -c "import json;print(len(json.load(open('data/raw/datasets.json'))['datasets']))"`

## Итерация 2 — Ingestion

- [x] `scripts/ingest.py` — `datasets.json` → `documents.jsonl`, очистка, метаданные

## Итерация 3 — Chunking

- [x] `app/chunker.py` — token-aware рекурсивная нарезка по предложениям + overlap
- [x] `tests/test_chunking.py`

## Итерация 4 — Эмбеддинги + индекс

- [x] `app/embedder.py` — multilingual-e5, префиксы query:/passage:, L2-норма
- [x] `app/index.py` — FAISS IndexFlatIP (dense) + BM25 (lexical), `HybridIndex`
- [x] `scripts/build_index.py` — ingest → chunk → embed → save

## Итерация 5 — Hybrid retrieval + RRF

- [x] `app/retriever.py` — dense + BM25 → Reciprocal Rank Fusion → top-k
- [x] `tests/test_fusion.py`, `tests/test_retrieval.py`

## Итерация 6 — Reranking

- [x] `app/reranker.py` — cross-encoder (`DiTy/cross-encoder-russian-msmarco`), retrieve-then-rerank
- [x] Флаг включения в конфиге/UI; откат на RRF при недоступности модели

## Итерация 7 — Generation

- [x] `app/prompts.py` — grounded-правила, формат цитат `[n]`, тексты отказа
- [x] `app/llm.py` — OpenAI-совместимый клиент
- [x] `app/generator.py` — ответ с цитатами / extractive fallback / отказ по порогу
- [x] `tests/test_generator.py`

## Итерация 8 — Streamlit UI

- [x] `app/main.py` — вопрос, настройки (top-k, reranker), этапы retrieval, источники

## Итерация 9 — Оценка качества

- [x] `scripts/evaluate.py` — recall@k / hit@k / MRR на golden-set; сравнение RRF vs rerank
- [x] `tests/test_eval.py`

## Итерация 10 — Тесты и README

- [x] 5+ тестов green (chunking, fusion, retrieval, generation, eval)
- [x] README: установка, сборка индекса, запуск, demo-вопросы, метрики, логи

## Итерация 11 — Документ о данных

- [x] `doc/DATA.md` — источник, что индексируем, golden-set, лицензия
