# Submission

## Ссылка на репозиторий с заданием

- Repo URL: `https://github.com/KonstantinPatrushev/rag-tutorial`
- Ветка: `patrushev_konstantin_hw`

## Автор

- ФИО / ник: Патрушев Константин

## Комментарий

Реализован **современный (SOTA) RAG** на собственных данных, повторяющий и
существенно расширяющий pipeline репозитория-образца.

**Данные.** HuggingFace **SberQuAD** — русскоязычный QA на абзацах Википедии.
Корпус: **2000 контекстов → 2000+ чанков** (критерий «1000+» выполнен).
Встроенные пары вопрос→контекст дают **golden-set** (`eval.json`, 200 вопросов)
для честной оценки retrieval. `datasets.json` и `eval.json` закоммичены —
сборка индекса воспроизводима без интернета.

**Pipeline.** ingest → token-aware chunking → dense-эмбеддинги
(`multilingual-e5`, FAISS) + BM25 → **Reciprocal Rank Fusion** →
**cross-encoder reranking** (`DiTy/cross-encoder-russian-msmarco`) → генерация ответа с цитатами
`[n]` через OpenAI-совместимую LLM (с extractive-fallback без ключа) и
**grounded-отказом** по порогу релевантности. UI — Streamlit с прозрачными
этапами поиска.

**Что улучшено относительно базового MVP (TF-IDF + demo-ответ):** semantic-поиск,
hybrid + RRF, reranking, FAISS, LLM-генерация с цитатами, оценка качества.
Подробно — `homework/IMPROVEMENTS.md`.

**Проверки.** `pytest` — 20+ тестов green (chunking, RRF, hybrid retrieval,
generation/отказ, метрики). 3 demo-ответа + 1 negative-отказ — в README с
логами. Метрики retrieval (recall@k / hit@k / MRR, сравнение с reranking и без) —
в README, отчёт в `data/index/eval_report.json`.

**Документация.** Пять документов планирования (`doc/00_project_idea.md`,
`vision.md`, `conventions.md`, `tasklist.md`, `workflow.md`) согласованы с кодом,
плюс документ о данных `doc/DATA.md`.

## Запуск

```bash
uv sync
uv run python scripts/build_index.py
uv run streamlit run app/main.py
```
