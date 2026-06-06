# Vision — техническое видение проекта

> Отправная точка для разработки.
> Идея продукта: [00_project_idea.md](00_project_idea.md)

**Учебный, но современный (SOTA) RAG** по русскоязычному корпусу Википедии
(датасет SberQuAD): hybrid retrieval + reranking + LLM-генерация с цитатами и
честным отказом. Локально, прозрачно, с измеримым качеством.

---

## 1. Технологии

| Слой | Выбор | Комментарий |
|------|-------|-------------|
| Язык | **Python 3.10+** | — |
| Окружение | **uv** + `.venv` | `uv sync`; локально, не коммитим |
| UI | **Streamlit** | Один entry point; видны этапы retrieval, score, источники |
| Данные | **HuggingFace SberQuAD** → `data/raw/datasets.json` | Абзацы рус. Википедии + golden-set вопрос→контекст |
| Чанкинг | **Token-aware рекурсивный** (`app/chunker.py`) | по предложениям, лимит в токенах + overlap |
| Эмбеддинги | **sentence-transformers** `multilingual-e5-small` | плотный semantic-поиск, мультиязычный |
| Lexical-поиск | **BM25** (`rank-bm25`) | точные слова, термины, имена, числа |
| Индекс | **FAISS** (dense, IndexFlatIP) + **BM25** + `chunks.jsonl` | локальные файлы в `data/index/` |
| Fusion | **Reciprocal Rank Fusion (RRF)** | объединяет dense и lexical по рангам |
| Reranking | **cross-encoder** `DiTy/cross-encoder-russian-msmarco` | retrieve-then-rerank, опционально |
| LLM | **OpenAI-совместимый API** + extractive-fallback | grounded-ответ с цитатами `[n]`; без ключа — офлайн-режим |
| Оценка | **recall@k, hit@k, MRR** на golden-set | `scripts/evaluate.py` |
| Тесты | **pytest** | chunking, RRF, hybrid retrieval, generation, метрики |

### Почему так (SOTA-обоснование)

- **Hybrid (dense + BM25)** сильнее любого одиночного ретривера: dense ловит
  смысл и перефразировки, BM25 — редкие термины/имена/числа.
- **RRF** объединяет ранжирования без подбора весов и не зависит от шкалы score.
- **Cross-encoder reranking** — классический паттерн «retrieve-then-rerank»:
  дёшево достаём кандидатов, дорого и точно пересортировываем только топ.
- **Grounded-генерация с цитатами** и **порогом отказа** — ответ привязан к
  источникам, выдумывание исключается.
- **Offline-fallback** гарантирует запуск на чистой машине без секретов.

### Воспроизводимость

- `datasets.json` и `eval.json` **коммитятся** → проверяющему не нужен интернет
  для сборки индекса (модели качаются с HuggingFace при первом запуске).
- Артефакты индекса (`data/index/`) и промежуточные `*.jsonl` — **не коммитятся**,
  пересобираются командой `build_index.py`.

### Окружение

```bash
uv venv
uv sync
```

---

## 2. Границы

- Корпус — отобранные абзацы SberQuAD (по умолчанию 2000 → ~2000+ чанков).
- LLM-провайдер не зашит: любой OpenAI-совместимый endpoint через `.env`.
- Не строим распределённую/прод-инфраструктуру: один процесс, локальные файлы.
