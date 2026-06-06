# Данные и назначение репозитория

Документ описывает, **какие данные** использует RAG, **откуда** они, **что**
попадает в индекс и как устроен **golden-set** для оценки.

---

## Назначение репозитория

**Для кого:** студенты и разработчики, изучающие современный RAG.

**Что демонстрирует:**

- полный pipeline: сырые данные → документы → чанки → эмбеддинги + BM25 →
  hybrid retrieval → RRF → cross-encoder reranking → ответ с источниками;
- ответ **только по найденным фрагментам** с цитатами `[n]` и указанием
  `doc_id` / score каждого этапа;
- явный **отказ**, если релевантного контекста нет;
- **измеримое качество** retrieval на golden-set (recall@k, hit@k, MRR);
- Streamlit UI с прозрачными этапами поиска.

---

## Источник данных

| Источник | Где | Комментарий |
|----------|-----|-------------|
| [SberQuAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad) (`kuznetsoffandrey/sberquad`, config `sberquad`) | HuggingFace `datasets` | Русскоязычный QA на абзацах Википедии: `context`, `question`, `answers` |
| Подготовленный корпус | `data/raw/datasets.json` | 2000 уникальных контекстов: `id`, `name`, `text`. **Коммитится** |
| Golden-set | `data/raw/eval.json` | 200 вопросов: `question`, `answer`, `gold_doc_id`. **Коммитится** |
| Скрипт подготовки | `scripts/prepare_datasets.py` | Выгрузка из SberQuAD, дедуп контекстов, сборка golden-set |

**Масштаб:** 2000 документов → **~2000+ чанков** после нарезки (> требования
«1000+ записей или 1000+ чанков»). Размер корпуса настраивается (`N_CONTEXTS`).

**Лицензия:** SberQuAD распространяется на HuggingFace; исходные тексты —
из русской Википедии (CC BY-SA). Уточняйте условия на странице датасета.

**Воспроизводимость:** `datasets.json` и `eval.json` закоммичены, поэтому
для сборки индекса интернет **не нужен** — скачиваются только модели
(эмбеддер/reranker) при первом запуске.

---

## Что индексируем

| Поле / артефакт | Индексируется? | Где используется |
|-----------------|:--------------:|------------------|
| `text` (контекст) | **Да** | чанки → эмбеддинги (FAISS) + BM25 |
| `name` | Нет (метаданные) | UI, источники — подпись документа |
| `doc_id` | Нет (метаданные) | UI, источники, метрики (сопоставление с gold) |
| `eval.json` | Нет | только оценка качества retrieval |

**Pipeline:**

```
datasets.json → documents.jsonl → chunks.jsonl → embeddings.npy + faiss.index + bm25.pkl
```

- **Чанки:** token-aware по предложениям, ~220 токенов, overlap ~40 (`app/chunker.py`).
- **Поиск:** dense (cosine по e5) + BM25 → RRF → cross-encoder rerank (`app/retriever.py`).

---

## Golden-set и метрики

SberQuAD даёт пары **вопрос → правильный контекст**. При подготовке для каждого
вопроса сохраняется `gold_doc_id` — id документа с правильным ответом. Это
позволяет честно мерить retrieval:

- **hit@k / recall@k** — попал ли правильный документ в top-k;
- **MRR** — средний обратный ранг правильного документа.

Запуск: `uv run python scripts/evaluate.py` (сравнивает hybrid+RRF и
hybrid+RRF+reranker). Метрики — в [README](../README.md).

---

## Что не индексируем

| Не индексируется | Причина |
|------------------|---------|
| `eval.json` | Только для оценки, не часть корпуса |
| `data/processed/*.jsonl` | Промежуточные артефакты, генерируются |
| `data/index/*` | Индекс пересобирается `build_index.py` |
| Секреты / API-ключи | Только через `.env` (не в репозитории) |

---

## Как обновить данные

1. (Опционально) поменять `N_CONTEXTS` / `N_EVAL` в `scripts/prepare_datasets.py`.
2. `uv run python scripts/prepare_datasets.py` — пересобрать корпус и golden-set.
3. `uv run python scripts/build_index.py` — пересобрать индекс.
4. `uv run python scripts/evaluate.py` — перепроверить метрики.

---

## Связанные документы

- [00_project_idea.md](00_project_idea.md) — идея и данные
- [vision.md](vision.md) — стек и обоснование SOTA-решений
- [tasklist.md](tasklist.md) — итерационный план
