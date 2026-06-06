# Hybrid RAG по русской Википедии (SberQuAD)

Учебный, но **близкий к SOTA** RAG: hybrid retrieval (dense + BM25) →
Reciprocal Rank Fusion → cross-encoder reranking → ответ с цитатами и честным
отказом. Качество retrieval измеряется на golden-set.

```
вопрос
  │
  ├─▶ dense (multilingual-e5, FAISS)  ─┐
  │                                     ├─▶ RRF ─▶ cross-encoder rerank ─▶ top-k
  └─▶ lexical (BM25)                  ─┘                                      │
                                                                              ▼
                          контекст + вопрос ─▶ LLM (или extractive) ─▶ ответ [n] + источники
```

**Документы:** [doc/vision.md](doc/vision.md) · [doc/DATA.md](doc/DATA.md) ·
[doc/tasklist.md](doc/tasklist.md) · [homework/IMPROVEMENTS.md](homework/IMPROVEMENTS.md) ·
**Данные:** [SberQuAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad)

## Требования

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- ~2 ГБ места под модели HuggingFace (качаются при первом запуске)

## Быстрый старт

```bash
# 1. Окружение и зависимости
uv venv
uv sync

# 2. Сборка индекса (ingest → chunk → embed → FAISS + BM25)
#    datasets.json уже в репозитории — интернет нужен только для скачивания моделей
uv run python scripts/build_index.py

# 3. Запуск UI
uv run streamlit run app/main.py
```

Откройте http://localhost:8501

> **LLM-генерация (опционально).** Скопируйте `.env.example` → `.env` и укажите
> `OPENAI_API_KEY`. Поддерживается любой OpenAI-совместимый провайдер через
> `OPENAI_BASE_URL` — например, **бесплатные модели OpenRouter**:
> ```env
> OPENAI_API_KEY=sk-or-...
> OPENAI_BASE_URL=https://openrouter.ai/api/v1
> RAG_LLM_MODEL=openai/gpt-oss-120b:free
> ```
> **Без ключа** проект работает в extractive-режиме — ответ собирается из
> найденных фрагментов. Запуск на чистой машине ключа не требует.

## Данные

Корпус — 2000 абзацев русской Википедии из **SberQuAD** (`data/raw/datasets.json`),
после нарезки — **2000+ чанков**. Подробно: [doc/DATA.md](doc/DATA.md).
Пересборка корпуса: `uv run python scripts/prepare_datasets.py`.

## Demo-вопросы

| Вопрос | Ожидание |
|--------|----------|
| **В каком году истекает договор аренды Байконура?** | «...действует до 2050 года [2]» + источник (doc_id=2) |
| **чем представлены органические остатки?** | ответ про известковые водоросли + источник (doc_id=0, rerank≈0.94) |
| **из чего состоят белки?** | ответ про аминокислоты + источник (doc_id=1575, rerank≈0.86) |
| как приготовить борщ? | **отказ** — в корпусе нет такой темы (все rerank-score < порога) |

Реальные ответы и логи — в разделе [«Логи»](#логи-демо-прогон) ниже.

## Проверка из консоли

```bash
uv run pytest -q                              # тесты
uv run python scripts/check_retrieval.py      # этапы hybrid retrieval
uv run python scripts/check_generator.py      # 3 ответа + 1 отказ
uv run python scripts/evaluate.py             # метрики качества retrieval
```

## Качество retrieval (golden-set)

Оценка на 150 вопросах SberQuAD с известным правильным документом
(`scripts/evaluate.py`, k=10):

<!-- METRICS -->
| Режим | hit@1 | hit@3 | hit@5 | recall@10 | MRR |
|-------|:-----:|:-----:|:-----:|:---------:|:---:|
| hybrid + RRF | 0.95 | 0.96 | 0.97 | 1.00 | 0.962 |
| hybrid + RRF + **rerank** | 0.94 | **0.99** | **0.99** | 1.00 | **0.963** |

(`scripts/evaluate.py --sample 100 --k 10`, отчёт → `data/index/eval_report.json`.)

**Как читать.** Гибрид уже очень силён: вопросы SberQuAD лексически близки своим
контекстам, поэтому dense + BM25 находят правильный документ в 95% случаев на
первом месте, а recall@10 = 100%. **Cross-encoder reranking** на таком почти
насыщенном бенчмарке поднимает hit@3/hit@5 (0.96→0.99, 0.97→0.99) — «подтягивает»
правильный документ ближе к топу — и, что важнее для продукта, даёт
**калиброванную оценку релевантности** (релевантные пары ~0.2–0.95,
нерелевантные < 0.01). Именно она обеспечивает корректный **отказ** на вопросах
вне корпуса (см. negative-кейс «борщ» в логах ниже). Цена — скорость: на CPU
reranking ~5 с/запрос против ~0.2 с у гибрида, поэтому в UI его можно отключить.
<!-- /METRICS -->

Cross-encoder reranking — одно из реализованных улучшений: измеримый прирост по
hit@3/hit@5 и калиброванный score для отказа (см.
[IMPROVEMENTS](homework/IMPROVEMENTS.md)).

## Логи (демо-прогон)

<!-- LOGS -->
Реальный вывод `uv run python scripts/check_generator.py` с LLM-генерацией
(провайдер OpenRouter, модель `openai/gpt-oss-120b:free`):

```text
LLM включена: True

--- Контекст есть: «В каком году истекает договор аренды Байконура?» ---
режим: llm, отказ: False
Ответ: Договор аренды комплекса «Байконур» действует до 2050 года [2].
  [1] doc_id=146, score=0.577 (rerank)
  [2] doc_id=2,   score=0.086 (rerank)   ← из этого фрагмента взят ответ «2050»

--- Контекст есть: «чем представлены органические остатки?» ---
режим: llm, отказ: False
Ответ: Органические остатки в протерозойских отложениях представлены известковыми
       выделениями сине-зелёных водорослей, ходами червей и остатками
       кишечнополостных [1]. Кроме того, встречаются известковые водоросли,
       скопления графито-углистого вещества (разложение Corycium enigmaticum),
       нитевидные водоросли, грибные нити и др. [1].
  [1] doc_id=0, score=0.935 (rerank)

--- Контекст есть: «из чего состоят белки?» ---
режим: llm, отказ: False
Ответ: Белки состоят из α-аминокислот, соединённых в цепочку пептидной связью [1].
  [1] doc_id=1575, score=0.864 (rerank)

--- Negative: «как приготовить борщ и какие нужны ингредиенты?» ---
режим: refused, отказ: True
Ответ: В базе не найдено релевантных фрагментов по этому вопросу.
       Ответить по имеющимся данным невозможно.
  [1] doc_id=889, score=0.005 (rerank)   ← все кандидаты ниже порога → отказ
```

Ответы **строго по контексту** и с **цитатами `[n]`**; на вопрос вне корпуса —
**отказ** (срабатывает до вызова LLM, по порогу релевантности). Без ключа те же
вопросы работают в extractive-режиме (сводка фрагментов с цитатами).
Вывод `uv run python scripts/check_retrieval.py` показывает для каждого источника
вклад этапов: `fusion=… (dense#…, bm25#…), rerank=…`.
<!-- /LOGS -->

## Структура проекта

```
rag-tutorial/
├── app/
│   ├── config.py       # пути, модели, top_k, RRF_K, пороги
│   ├── chunker.py      # token-aware нарезка
│   ├── embedder.py     # плотные эмбеддинги (multilingual-e5)
│   ├── index.py        # FAISS (dense) + BM25 (lexical) -> HybridIndex
│   ├── retriever.py    # hybrid + RRF + rerank
│   ├── reranker.py     # cross-encoder
│   ├── llm.py          # OpenAI-совместимый клиент
│   ├── generator.py    # grounded-ответ + отказ
│   ├── prompts.py      # правила, цитаты, отказ
│   └── main.py         # Streamlit UI
├── scripts/
│   ├── prepare_datasets.py  # SberQuAD -> datasets.json + eval.json
│   ├── ingest.py
│   ├── build_index.py
│   ├── evaluate.py          # recall@k / hit@k / MRR
│   ├── check_retrieval.py
│   └── check_generator.py
├── data/
│   ├── raw/datasets.json    # корпус (коммитится)
│   ├── raw/eval.json        # golden-set (коммитится)
│   ├── processed/           # documents.jsonl, chunks.jsonl (генерируются)
│   └── index/               # faiss.index, bm25.pkl, embeddings.npy (генерируются)
├── tests/
└── doc/
```

## Реализованные улучшения (поверх базового TF-IDF MVP)

1. **Semantic-эмбеддинги** (multilingual-e5) вместо TF-IDF.
2. **Hybrid search**: dense + BM25, объединение через **RRF**.
3. **Cross-encoder reranking** (retrieve-then-rerank).
4. **Векторный индекс FAISS**.
5. **LLM-генерация** с цитатами + **grounded-отказ** (offline-fallback).
6. **Оценка качества** (recall@k / hit@k / MRR) на golden-set.

Подробности и план — [homework/IMPROVEMENTS.md](homework/IMPROVEMENTS.md).

## Ограничения

- Корпус — отобранные 2000 абзацев (масштаб настраивается в `prepare_datasets.py`).
- Reranking на CPU заметно медленнее dense-поиска; в UI его можно отключить.
- Качество LLM-ответа зависит от выбранного провайдера/модели.

## Контакты

Автор: **Патрушев Константин** · Telegram: [@KAPatrushev](https://t.me/KAPatrushev)

По вопросам **сотрудничества** пишите в Telegram [@KAPatrushev](https://t.me/KAPatrushev).
Резюме (ML / Data Science): [Патрушев_резюме_ML.pdf](Патрушев_резюме_ML.pdf).
