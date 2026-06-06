"""Подготовка корпуса из HuggingFace SberQuAD → data/raw/datasets.json (+ eval.json).

SberQuAD (kuznetsoffandrey/sberquad) — русскоязычный QA-датасет на абзацах
Википедии: поля context (абзац), question (вопрос), answers (эталонный ответ).
Это идеально для учебного RAG: контексты дают корпус, а пары вопрос→контекст
дают честный golden-set для метрик retrieval (recall@k, MRR, hit-rate).

Что делает скрипт (детерминированно, без рандома — воспроизводимо):
  1. берёт train-сплит SberQuAD;
  2. собирает первые N уникальных контекстов → корпус datasets.json
     (поля id, name, text — тот же контракт, что у остального пайплайна);
  3. строит eval.json: вопросы, чей контекст попал в корпус, с gold_doc_id
     и эталонным ответом.

Артефакты коммитятся в репозиторий, поэтому проверяющему НЕ нужен интернет:
build_index.py работает офлайн на готовом datasets.json.

Запуск (только при пересборке корпуса):
    uv run python scripts/prepare_datasets.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EVAL_SET, RAW_DATASETS

HF_DATASET = "kuznetsoffandrey/sberquad"
HF_CONFIG = "sberquad"
SPLIT = "train"

N_CONTEXTS = 2000   # уникальных контекстов в корпусе (даёт 1000+ записей и чанков)
N_EVAL = 200        # размер golden-set для оценки retrieval


# В SberQuAD поле title — служебная метка ("SberChallenge"), не настоящий
# заголовок статьи. Поэтому имя документа собираем из начала контекста.
_GENERIC_TITLES = {"", "sberchallenge"}


def short_name(title: str, context: str) -> str:
    """Имя документа: заголовок статьи, иначе — первое предложение контекста."""
    title = (title or "").strip()
    if title.lower() not in _GENERIC_TITLES:
        return title
    head = context.strip().replace("\n", " ")
    # первое предложение, но не длиннее 80 символов
    for sep in (". ", "! ", "? "):
        if sep in head[:90]:
            head = head.split(sep, 1)[0]
            break
    return (head[:80] + "…") if len(head) > 80 else head


def main() -> None:
    from datasets import load_dataset

    print(f"Загрузка {HF_DATASET} [{SPLIT}] …")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=SPLIT)

    # 1. Уникальные контексты → корпус (сохраняем порядок появления).
    context_to_id: dict[str, int] = {}
    datasets: list[dict] = []
    for row in ds:
        ctx = row["context"].strip()
        if ctx in context_to_id:
            continue
        doc_id = len(datasets)
        context_to_id[ctx] = doc_id
        datasets.append(
            {"id": doc_id, "name": short_name(row.get("title", ""), ctx), "text": ctx}
        )
        if len(datasets) >= N_CONTEXTS:
            break

    selected_contexts = set(context_to_id)

    # 2. Golden-set: вопросы, чей контекст попал в корпус.
    eval_items: list[dict] = []
    for row in ds:
        ctx = row["context"].strip()
        if ctx not in selected_contexts:
            continue
        answers = row.get("answers", {}).get("text", [])
        eval_items.append(
            {
                "question": row["question"].strip(),
                "answer": answers[0] if answers else "",
                "gold_doc_id": context_to_id[ctx],
            }
        )
        if len(eval_items) >= N_EVAL:
            break

    RAW_DATASETS.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATASETS.write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    EVAL_SET.write_text(
        json.dumps({"eval": eval_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Корпус:    {len(datasets)} документов -> {RAW_DATASETS}")
    print(f"Golden-set: {len(eval_items)} вопросов  -> {EVAL_SET}")


if __name__ == "__main__":
    main()
