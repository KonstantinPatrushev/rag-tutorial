"""Ingestion: datasets.json → documents.jsonl.

Нормализует сырой корпус (datasets.json) в построчный documents.jsonl
с метаданными (doc_id, name, source_file). Это первый шаг офлайн-пайплайна.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DOCUMENTS_JSONL, RAW_DATASETS


def clean_text(text: str) -> str:
    """Нормализация пробелов и переносов строк."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_datasets(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["datasets"]


def ingest_item(item: dict, source_file: str) -> dict:
    return {
        "doc_id": str(item["id"]),
        "name": item["name"].strip(),
        "text": clean_text(item["text"]),
        "source_file": source_file,
    }


def write_documents(documents: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def run(input_path: Path = RAW_DATASETS, output_path: Path = DOCUMENTS_JSONL) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    source_file = str(input_path.relative_to(ROOT))
    datasets = load_datasets(input_path)
    documents = [ingest_item(item, source_file) for item in datasets]
    write_documents(documents, output_path)
    return len(documents)


def main() -> None:
    count = run()
    print(f"Записано {count} документов -> {DOCUMENTS_JSONL}")


if __name__ == "__main__":
    main()
