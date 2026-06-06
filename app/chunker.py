"""Chunking: documents.jsonl → chunks.jsonl.

Token-aware рекурсивная нарезка: текст → предложения → жадная упаковка в чанки
с целевым лимитом токенов и перекрытием (overlap). Перекрытие сохраняет контекст
на границах чанков, что улучшает retrieval. Счёт токенов — детерминированный
лёгкий токенайзер (без загрузки нейромодели), поэтому chunking быстрый и
тестируется офлайн.
"""

import json
import re
from pathlib import Path

from app.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNKS_JSONL, DOCUMENTS_JSONL

# Слова, числа и одиночная пунктуация — близко к тому, как текст видит subword-токенайзер.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# Конец предложения: . ! ? … с последующим пробелом, либо перенос строки.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def count_tokens(text: str) -> int:
    """Приблизительное число токенов (детерминированно, без модели)."""
    return len(_TOKEN_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Грубая, но устойчивая разбивка на предложения."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def _split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Предложение длиннее лимита — режем по словам."""
    words = sentence.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= max_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Нарезка текста на чанки с целевым размером в токенах и overlap.

    Жадно упаковываем предложения в чанк, пока не превысим max_tokens.
    Перекрытие реализуется переносом хвостовых предложений предыдущего чанка
    в начало следующего (по границам предложений, без разрыва слов).
    """
    if not text.strip():
        return []

    # Готовим плоский список «единиц» — предложений, длинные дробим по словам.
    units: list[str] = []
    for sent in split_sentences(text):
        if count_tokens(sent) <= max_tokens:
            units.append(sent)
        else:
            units.extend(_split_long_sentence(sent, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> list[str]:
        """Закрывает текущий чанк, возвращает хвост для overlap следующего.

        Overlap берём по словам и ограничиваем overlap_tokens, чтобы перенос
        не раздувал следующий чанк (даже если последнее предложение длинное).
        """
        if not current:
            return []
        chunk = " ".join(current)
        chunks.append(chunk)
        if overlap_tokens <= 0:
            return []
        words = chunk.split()
        tail_words = words[-overlap_tokens:]
        return [" ".join(tail_words)] if tail_words else []

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            tail = flush()
            current = list(tail)
            current_tokens = sum(count_tokens(u) for u in current)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """Один документ → список чанков с метаданными."""
    chunks = []
    for i, text in enumerate(chunk_text(doc["text"])):
        chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_{i}",
                "doc_id": doc["doc_id"],
                "name": doc["name"],
                "text": text,
            }
        )
    return chunks


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# Обратная совместимость с прежним именем (используется в retriever/тестах).
load_documents = load_jsonl


def write_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run(
    input_path: Path = DOCUMENTS_JSONL,
    output_path: Path = CHUNKS_JSONL,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден файл: {input_path}")

    documents = load_jsonl(input_path)
    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    write_chunks(all_chunks, output_path)
    return len(all_chunks)


def main() -> None:
    count = run()
    print(f"Записано {count} чанков -> {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
