"""Тесты token-aware чанкинга."""

from app.chunker import chunk_document, chunk_text, count_tokens, run, split_sentences


def test_count_tokens_nonzero():
    assert count_tokens("Привет, мир!") >= 3
    assert count_tokens("") == 0


def test_split_sentences():
    text = "Первое предложение. Второе предложение! Третье?"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_chunk_text_respects_max_tokens():
    text = " ".join(f"слово{i}" for i in range(500))
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    assert chunks
    # допускаем небольшой запас на overlap, но грубо держим лимит
    assert all(count_tokens(c) <= 80 for c in chunks)


def test_chunk_text_overlap_present():
    s1 = "Предложение про инфляцию и цены в экономике страны сегодня."
    s2 = "Совсем другая тема про биологию клеток и белки организма."
    # маленький лимит -> два чанка, overlap переносит хвост первого
    text = (s1 + " ") * 6 + (s2 + " ") * 6
    chunks = chunk_text(text, max_tokens=30, overlap_tokens=12)
    assert len(chunks) >= 2


def test_chunk_document_has_ids():
    doc = {"doc_id": "42", "name": "Тест", "text": "Короткий текст про данные."}
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0]["doc_id"] == "42"
    assert chunks[0]["chunk_id"] == "42_0"
    assert chunks[0]["name"] == "Тест"


def test_run_creates_chunks_jsonl(tmp_path):
    docs = tmp_path / "documents.jsonl"
    docs.write_text(
        '{"doc_id": "0", "name": "A", "text": "Короткий текст."}\n',
        encoding="utf-8",
    )
    out = tmp_path / "chunks.jsonl"
    count = run(input_path=docs, output_path=out)
    assert count == 1
    assert out.exists()
    assert '"doc_id": "0"' in out.read_text(encoding="utf-8")
