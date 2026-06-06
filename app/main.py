"""Streamlit UI: вопрос → этапы retrieval → ответ с цитатами → источники.

Показывает «внутренности» RAG (это учебный проект): для каждого источника
видны fusion-score (RRF), позиции в dense- и BM25-ранжировании и cross-encoder
rerank-score. Так наглядно, почему фрагмент попал в ответ.
"""

import streamlit as st

from app.config import (
    FAISS_INDEX,
    INDEX_CHUNKS_JSONL,
    TOP_K,
    USE_RERANKER,
    llm_enabled,
)
from app.generator import ask
from app.index import HybridIndex
from app.retriever import Retriever

DEMO_QUESTIONS = [
    "В каком году истекает договор аренды Байконура?",
    "чем представлены органические остатки?",
    "из чего состоят белки?",
    "как приготовить борщ?",  # negative — нет в корпусе
]


def index_exists() -> bool:
    return FAISS_INDEX.exists() and INDEX_CHUNKS_JSONL.exists()


@st.cache_resource
def load_retriever(use_reranker: bool) -> Retriever:
    return Retriever(index=HybridIndex.load(), use_reranker=use_reranker)


def render_source(i: int, src: dict) -> None:
    rr = src.get("rerank_score")
    score_line = f"score={src['score']:.4f} ({src['score_kind']})"
    label = f"[{i}] doc_id={src['doc_id']} · {score_line}"
    with st.expander(label, expanded=(i <= 3)):
        st.markdown(f"**{src['name']}**")
        meta = []
        if src.get("dense_rank"):
            meta.append(f"dense #{src['dense_rank']}")
        if src.get("bm25_rank"):
            meta.append(f"BM25 #{src['bm25_rank']}")
        if src.get("fusion_score") is not None:
            meta.append(f"RRF={src['fusion_score']:.4f}")
        if rr is not None:
            meta.append(f"rerank={rr:.3f}")
        st.caption(" · ".join(meta))
        st.text(src["text"])


def main() -> None:
    st.set_page_config(page_title="Hybrid RAG", layout="wide")
    st.title("Hybrid RAG — поиск по русской Википедии (SberQuAD)")
    st.caption(
        "dense (multilingual-e5) + BM25 → Reciprocal Rank Fusion → cross-encoder rerank "
        "→ ответ с цитатами"
    )

    if not index_exists():
        st.error(
            "Индекс не собран. Сначала выполните:\n\n"
            "`uv run python scripts/build_index.py`"
        )
        st.stop()

    with st.sidebar:
        st.header("Настройки")
        k = st.slider("top-k источников", 1, 10, TOP_K)
        use_reranker = st.toggle("cross-encoder reranker", value=USE_RERANKER)
        mode = "LLM (API)" if llm_enabled() else "extractive (без ключа)"
        st.caption(f"Режим генерации: **{mode}**")
        st.divider()
        st.subheader("Demo-вопросы")
        for q in DEMO_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state["question"] = q

    question = st.text_input("Ваш вопрос", key="question")

    if st.button("Спросить", type="primary"):
        if not question.strip():
            st.warning("Введите вопрос.")
            st.stop()

        retriever = load_retriever(use_reranker)
        with st.spinner("Поиск и генерация…"):
            result = ask(question.strip(), k=k, retriever=retriever)

        st.subheader("Ответ")
        if result["refused"]:
            st.warning(result["answer"])
        else:
            st.markdown(result["answer"])
            st.caption(f"Режим: {result['mode']}")

        st.subheader(f"Источники (top-{k})")
        if not result["sources"]:
            st.info("Источники не найдены.")
        for i, src in enumerate(result["sources"], 1):
            render_source(i, src)


if __name__ == "__main__":
    main()
