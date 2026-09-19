"""
Streamlit chat UI for the RAG pipeline.

Thin wrapper around src.generation.Generator. All retrieval and generation
logic lives in the library; this file only handles presentation and
session state.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from generation.generator import Generator, GeneratedAnswer
from retrieval.retriever import Retriever

# ─── Config ────────────────────────────────────────────────────────── #

INDEX_FILE = "data/processed/indexes/books_v2.faiss"
METADATA_FILE = "data/processed/indexes/books_v2_meta.parquet"
EMBEDDING_MODEL_KEY = "bge-small"

GENERATION_MODELS = {
    "gemini-flash (fast)": "gemini-flash",
    "gemini-pro (better)": "gemini-pro",
}

DEFAULT_K = 5
DEFAULT_MODEL_LABEL = "gemini-flash (fast)"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Page setup ────────────────────────────────────────────────────── #

st.set_page_config(
    page_title="ML Books RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ML Books RAG")
st.caption(
    "Ask questions about *Pattern Recognition and Machine Learning* "
    "and *Deep Learning*. Answers are grounded in the source books."
)


# ─── Cached resources ──────────────────────────────────────────────── #
# @st.cache_resource ensures the model + index are loaded once per
# Streamlit session, not once per interaction.


@st.cache_resource(show_spinner="Loading retriever and generator…")
def load_generator(model_key: str) -> Generator:
    """Build a Generator with the given generation model."""
    retriever = Retriever(
        index_path=INDEX_FILE,
        metadata_path=METADATA_FILE,
        model_key=EMBEDDING_MODEL_KEY,
    )
    return Generator(
        retriever=retriever,
        model_key=model_key,
        top_k=DEFAULT_K,
    )


# ─── Session state ─────────────────────────────────────────────────── #

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": ..., "content": ..., "extra": ...}


# ─── Sidebar ───────────────────────────────────────────────────────── #

with st.sidebar:
    st.header("Settings")

    model_label = st.selectbox(
        "Generation model",
        options=list(GENERATION_MODELS.keys()),
        index=list(GENERATION_MODELS.keys()).index(DEFAULT_MODEL_LABEL),
    )
    model_key = GENERATION_MODELS[model_label]

    k = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=DEFAULT_K,
    )

    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Sources are cited from *Pattern Recognition and Machine Learning* "
        "(Bishop) and *Deep Learning* (Goodfellow et al.)."
    )


# ─── Load generator (cached) ───────────────────────────────────────── #

try:
    generator = load_generator(model_key)
    # Update top_k dynamically from the sidebar
    generator.top_k = k
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to initialize generator: {e}")
    st.stop()


# ─── Render chat history ───────────────────────────────────────────── #

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            citations = msg.get("citations") or []
            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        section = f", {c.section}" if c.section else ""
                        st.markdown(f"- **[{c.index}]** {c.book}, page {c.page}{section}")

            if show_chunks and msg.get("chunks"):
                with st.expander("Retrieved chunks"):
                    for i, chunk in enumerate(msg["chunks"], start=1):
                        score = chunk.get("score", 0.0)
                        book = chunk.get("book", "")
                        page = chunk.get("page", "?")
                        st.markdown(
                            f"**[{i}]** score={score:.4f} · {book}, page {page}"
                        )
                        st.text_area(
                            label=f"chunk-{i}",
                            value=chunk.get("content", ""),
                            height=160,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"chunk-{len(st.session_state.messages)}-{i}",
                        )
                        st.divider()


# ─── Input ─────────────────────────────────────────────────────────── #

query = st.chat_input("Ask a question about the books…")

if query:
    # Echo the user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate the answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and thinking…"):
            result: GeneratedAnswer = generator.answer(query)

        # Render the answer with streaming-like typing effect
        placeholder = st.empty()
        placeholder.markdown(result.answer)

        # Sources
        if result.citations:
            with st.expander("Sources"):
                for c in result.citations:
                    section = f", {c.section}" if c.section else ""
                    st.markdown(f"- **[{c.index}]** {c.book}, page {c.page}{section}")

        # Optional chunks
        if show_chunks and result.chunks:
            with st.expander("Retrieved chunks"):
                for i, chunk in enumerate(result.chunks, start=1):
                    score = chunk.get("score", 0.0)
                    book = chunk.get("book", "")
                    page = chunk.get("page", "?")
                    st.markdown(
                        f"**[{i}]** score={score:.4f} · {book}, page {page}"
                    )
                    st.text_area(
                        label=f"chunk-{i}",
                        value=chunk.get("content", ""),
                        height=160,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"new-chunk-{i}",
                    )
                    st.divider()

    # Persist the assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": result.citations,
            "chunks": result.chunks if show_chunks else [],
        }
    )