#!/usr/bin/env python
"""Streamlit-Frontend: klassische Chat-Oberfläche mit unabhängiger Auswahl von
Parser, Chunking, Embedding, Retrieval-Strategie und Antwort-LLM.

Nutzung:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

from rag_manual_bot.citations import source_pdf_link  # noqa: E402
from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.chunking_backends import CHUNKING_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.custom_demo import (  # noqa: E402
    DEFAULT_CHUNKING,
    DEFAULT_EMBEDDING,
    DEFAULT_PARSER,
    estimate_build_seconds,
    is_known_combination,
    is_parser_cached,
    resolve_collection,
)
from rag_manual_bot.ingestion.embedding_models import EMBEDDING_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.parser_backends import PARSER_BACKENDS  # noqa: E402
from rag_manual_bot.rag.chain import build_rag_chain  # noqa: E402
from rag_manual_bot.rag.llms import mistral_available  # noqa: E402
from rag_manual_bot.rag.retrieval_backends import RETRIEVAL_BACKENDS  # noqa: E402
from rag_manual_bot.rag.vectorstore import load_vectorstore  # noqa: E402

PARSER_LABELS = {
    "pymupdf4llm": "PyMuPDF4LLM (Produktiv)",
    "pdfplumber": "pdfplumber",
    "docling": "Docling (sehr langsam, siehe docs/PARSER.md)",
    "unstructured": "Unstructured (langsam, siehe docs/PARSER.md)",
}
CHUNKING_LABELS = {
    "header_recursive_400": "Header+Recursive 400",
    "header_recursive_800": "Header+Recursive 800 (Produktiv)",
    "header_recursive_1600": "Header+Recursive 1600",
    "recursive_only": "Recursive-only",
    "token_based": "Token-basiert",
    "semantic": "Semantic (langsam, siehe docs/CHUNKING.md)",
}
EMBEDDING_LABELS = {
    "text-embedding-3-small": "text-embedding-3-small (Produktiv)",
    "text-embedding-3-large": "text-embedding-3-large",
    "text-embedding-ada-002": "text-embedding-ada-002",
    "multilingual-e5-large": "multilingual-e5-large (lokal/HF)",
    "bge-m3": "bge-m3 (lokal/HF)",
    "paraphrase-multilingual-mpnet": "paraphrase-multilingual-mpnet (lokal/HF)",
    "minilm-l6-en": "minilm-l6-en (lokal/HF)",
}

PARSER_OPTIONS = {PARSER_LABELS[k]: k for k in PARSER_BACKENDS}
CHUNKING_OPTIONS = {CHUNKING_LABELS[k]: k for k in CHUNKING_BACKENDS}
EMBEDDING_OPTIONS = {EMBEDDING_LABELS[k]: k for k in EMBEDDING_BACKENDS}

LLM_OPTIONS = {
    f"OpenAI ({settings.llm_model})": "openai",
    f"Mistral ({settings.mistral_model})": "mistral",
    f"Qwen ({settings.qwen_model})": "qwen",
}

RETRIEVAL_OPTIONS = {
    "MMR (Produktiv)": "mmr",
    "Similarity (ohne MMR-Diversität)": "similarity",
    "Rerank (Cross-Encoder)": "rerank",
    "Hybrid (BM25 + Dense)": "hybrid",
}


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"~{seconds:.0f} Sek."
    minutes = seconds / 60
    if minutes < 90:
        return f"~{minutes:.0f} Min."
    return f"~{minutes / 60:.1f} Std."


@st.cache_resource(show_spinner=False)
def _get_chain(
    corpus_mode: str,
    parser_key: str,
    chunking_key: str,
    embedding_key: str,
    llm_provider: str,
    retrieval_strategy: str,
):
    corpus = "full" if corpus_mode == "produktiv" else "demo"
    persist_directory, collection_name, embeddings = resolve_collection(
        parser_key, chunking_key, embedding_key, corpus=corpus
    )
    vectorstore = load_vectorstore(persist_directory, collection_name, embeddings=embeddings)
    return build_rag_chain(vectorstore, llm_provider=llm_provider, retrieval_strategy=retrieval_strategy)


def _messages_to_chat_history(messages: list[dict]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def _format_sources(source_documents) -> str:
    seen = set()
    lines = []
    for doc in source_documents:
        meta = doc.metadata
        key = (meta.get("source_file"), meta.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        label = f"{meta.get('source_file')}, Seite {meta.get('page_number')}"
        if meta.get("pdf_page_index") is not None:
            link = source_pdf_link(meta["source_file"], meta["pdf_page_index"])
            lines.append(f"- [{label}]({link})")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


st.set_page_config(page_title="Abaqus Handbuch-Chatbot", page_icon="🛠️", layout="centered")

with st.sidebar:
    st.header("Einstellungen")

    corpus_label = st.radio(
        "Korpus",
        ["Produktiv (voller Korpus, ~5.100 Seiten)", "Demo-Korpus (~53 Seiten)"],
        index=0,
    )
    corpus_mode = "produktiv" if corpus_label.startswith("Produktiv") else "demo"
    corpus = "full" if corpus_mode == "produktiv" else "demo"

    st.subheader("Wissensbasis")
    parser_label = st.selectbox("Parser", list(PARSER_OPTIONS), index=list(PARSER_OPTIONS.values()).index(DEFAULT_PARSER))
    chunking_label = st.selectbox(
        "Chunking", list(CHUNKING_OPTIONS), index=list(CHUNKING_OPTIONS.values()).index(DEFAULT_CHUNKING)
    )
    embedding_label = st.selectbox(
        "Embedding", list(EMBEDDING_OPTIONS), index=list(EMBEDDING_OPTIONS.values()).index(DEFAULT_EMBEDDING)
    )
    parser_key = PARSER_OPTIONS[parser_label]
    chunking_key = CHUNKING_OPTIONS[chunking_label]
    embedding_key = EMBEDDING_OPTIONS[embedding_label]

    if corpus_mode == "produktiv":
        st.caption("Voller ~5.100-Seiten-Korpus.")
    else:
        st.caption("~53-Seiten-Demo-Korpus (siehe docs/PARSER.md) — nicht der volle Produktiv-Korpus.")

    if not is_known_combination(parser_key, chunking_key, embedding_key, corpus=corpus):
        estimate_s = estimate_build_seconds(parser_key, chunking_key, embedding_key, corpus=corpus)
        parser_note = (
            " (Parser bereits in dieser Sitzung ausgeführt — der teuerste Schritt entfällt.)"
            if is_parser_cached(parser_key, corpus)
            else ""
        )
        message = (
            f"Diese Kombination wurde noch nie gebaut und wird jetzt beim ersten "
            f"Absenden einer Frage einmalig neu erzeugt — geschätzte Bauzeit: "
            f"**{_format_duration(estimate_s)}**{parser_note} (grobe Schätzung, siehe "
            f"docs/PARSER.md, docs/EMBEDDING.md). Der gesamte Chatbot ist währenddessen "
            f"blockiert. Danach ist die Kombination gecacht — und der Parser-Schritt "
            f"bleibt für weitere Kombinationen mit demselben Parser in dieser Sitzung "
            f"ebenfalls gecacht."
        )
        if estimate_s > 1800:
            st.error(message, icon="🛑")
        else:
            st.warning(message, icon="⏳")

    st.divider()

    llm_label = st.selectbox("LLM", list(LLM_OPTIONS), index=0)
    llm_provider = LLM_OPTIONS[llm_label]
    if llm_provider == "mistral" and not mistral_available():
        st.warning(
            "MISTRAL_API_KEY ist nicht in .env gesetzt. "
            "Bitte eintragen, um Mistral zu verwenden.",
            icon="⚠️",
        )

    retrieval_label = st.selectbox("Retrieval-Strategie", list(RETRIEVAL_OPTIONS), index=0)
    retrieval_strategy = RETRIEVAL_OPTIONS[retrieval_label]
    if retrieval_strategy in ("rerank", "hybrid"):
        st.caption(
            "Braucht requirements-retrieval-comparison.txt (rank_bm25 / "
            "sentence-transformers) und ist beim ersten Aufruf pro Sitzung "
            "langsamer (Cross-Encoder-Modell-Download bzw. BM25-Indexaufbau, "
            "siehe docs/RETRIEVAL.md)."
        )

    st.divider()
    if st.button("Gesprächsverlauf zurücksetzen", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🛠️ Abaqus Handbuch-Chatbot")
st.caption(
    "RAG-Chatbot für Abaqus-Handbücher — mit Seiten-genauem Source-Tracking. "
    "Quellen-Links öffnen die PDF lokal auf der passenden Seite "
    "(je nach Browser-Sicherheitsrichtlinie evtl. mit Bestätigung)."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Quellen"):
                st.markdown(message["sources"])

question = st.chat_input("Frage zur Bedienung von Abaqus...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Wissensbasis wird geladen (ggf. einmalig neu aufgebaut) ..."):
                chain = _get_chain(corpus_mode, parser_key, chunking_key, embedding_key, llm_provider, retrieval_strategy)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        with st.spinner("Suche im Handbuch..."):
            chat_history = _messages_to_chat_history(st.session_state.messages[:-1])
            try:
                result = chain.invoke({"question": question, "chat_history": chat_history})
            except Exception as exc:  # API-/Netzwerkfehler nicht die App abstürzen lassen
                st.error(f"Fehler bei der Anfrage: {exc}")
                st.stop()

        st.markdown(result["answer"])
        sources = _format_sources(result["source_documents"])
        if sources:
            with st.expander("Quellen"):
                st.markdown(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": sources}
    )
