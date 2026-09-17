#!/usr/bin/env python
"""Streamlit-Frontend: klassische Chat-Oberfläche mit Auswahl zwischen der
Produktiv-Pipeline und der Best-of-Breed-Pipeline (siehe docs/BEST_OF_BREED.md).
Das Antwort-LLM ist je Pipeline fest vorgegeben, nicht separat wählbar.

Nutzung:
    streamlit run app.py
"""

import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.custom_demo import resolve_collection  # noqa: E402
from rag_manual_bot.rag.chain import build_rag_chain  # noqa: E402
from rag_manual_bot.rag.vectorstore import load_vectorstore  # noqa: E402

# Die wählbaren Gesamt-Pipelines (siehe docs/BEST_OF_BREED.md) - jede legt
# Parser, Chunking, Embedding, Retrieval-Strategie UND Antwort-LLM fest.
# Best-of-Breed wird nur noch auf dem vollen ~5.100-Seiten-Produktivkorpus
# angeboten - die RAGAS-Validierung (docs/BEST_OF_BREED.md) läuft weiterhin
# auf dem 53-Seiten-Demo-Korpus, eine offizielle Vollkorpus-RAGAS-Auswertung
# ist bewusst nicht Teil dieser Arbeit; die
# anfängliche ~53-Seiten-Demo-Korpus-Chat-Variante
# (scripts/build_best_of_breed_demo.py) war nur ein Zwischenschritt zur
# Pipeline-Auswahl und ist hier entfernt.
# GPT-4o ist bei Best-of-Breed fest hinterlegt, weil es dort das mit Abstand
# staerkste Antwort-LLM ist (⌀ 0.863 vs. gpt-4o-mini 0.821 vs. Mistral 0.772,
# RAGAS/Claude-Richter, eval_results/best_of_breed_raw_20260915_094906.csv)
# - anders als auf der Produktiv-Pipeline, wo gpt-4o-mini knapp vorne liegt
# (siehe docs/LLM.md).
PIPELINES = {
    "Produktiv (voller Korpus, ~5.100 Seiten)": {
        "corpus": "full",
        "parser": "pymupdf4llm",
        "chunking": "header_recursive_800",
        "embedding": "text-embedding-3-small",
        "retrieval": "mmr",
        "llm": "openai",
        "components": [
            ("Parser", "PyMuPDF4LLM"),
            ("Chunking", "Header+Recursive 800"),
            ("Embedding", "text-embedding-3-small"),
            ("Retrieval", "MMR"),
            ("LLM", settings.llm_model),
        ],
    },
    "Best-of-Breed (voller Korpus, ~5.100 Seiten)": {
        "corpus": "full",
        "parser": "unstructured",
        "chunking": "semantic",
        "embedding": "text-embedding-3-large",
        "retrieval": "rerank",
        "llm": "gpt4o",
        "components": [
            ("Parser", "Unstructured"),
            ("Chunking", "Semantic"),
            ("Embedding", "text-embedding-3-large"),
            ("Retrieval", "Rerank (Cross-Encoder)"),
            ("LLM", settings.gpt4o_model),
        ],
    },
}


@st.cache_resource(show_spinner=False)
def _get_chain(pipeline_label: str):
    pipeline = PIPELINES[pipeline_label]
    persist_directory, collection_name, embeddings = resolve_collection(
        pipeline["parser"], pipeline["chunking"], pipeline["embedding"], corpus=pipeline["corpus"]
    )
    vectorstore = load_vectorstore(persist_directory, collection_name, embeddings=embeddings)
    return build_rag_chain(vectorstore, llm_provider=pipeline["llm"], retrieval_strategy=pipeline["retrieval"])


def _messages_to_chat_history(messages: list[dict]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


def _static_pdf_link(source_file: str, pdf_page_index: int) -> str:
    """Baut einen PDF-Seiten-Link über Streamlits Static-File-Serving
    (app/static/ -> static/-Symlink auf data/raw_pdfs/, siehe
    .streamlit/config.toml). Anders als citations.py::source_pdf_link()
    (file://-Link, fuer den Terminal-Chat) funktioniert das im Browser
    zuverlaessig, weil der Link vom selben http(s)-Origin wie die
    Streamlit-Seite kommt - file://-Links werden von modernen Browsern beim
    Klick von einer http(s)-Seite aus blockiert."""
    return f"app/static/{quote(source_file)}#page={pdf_page_index}"


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
            link = _static_pdf_link(meta["source_file"], meta["pdf_page_index"])
            lines.append(f"- [{label}]({link})")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


st.set_page_config(page_title="Abaqus Handbuch-Chatbot", page_icon="🛠️", layout="centered")

with st.sidebar:
    st.header("Einstellungen")

    pipeline_label = st.radio("Pipeline", list(PIPELINES), index=0)
    pipeline = PIPELINES[pipeline_label]
    st.caption(
        "  \n".join(f"**{label}:** {value}" for label, value in pipeline["components"])
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
                chain = _get_chain(pipeline_label)
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
