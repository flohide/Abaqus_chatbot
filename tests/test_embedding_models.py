"""Tests für den Embedding-Modell-Vergleich (siehe docs/EMBEDDING.md).

Laufen vollständig offline: Konstruieren eines `OpenAIEmbeddings`-Clients
prüft nur die übergebenen Parameter (kein API-Call), die HuggingFace-Modelle
werden nicht real geladen - `_PrefixedHuggingFaceEmbeddings` bekommt dafür
eine injizierte Fake-Embeddings-Instanz.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.embeddings import Embeddings

from rag_manual_bot.config import settings
from rag_manual_bot.ingestion.embedding_models import (
    EMBEDDING_BACKENDS,
    _PrefixedHuggingFaceEmbeddings,
    _resolve_openai_model,
)


class _RecordingEmbeddings(Embeddings):
    """Zeichnet auf, mit welchen (bereits ggf. prefixten) Texten sie aufgerufen wurde."""

    def __init__(self):
        self.last_documents: list[str] = []
        self.last_query: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.last_documents = list(texts)
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [0.0]


def test_registry_has_three_openai_and_four_huggingface_entries():
    assert set(EMBEDDING_BACKENDS) == {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
        "multilingual-e5-large",
        "bge-m3",
        "paraphrase-multilingual-mpnet",
        "minilm-l6-en",
    }


def test_resolve_openai_model_applies_configured_prefix():
    resolved = _resolve_openai_model("text-embedding-3-large")
    if "/" in settings.embedding_model:
        prefix = settings.embedding_model.rsplit("/", 1)[0]
        assert resolved == f"{prefix}/text-embedding-3-large"
    else:
        assert resolved == "text-embedding-3-large"


def test_openai_backends_build_client_with_resolved_model():
    for name in ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"):
        client = EMBEDDING_BACKENDS[name]()
        assert client.model == _resolve_openai_model(name)


def test_huggingface_prefix_wrapper_applies_query_and_passage_prefixes():
    fake = _RecordingEmbeddings()
    wrapped = _PrefixedHuggingFaceEmbeddings(
        "intfloat/multilingual-e5-large", query_prefix="query: ", passage_prefix="passage: ", inner=fake
    )

    wrapped.embed_query("Wie erstellt man ein Modell?")
    assert fake.last_query == "query: Wie erstellt man ein Modell?"

    wrapped.embed_documents(["Text A", "Text B"])
    assert fake.last_documents == ["passage: Text A", "passage: Text B"]


def test_huggingface_prefix_wrapper_defaults_to_no_prefix():
    fake = _RecordingEmbeddings()
    wrapped = _PrefixedHuggingFaceEmbeddings("BAAI/bge-m3", inner=fake)

    wrapped.embed_query("Frage")
    assert fake.last_query == "Frage"

    wrapped.embed_documents(["Chunk"])
    assert fake.last_documents == ["Chunk"]
