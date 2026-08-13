import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_manual_bot.rag.chain import format_docs
from rag_manual_bot.rag.retriever import build_retriever


class FakeEmbeddings(Embeddings):
    """Deterministische, offline lauffähige Embeddings für Tests (kein OpenAI-Call)."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _sample_vectorstore() -> Chroma:
    docs = [
        Document(
            page_content="So erstellt man ein neues Modell in Abaqus/CAE.",
            metadata={"source_file": "GETTINGSTARTED.pdf", "page_number": "10", "section": "Model"},
        ),
        Document(
            page_content="Definition von Randbedingungen im Load-Modul.",
            metadata={"source_file": "GETTINGSTARTED.pdf", "page_number": "22", "section": "Load"},
        ),
    ]
    return Chroma.from_documents(docs, embedding=FakeEmbeddings())


def test_build_retriever_returns_documents():
    retriever = build_retriever(_sample_vectorstore())
    results = retriever.invoke("Modell erstellen")
    assert len(results) > 0
    assert all(isinstance(d.metadata.get("page_number"), str) for d in results)


def test_format_docs_includes_source_citation():
    docs = [
        Document(
            page_content="Beispieltext.",
            metadata={"source_file": "GETTINGSTARTED.pdf", "page_number": "42", "section": "Intro"},
        )
    ]
    formatted = format_docs(docs)
    assert "GETTINGSTARTED.pdf" in formatted
    assert "Seite 42" in formatted
    assert "Beispieltext." in formatted
