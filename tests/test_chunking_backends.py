"""Tests für die Chunking-Backends im Chunking-Vergleich (siehe docs/CHUNKING.md).

Laufen vollständig offline (deterministische Fake-Embeddings für
`semantic_backend`, kein OpenAI-Call).
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tiktoken
from langchain_core.embeddings import Embeddings

from rag_manual_bot.ingestion.chunking_backends import (
    CHUNKING_BACKENDS,
    header_recursive_backend,
    recursive_only_backend,
    semantic_backend,
    token_backend,
)
from rag_manual_bot.ingestion.loader import PageDocument


class FakeEmbeddings(Embeddings):
    """Deterministische, offline lauffähige Embeddings für Tests (kein OpenAI-Call)."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _metadata_keys_present(chunks) -> bool:
    return all(
        {"source_file", "page_number", "pdf_page_index", "section"} <= set(c.metadata)
        for c in chunks
    )


def test_registry_contains_all_six_variants():
    assert set(CHUNKING_BACKENDS) == {
        "header_recursive_400",
        "header_recursive_800",
        "header_recursive_1600",
        "recursive_only",
        "token_based",
        "semantic",
    }


def test_header_recursive_keeps_headers_and_metadata():
    page = PageDocument(
        text=(
            "# Kapitel 1\n\n"
            "Einleitungstext.\n\n"
            "## Abschnitt 1.1\n\n"
            "Details zum Abschnitt."
        ),
        source_file="manual.pdf",
        page_number="12",
        pdf_page_index=14,
    )

    chunk = header_recursive_backend.build(chunk_size=800, chunk_overlap=150)
    chunks = chunk([page])

    assert len(chunks) >= 2
    assert _metadata_keys_present(chunks)
    assert all(c.metadata["source_file"] == "manual.pdf" for c in chunks)
    assert all(c.metadata["page_number"] == "12" for c in chunks)
    assert all(c.metadata["pdf_page_index"] == 14 for c in chunks)
    sections = {c.metadata["section"] for c in chunks}
    assert any("Abschnitt 1.1" in s for s in sections)


def test_header_recursive_size_variants_scale_chunk_count():
    long_text = "# Großer Abschnitt\n\n" + ("Ein langer Satz mit Inhalt. " * 300)
    page = PageDocument(text=long_text, source_file="manual.pdf", page_number="1", pdf_page_index=1)

    small = header_recursive_backend.build(chunk_size=400, chunk_overlap=75)([page])
    large = header_recursive_backend.build(chunk_size=1600, chunk_overlap=300)([page])

    assert all(len(c.page_content) <= 450 for c in small)  # chunk_size + Toleranz
    assert all(len(c.page_content) <= 1700 for c in large)
    assert len(small) > len(large)  # kleinere Chunks -> mehr Stück


def test_header_recursive_merges_section_continuing_across_page_break():
    # Seite 12 endet mitten im Abschnitt "Abschnitt 1.1", Seite 13 fuehrt ihn
    # ohne neue Ueberschrift fort.
    page_12 = PageDocument(
        text="# Kapitel 1\n\n## Abschnitt 1.1\n\nErster Teil des Abschnitts.",
        source_file="manual.pdf",
        page_number="12",
        pdf_page_index=14,
    )
    page_13 = PageDocument(
        text="Fortsetzung desselben Abschnitts ohne eigene Ueberschrift.",
        source_file="manual.pdf",
        page_number="13",
        pdf_page_index=15,
    )

    chunk = header_recursive_backend.build(chunk_size=800, chunk_overlap=150)
    chunks = chunk([page_12, page_13])

    assert len(chunks) == 1
    merged = chunks[0]
    assert "Erster Teil" in merged.page_content
    assert "Fortsetzung desselben Abschnitts" in merged.page_content
    assert merged.metadata["page_number"] == "12-13"
    assert merged.metadata["pdf_page_index"] == 14  # Deep-Link zeigt auf die Startseite
    assert "Abschnitt 1.1" in merged.metadata["section"]


def test_header_recursive_does_not_merge_when_next_page_starts_with_heading():
    page_12 = PageDocument(
        text="# Kapitel 1\n\nText auf Seite 12.",
        source_file="manual.pdf",
        page_number="12",
        pdf_page_index=14,
    )
    page_13 = PageDocument(
        text="# Kapitel 2\n\nText auf Seite 13.",
        source_file="manual.pdf",
        page_number="13",
        pdf_page_index=15,
    )

    chunk = header_recursive_backend.build(chunk_size=800, chunk_overlap=150)
    chunks = chunk([page_12, page_13])

    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == "12"
    assert chunks[1].metadata["page_number"] == "13"


def test_header_recursive_does_not_merge_across_non_consecutive_pages():
    # Zwischen den physischen PDF-Positionen liegt eine vom Loader bereits
    # herausgefilterte Leerseite (pdf_page_index springt von 14 auf 16) -
    # trotz fehlender Ueberschrift auf der Folgeseite darf hier nicht
    # zusammengefuehrt werden.
    page_12 = PageDocument(
        text="# Kapitel 1\n\nText auf Seite 12.",
        source_file="manual.pdf",
        page_number="12",
        pdf_page_index=14,
    )
    page_14 = PageDocument(
        text="Text ohne eigene Ueberschrift auf Seite 14.",
        source_file="manual.pdf",
        page_number="14",
        pdf_page_index=16,
    )

    chunk = header_recursive_backend.build(chunk_size=800, chunk_overlap=150)
    chunks = chunk([page_12, page_14])

    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == "12"
    assert chunks[1].metadata["page_number"] == "14"


def test_recursive_only_has_no_section_and_respects_chunk_size():
    long_text = "Ein langer Satz mit Inhalt. " * 300
    page = PageDocument(text=long_text, source_file="manual.pdf", page_number="1", pdf_page_index=1)

    chunks = recursive_only_backend.chunk([page])

    assert len(chunks) > 1
    assert _metadata_keys_present(chunks)
    assert all(c.metadata["section"] == "" for c in chunks)
    assert all(len(c.page_content) <= 900 for c in chunks)


def test_token_backend_respects_token_budget():
    long_text = "Ein langer Satz mit technischem Inhalt zu Abaqus. " * 300
    page = PageDocument(text=long_text, source_file="manual.pdf", page_number="1", pdf_page_index=1)

    chunks = token_backend.chunk([page])
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 1
    assert _metadata_keys_present(chunks)
    assert all(c.metadata["section"] == "" for c in chunks)
    assert all(len(encoding.encode(c.page_content)) <= 250 for c in chunks)  # 200 + Toleranz


def test_semantic_backend_runs_offline_with_fake_embeddings():
    text = (
        "Abaqus/CAE ist die grafische Oberfläche für Abaqus. "
        "Man erstellt zuerst ein Part im Part-Modul. "
        "Danach werden Materialien und Sections definiert. "
        "Randbedingungen werden im Load-Modul festgelegt. "
        "Zum Schluss wird der Job im Job-Modul eingereicht. "
        "Die Ergebnisse lassen sich im Visualization-Modul auswerten."
    )
    page = PageDocument(text=text, source_file="manual.pdf", page_number="5", pdf_page_index=7)

    chunks = semantic_backend.chunk([page], embeddings=FakeEmbeddings())

    assert len(chunks) >= 1
    assert _metadata_keys_present(chunks)
    assert all(c.metadata["source_file"] == "manual.pdf" for c in chunks)
    assert all(c.metadata["pdf_page_index"] == 7 for c in chunks)
    assert all(c.metadata["section"] == "" for c in chunks)
    assert "".join(c.page_content for c in chunks).strip() != ""


def test_all_backends_skip_empty_pages():
    page = PageDocument(text="   \n\n  ", source_file="manual.pdf", page_number="1", pdf_page_index=1)
    for name, chunk in CHUNKING_BACKENDS.items():
        if name == "semantic":
            continue  # separat getestet (braucht injizierte Fake-Embeddings)
        assert chunk([page]) == [], f"{name} sollte für leere Seiten keine Chunks erzeugen"
