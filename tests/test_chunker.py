import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_manual_bot.ingestion.chunker import chunk_pages
from rag_manual_bot.ingestion.loader import PageDocument


def test_chunk_pages_keeps_headers_and_metadata():
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

    chunks = chunk_pages([page])

    assert len(chunks) >= 2
    assert all(c.metadata["source_file"] == "manual.pdf" for c in chunks)
    assert all(c.metadata["page_number"] == "12" for c in chunks)
    assert all(c.metadata["pdf_page_index"] == 14 for c in chunks)
    sections = {c.metadata["section"] for c in chunks}
    assert any("Abschnitt 1.1" in s for s in sections)


def test_chunk_pages_splits_oversized_section():
    long_text = "# Großer Abschnitt\n\n" + ("Ein langer Satz. " * 200)
    page = PageDocument(text=long_text, source_file="manual.pdf", page_number="1", pdf_page_index=1)

    chunks = chunk_pages([page])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 900 for c in chunks)  # chunk_size + Toleranz


def test_chunk_pages_skips_empty_sections():
    page = PageDocument(text="   \n\n  ", source_file="manual.pdf", page_number="1", pdf_page_index=1)
    chunks = chunk_pages([page])
    assert chunks == []
