"""Parser-Backend: PyMuPDF4LLM (Produktiv-Parser, siehe docs/PARSER.md)."""

from pathlib import Path

import pymupdf4llm


def parse(pdf_path: Path) -> dict[int, str]:
    """Gibt Markdown-Text pro Seite zurück (1-basiert)."""
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    return {i: page["text"] for i, page in enumerate(pages, start=1)}
