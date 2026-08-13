"""Parser-Backend: pdfplumber (siehe docs/PARSER.md)."""

from pathlib import Path

import pdfplumber


def parse(pdf_path: Path) -> dict[int, str]:
    """Gibt Fließtext pro Seite zurück (1-basiert). Keine Tabellenerkennung."""
    result: dict[int, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            result[i] = page.extract_text() or ""
    return result
