"""Lädt PDFs seitenweise als Markdown (Layout, Tabellen, Überschriften bleiben erhalten)."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm


@dataclass
class PageDocument:
    """Eine einzelne PDF-Seite als Markdown-Text mit Herkunfts-Metadaten."""

    text: str
    source_file: str
    page_number: str  # gedrucktes Seiten-Label (z. B. "1059" oder "ix"), nicht die PDF-Position
    pdf_page_index: int  # 1-basierte physische Position in der PDF-Datei (für Deep-Links, #page=N)


def load_pdf(pdf_path: Path) -> list[PageDocument]:
    """Konvertiert eine einzelne PDF-Datei in eine Liste von Seiten-Dokumenten.

    Handbücher haben oft ein Titelblatt/römisch nummeriertes Vorwort vor dem
    eigentlichen Inhalt, wodurch die reine PDF-Seitenposition von der unten
    auf der Seite gedruckten Seitenzahl abweicht. PyMuPDFs Page-Labels
    (`/PageLabels` im PDF-Katalog) bilden genau diese gedruckte Nummerierung
    ab und werden hier statt der sequenziellen Position verwendet.
    """
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    doc = pymupdf.open(str(pdf_path))
    try:
        result = []
        for page in pages:
            if not page["text"].strip():
                continue
            pdf_index_1based = page["metadata"]["page_number"]
            printed_label = doc[pdf_index_1based - 1].get_label() or str(pdf_index_1based)
            result.append(
                PageDocument(
                    text=page["text"],
                    source_file=pdf_path.name,
                    page_number=printed_label,
                    pdf_page_index=pdf_index_1based,
                )
            )
        return result
    finally:
        doc.close()


def load_pdf_directory(pdf_dir: Path) -> list[PageDocument]:
    """Lädt alle PDFs in einem Verzeichnis (sortiert für reproduzierbare Ingestion)."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Keine PDF-Dateien in {pdf_dir} gefunden.")

    all_pages: list[PageDocument] = []
    for pdf_path in pdf_paths:
        all_pages.extend(load_pdf(pdf_path))
    return all_pages
