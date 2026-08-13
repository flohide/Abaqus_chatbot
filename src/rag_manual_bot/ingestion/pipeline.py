"""Orchestriert die gesamte Ingestion: PDFs laden -> chunken -> embedden -> persistieren."""

from ..config import settings
from ..rag.vectorstore import build_vectorstore
from .chunker import chunk_pages
from .loader import load_pdf_directory


def run_ingestion() -> int:
    """Baut die Wissensbasis komplett neu auf. Gibt die Anzahl gespeicherter Chunks zurück."""
    pages = load_pdf_directory(settings.raw_pdf_dir)
    chunks = chunk_pages(pages)
    build_vectorstore(chunks)
    return len(chunks)
