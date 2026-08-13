"""Hybrid-Chunking für technische Handbücher: Markdown-Header-Splitting
(hält Abschnitte zusammen) + Recursive-Splitting als Sicherheitsnetz für
sehr lange Abschnitte. Jeder Chunk behält Quelldatei, Seitenzahl und den
Überschriftenpfad als Metadaten für das Source-Tracking im Chat.

Dünner Wrapper um `chunking_backends/header_recursive_backend.py` mit den
Produktiv-Settings aus `.env` - dieselbe Logik ist dort zusätzlich unter
mehreren chunk_size-Varianten für den Chunking-Vergleich registriert (siehe
docs/CHUNKING.md).
"""

from langchain_core.documents import Document

from ..config import settings
from .chunking_backends import header_recursive_backend
from .loader import PageDocument


def chunk_pages(pages: list[PageDocument]) -> list[Document]:
    """Zerlegt Seiten-Dokumente in durchsuchbare, metadaten-angereicherte Chunks."""
    chunk = header_recursive_backend.build(settings.chunk_size, settings.chunk_overlap)
    return chunk(pages)
