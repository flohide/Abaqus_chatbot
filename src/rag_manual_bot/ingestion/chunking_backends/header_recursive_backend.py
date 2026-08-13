"""Chunking-Backend: Hybrid-Chunking (Markdown-Header-Split + Recursive-Fallback).

Parametrisierte Fassung der Produktiv-Logik aus `ingestion/chunker.py` (die
weiterhin `build(settings.chunk_size, settings.chunk_overlap)` aufruft) -
hier zusätzlich unter mehreren `chunk_size`-Varianten für den
Chunking-Vergleich (siehe docs/CHUNKING.md) registriert.
"""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..loader import PageDocument

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def _section_path(metadata: dict) -> str:
    """Baut aus den Header-Metadaten einen lesbaren Pfad wie 'h1 > h2 > h3'."""
    parts = [metadata[key] for key in ("h1", "h2", "h3") if metadata.get(key)]
    return " > ".join(parts)


def build(chunk_size: int, chunk_overlap: int) -> Callable[[list[PageDocument]], list[Document]]:
    """Baut eine `chunk(pages)`-Funktion für das gegebene chunk_size/chunk_overlap-Paar."""

    def chunk(pages: list[PageDocument]) -> list[Document]:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[Document] = []
        for page in pages:
            header_sections = header_splitter.split_text(page.text)
            if not header_sections:
                header_sections = [Document(page_content=page.text, metadata={})]

            for section in header_sections:
                section_path = _section_path(section.metadata)
                sub_chunks = (
                    [section.page_content]
                    if len(section.page_content) <= chunk_size
                    else fallback_splitter.split_text(section.page_content)
                )
                for sub_chunk in sub_chunks:
                    if not sub_chunk.strip():
                        continue
                    chunks.append(
                        Document(
                            page_content=sub_chunk,
                            metadata={
                                "source_file": page.source_file,
                                "page_number": page.page_number,
                                "pdf_page_index": page.pdf_page_index,
                                "section": section_path,
                            },
                        )
                    )
        return chunks

    return chunk
