"""Chunking-Backend: Hybrid-Chunking (Markdown-Header-Split + Recursive-Fallback).

Parametrisierte Fassung der Produktiv-Logik aus `ingestion/chunker.py` (die
weiterhin `build(settings.chunk_size, settings.chunk_overlap)` aufruft) -
hier zusätzlich unter mehreren `chunk_size`-Varianten für den
Chunking-Vergleich (siehe docs/CHUNKING.md) registriert.
"""

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class _MergedSection:
    """Ein Header-Abschnitt, ggf. über mehrere Seiten zusammengeführt
    (siehe `_merge_page_boundary_continuations`)."""

    text: str
    metadata: dict
    source_file: str
    page_start: str
    page_end: str
    pdf_page_index_start: int
    pdf_page_index_end: int


def _merge_page_boundary_continuations(
    pages: list[PageDocument], header_splitter: MarkdownHeaderTextSplitter
) -> list[_MergedSection]:
    """Führt einen Header-Abschnitt, der ohne neue Überschrift über eine
    Seitengrenze hinweg fortgesetzt wird, mit dem letzten Abschnitt der
    vorigen Seite zusammen.

    Kriterium für "Fortsetzung": Der erste Header-Abschnitt einer Seite trägt
    keine Header-Metadaten (= keine Überschrift vor diesem Inhalt auf dieser
    Seite) und die Seite folgt physisch direkt auf die vorige, bereits
    erfasste Seite derselben Quelldatei. Andernfalls bleibt das Verhalten
    identisch zum reinen Pro-Seite-Splitting."""
    merged: list[_MergedSection] = []
    for page in pages:
        header_sections = header_splitter.split_text(page.text)
        if not header_sections:
            header_sections = [Document(page_content=page.text, metadata={})]

        for section in header_sections:
            is_continuation = (
                merged
                and not section.metadata
                and page.source_file == merged[-1].source_file
                and page.pdf_page_index == merged[-1].pdf_page_index_end + 1
            )
            if is_continuation:
                prev = merged[-1]
                prev.text += "\n" + section.page_content
                prev.page_end = page.page_number
                prev.pdf_page_index_end = page.pdf_page_index
            else:
                merged.append(
                    _MergedSection(
                        text=section.page_content,
                        metadata=section.metadata,
                        source_file=page.source_file,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        pdf_page_index_start=page.pdf_page_index,
                        pdf_page_index_end=page.pdf_page_index,
                    )
                )
    return merged


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
        for section in _merge_page_boundary_continuations(pages, header_splitter):
            section_path = _section_path(section.metadata)
            page_number = (
                section.page_start
                if section.page_start == section.page_end
                else f"{section.page_start}-{section.page_end}"
            )
            sub_chunks = (
                [section.text] if len(section.text) <= chunk_size else fallback_splitter.split_text(section.text)
            )
            for sub_chunk in sub_chunks:
                if not sub_chunk.strip():
                    continue
                chunks.append(
                    Document(
                        page_content=sub_chunk,
                        metadata={
                            "source_file": section.source_file,
                            "page_number": page_number,
                            "pdf_page_index": section.pdf_page_index_start,
                            "section": section_path,
                        },
                    )
                )
        return chunks

    return chunk
