"""Chunking-Backend: reines Recursive-Character-Splitting (naive Baseline).

Im Gegensatz zu `header_recursive_backend.py` ohne Markdown-Header-Splitting -
zerlegt jede Seite direkt zeichenbasiert an Absatz-/Satz-/Wortgrenzen, ohne
die Abschnittsstruktur des Handbuchs zu berücksichtigen. Dient als naive
Baseline im Chunking-Vergleich (siehe docs/CHUNKING.md), chunk_size/overlap
identisch zum Produktiv-Default (800/150), damit der Vergleich mit
`header_recursive_800` ausschließlich die Header-Awareness isoliert.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..loader import PageDocument

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 150


def chunk(pages: list[PageDocument]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page in pages:
        for sub_chunk in splitter.split_text(page.text):
            if not sub_chunk.strip():
                continue
            chunks.append(
                Document(
                    page_content=sub_chunk,
                    metadata={
                        "source_file": page.source_file,
                        "page_number": page.page_number,
                        "pdf_page_index": page.pdf_page_index,
                        "section": "",
                    },
                )
            )
    return chunks
