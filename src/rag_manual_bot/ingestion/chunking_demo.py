"""Baut den Chunking-Vergleichs-Demo-Korpus: denselben Demo-Korpus wie der
Parser-Vergleich (siehe `parser_demo.py`), einmal mit jeder registrierten
Chunking-Strategie zerlegt, jeweils in eine eigene Chroma-Collection unter
vectorstore_chunking_demo/.

Der Parser ist bewusst fix auf PyMuPDF4LLM gesetzt (Produktiv-Parser, siehe
docs/PARSER.md) - der Chunking-Vergleich variiert ausschließlich die
Chunking-Strategie, nicht zusätzlich den Parser (siehe docs/CHUNKING.md).
"""

from .chunking_backends import CHUNKING_BACKENDS
from .loader import PageDocument
from .parser_backends import PARSER_BACKENDS
from .parser_demo import DEMO_PDF_PATH, build_demo_pdf
from ..config import settings
from ..rag.vectorstore import build_vectorstore

_FIXED_PARSER = "pymupdf4llm"


def run_chunking_demo_ingestion(chunkers: list[str] | None = None) -> dict[str, int]:
    """Baut für jede gewählte Chunking-Strategie eine eigene Collection.
    Gibt {chunker_name: anzahl_chunks} zurück."""
    chunkers = chunkers or list(CHUNKING_BACKENDS)
    refs = build_demo_pdf()

    parse = PARSER_BACKENDS[_FIXED_PARSER]
    pages_by_index = parse(DEMO_PDF_PATH)

    pages = [
        PageDocument(
            text=pages_by_index.get(i, ""),
            source_file=ref.source_file,
            page_number=ref.page_number,
            pdf_page_index=ref.pdf_page_index,
        )
        for i, ref in enumerate(refs, start=1)
        if pages_by_index.get(i, "").strip()
    ]

    result = {}
    for chunker_name in chunkers:
        chunk = CHUNKING_BACKENDS[chunker_name]
        chunks = chunk(pages)
        build_vectorstore(
            chunks,
            persist_directory=settings.chunking_demo_vectorstore_dir,
            collection_name=f"chunking_demo_{chunker_name}",
        )
        result[chunker_name] = len(chunks)
    return result
