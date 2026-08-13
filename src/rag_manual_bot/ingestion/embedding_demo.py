"""Baut den Embedding-Modell-Vergleichs-Demo-Korpus: denselben Demo-Korpus wie
Parser- und Chunking-Vergleich (siehe `parser_demo.py`), fix mit PyMuPDF4LLM
geparst und mit dem Produktiv-Chunking (`chunker.chunk_pages`,
Header+Recursive 800) zerlegt - isoliert wird ausschließlich das
Embedding-Modell, jedes Modell erhält eine eigene Chroma-Collection unter
vectorstore_embedding_demo/ (siehe docs/EMBEDDING.md).
"""

from .chunker import chunk_pages
from .embedding_models import EMBEDDING_BACKENDS
from .loader import PageDocument
from .parser_backends import PARSER_BACKENDS
from .parser_demo import DEMO_PDF_PATH, build_demo_pdf
from ..config import settings
from ..rag.vectorstore import build_vectorstore

_FIXED_PARSER = "pymupdf4llm"


def run_embedding_demo_ingestion(models: list[str] | None = None) -> dict[str, int]:
    """Baut für jedes gewählte Embedding-Modell eine eigene Collection.
    Gibt {model_name: anzahl_chunks} zurück."""
    models = models or list(EMBEDDING_BACKENDS)
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
    chunks = chunk_pages(pages)

    result = {}
    for model_name in models:
        build_vectorstore(
            chunks,
            persist_directory=settings.embedding_demo_vectorstore_dir,
            collection_name=f"embedding_demo_{model_name}",
            embeddings=EMBEDDING_BACKENDS[model_name](),
        )
        result[model_name] = len(chunks)
    return result
