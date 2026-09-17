"""Chunking-Backend: Semantic Chunking (embedding-basierte Breakpoints).

Nutzt `langchain_experimental.text_splitter.SemanticChunker`: Sätze werden
einzeln eingebettet, Chunk-Grenzen entstehen dort, wo die Embedding-Distanz
zwischen aufeinanderfolgenden Sätzen ein Perzentil-Schwellwert überschreitet
- statt fester Zeichen-/Token-Grenzen oder Markdown-Struktur. Genau die in
docs/DOKUMENTATION.md §3 als "unnötig teuer" verworfene Alternative, hier
empirisch gegen die anderen Backends geprüft (siehe docs/CHUNKING.md).

`embeddings` ist injizierbar (Default: Produktiv-`OpenAIEmbeddings`), damit
Tests offline mit einer Fake-Implementierung laufen können.

`langchain_experimental` wird bewusst erst innerhalb von `chunk()` importiert
(nicht auf Modulebene): Es ist eine optionale Extra-Dependency (siehe
requirements-chunking-comparison.txt), die z. B. der Parser-Vergleich/die
Parser-RAGAS-Evaluation nicht braucht - `chunking_backends/__init__.py` (und
damit `eval/run.py`) muss aber auch ohne sie importierbar bleiben.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ...config import settings
from ..loader import PageDocument


def _default_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        request_timeout=settings.api_request_timeout,
        max_retries=settings.api_max_retries,
    )


def chunk(pages: list[PageDocument], embeddings: Embeddings | None = None) -> list[Document]:
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(embeddings or _default_embeddings())

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
