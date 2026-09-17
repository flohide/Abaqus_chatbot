"""Erzeugt bzw. lädt lokale Chroma-Vektorstores.

Unterstützt neben dem Produktiv-Vectorstore auch beliebige weitere
Collections (z. B. die Parser-/Chunking-/Embedding-Vergleichs-Collections in
vectorstore_parser_demo/, vectorstore_chunking_demo/,
vectorstore_embedding_demo/) über explizite persist_directory/collection_name.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..config import settings


def _embeddings() -> OpenAIEmbeddings:
    """Produktiv-Default: OpenAI-Embeddings mit `settings.embedding_model`."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        request_timeout=settings.api_request_timeout,
        max_retries=settings.api_max_retries,
    )


def _open_store(
    persist_directory: Path | None, collection_name: str | None, embeddings: Embeddings | None
) -> Chroma:
    persist_directory = persist_directory or settings.vectorstore_dir
    collection_name = collection_name or settings.collection_name
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or _embeddings(),
        persist_directory=str(persist_directory),
    )


def _add_in_batches(store: Chroma, documents: list[Document], batch_size: int = 200) -> None:
    for i in range(0, len(documents), batch_size):
        store.add_documents(documents[i : i + batch_size])


def build_vectorstore(
    documents: list[Document],
    persist_directory: Path | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Embedded Chunks und persistiert sie neu (überschreibt eine bestehende Collection).

    `embeddings` überschreibt den Produktiv-Default (siehe `docs/EMBEDDING.md`,
    `ingestion/embedding_models.py::EMBEDDING_BACKENDS`) - wer eine Collection
    damit aufbaut, muss sie später mit demselben Embeddings-Objekt laden
    (`load_vectorstore`), da unterschiedliche Modelle unterschiedliche
    Dimensionalität und Vektorräume haben."""
    store = _open_store(persist_directory, collection_name, embeddings)
    existing_ids = store.get()["ids"]
    if existing_ids:
        store.delete(ids=existing_ids)
    _add_in_batches(store, documents)
    return store


def add_documents_to_vectorstore(
    documents: list[Document],
    persist_directory: Path | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Ergänzt Chunks in einer bestehenden Collection, ohne sie zu leeren -
    für das inkrementelle Ingestion-Update (siehe `ingestion/pipeline.py`).
    Für neue/geänderte Dateien müssen deren
    alte Chunks vorher separat über `delete_source_files` entfernt werden."""
    store = _open_store(persist_directory, collection_name, embeddings)
    _add_in_batches(store, documents)
    return store


def delete_source_files(
    source_files: set[str] | list[str],
    persist_directory: Path | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> None:
    """Entfernt alle Chunks der genannten Quelldateien aus einer bestehenden
    Collection (Metadaten-Feld `source_file`, siehe `chunking_backends/
    header_recursive_backend.py`) - für das inkrementelle Ingestion-Update."""
    if not source_files:
        return
    store = _open_store(persist_directory, collection_name, embeddings)
    for name in source_files:
        store.delete(where={"source_file": name})


def load_vectorstore(
    persist_directory: Path | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Lädt einen zuvor erzeugten Vektorstore für die Chat-Sitzung.

    `embeddings` muss zu dem Modell passen, mit dem die Collection via
    `build_vectorstore` aufgebaut wurde (siehe dortiger Hinweis)."""
    persist_directory = persist_directory or settings.vectorstore_dir
    collection_name = collection_name or settings.collection_name
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Kein Vektorstore unter {persist_directory} gefunden. "
            "Bitte zuerst die passende Ingestion ausführen."
        )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or _embeddings(),
        persist_directory=str(persist_directory),
    )
