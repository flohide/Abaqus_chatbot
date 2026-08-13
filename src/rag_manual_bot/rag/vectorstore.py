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
    )


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
    persist_directory = persist_directory or settings.vectorstore_dir
    collection_name = collection_name or settings.collection_name
    persist_directory.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings or _embeddings(),
        persist_directory=str(persist_directory),
    )
    existing_ids = store.get()["ids"]
    if existing_ids:
        store.delete(ids=existing_ids)

    batch_size = 200
    for i in range(0, len(documents), batch_size):
        store.add_documents(documents[i : i + batch_size])
    return store


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
