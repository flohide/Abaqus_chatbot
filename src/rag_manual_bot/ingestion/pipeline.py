"""Orchestriert die gesamte Ingestion: PDFs laden -> chunken -> embedden -> persistieren.

Liest standardmäßig inkrementell ein:
Ein Fingerabdruck-Manifest neben der Collection merkt sich pro PDF einen
SHA-256-Hash. Nur neue oder inhaltlich geänderte Dateien werden neu geparst,
gechunkt und eingebettet; entfernte Dateien werden aus der Collection
gelöscht; unveränderte Dateien werden komplett übersprungen.
"""

import hashlib
import json
from pathlib import Path

from langchain_core.embeddings import Embeddings

from ..config import settings
from ..rag.vectorstore import add_documents_to_vectorstore, build_vectorstore, delete_source_files
from .chunker import chunk_pages
from .loader import load_pdf, load_pdf_directory

MANIFEST_FILENAME = "ingestion_manifest.json"


def _manifest_path(persist_directory: Path) -> Path:
    return persist_directory / MANIFEST_FILENAME


def _file_fingerprint(path: Path) -> str:
    """SHA-256 über den Dateiinhalt - erkennt jede inhaltliche Änderung
    zuverlässig, unabhängig von mtime (die z. B. bei einem frischen
    Checkout oder Kopiervorgang nicht aussagekräftig wäre)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(persist_directory: Path) -> dict[str, str]:
    path = _manifest_path(persist_directory)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_manifest(persist_directory: Path, manifest: dict[str, str]) -> None:
    _manifest_path(persist_directory).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def run_ingestion(force: bool = False, embeddings: Embeddings | None = None) -> int:
    """Baut die Wissensbasis auf und gibt die Anzahl neu geschriebener Chunks zurück.

    `force=True` erzwingt einen kompletten Neuaufbau aller Dateien (z. B. nach
    einem Chunking- oder Embedding-Modell-Wechsel, wo der reine
    Dateiinhalts-Fingerabdruck nichts von der geänderten Verarbeitung weiß).
    Ohne `force` wird beim ersten Lauf (keine Collection vorhanden) ebenfalls
    komplett neu gebaut; bei jedem weiteren Lauf nur das Delta gegenüber dem
    Manifest verarbeitet. `embeddings` überschreibt den Produktiv-Default nur
    für Tests/Sonderfälle (siehe `rag/vectorstore.py`)."""
    pdf_paths = sorted(settings.raw_pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Keine PDF-Dateien in {settings.raw_pdf_dir} gefunden.")
    current_fingerprints = {p.name: _file_fingerprint(p) for p in pdf_paths}

    persist_directory = settings.vectorstore_dir
    if force or not persist_directory.exists():
        pages = load_pdf_directory(settings.raw_pdf_dir)
        chunks = chunk_pages(pages)
        build_vectorstore(chunks, embeddings=embeddings)
        _save_manifest(persist_directory, current_fingerprints)
        return len(chunks)

    previous_fingerprints = _load_manifest(persist_directory)
    changed_or_new = {
        name for name, fingerprint in current_fingerprints.items() if previous_fingerprints.get(name) != fingerprint
    }
    removed = set(previous_fingerprints) - set(current_fingerprints)

    if not changed_or_new and not removed:
        return 0

    delete_source_files(changed_or_new | removed, embeddings=embeddings)

    new_chunks = []
    for name in sorted(changed_or_new):
        new_chunks.extend(chunk_pages(load_pdf(settings.raw_pdf_dir / name)))
    if new_chunks:
        add_documents_to_vectorstore(new_chunks, embeddings=embeddings)

    _save_manifest(persist_directory, current_fingerprints)
    return len(new_chunks)
