"""Tests für das inkrementelle Ingestion-Update (siehe `ingestion/pipeline.py`).
Laufen vollständig offline mit
deterministischen Fake-Embeddings (kein OpenAI-Call) und winzigen, on-the-fly
erzeugten PDFs statt der echten Handbücher.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pymupdf
from langchain_core.embeddings import Embeddings

from rag_manual_bot.config import settings
from rag_manual_bot.ingestion.pipeline import MANIFEST_FILENAME, run_ingestion
from rag_manual_bot.rag.vectorstore import load_vectorstore


class FakeEmbeddings(Embeddings):
    """Deterministische, offline lauffähige Embeddings für Tests (kein OpenAI-Call)."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in digest[:16]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _write_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text)
        doc.save(str(path))
    finally:
        doc.close()


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    raw_pdf_dir = tmp_path / "raw_pdfs"
    vectorstore_dir = tmp_path / "vectorstore"
    raw_pdf_dir.mkdir()
    monkeypatch.setattr(settings, "raw_pdf_dir", raw_pdf_dir)
    monkeypatch.setattr(settings, "vectorstore_dir", vectorstore_dir)
    return raw_pdf_dir, vectorstore_dir


def _collection_source_files(vectorstore_dir: Path) -> set[str]:
    store = load_vectorstore(vectorstore_dir, embeddings=FakeEmbeddings())
    return {m["source_file"] for m in store.get()["metadatas"]}


def test_first_run_builds_everything(tmp_path, monkeypatch):
    raw_pdf_dir, vectorstore_dir = _setup(tmp_path, monkeypatch)
    _write_pdf(raw_pdf_dir / "a.pdf", "Inhalt von Dokument A.")
    _write_pdf(raw_pdf_dir / "b.pdf", "Inhalt von Dokument B.")

    num_chunks = run_ingestion(embeddings=FakeEmbeddings())

    assert num_chunks == 2
    assert _collection_source_files(vectorstore_dir) == {"a.pdf", "b.pdf"}
    manifest = json.loads((vectorstore_dir / MANIFEST_FILENAME).read_text())
    assert set(manifest) == {"a.pdf", "b.pdf"}


def test_second_run_without_changes_is_a_noop(tmp_path, monkeypatch):
    raw_pdf_dir, vectorstore_dir = _setup(tmp_path, monkeypatch)
    _write_pdf(raw_pdf_dir / "a.pdf", "Inhalt von Dokument A.")
    run_ingestion(embeddings=FakeEmbeddings())

    num_chunks = run_ingestion(embeddings=FakeEmbeddings())

    assert num_chunks == 0
    assert _collection_source_files(vectorstore_dir) == {"a.pdf"}


def test_changed_file_is_reprocessed_others_untouched(tmp_path, monkeypatch):
    raw_pdf_dir, vectorstore_dir = _setup(tmp_path, monkeypatch)
    _write_pdf(raw_pdf_dir / "a.pdf", "Ursprünglicher Inhalt von A.")
    _write_pdf(raw_pdf_dir / "b.pdf", "Inhalt von Dokument B.")
    run_ingestion(embeddings=FakeEmbeddings())

    _write_pdf(raw_pdf_dir / "a.pdf", "Komplett geänderter Inhalt von A.")
    num_chunks = run_ingestion(embeddings=FakeEmbeddings())

    assert num_chunks == 1
    store = load_vectorstore(vectorstore_dir, embeddings=FakeEmbeddings())
    result = store.get()
    assert {m["source_file"] for m in result["metadatas"]} == {"a.pdf", "b.pdf"}
    a_texts = [doc for doc, m in zip(result["documents"], result["metadatas"]) if m["source_file"] == "a.pdf"]
    assert any("Komplett geändert" in t for t in a_texts)
    assert not any("Ursprünglicher" in t for t in a_texts)


def test_removed_file_is_deleted_from_collection(tmp_path, monkeypatch):
    raw_pdf_dir, vectorstore_dir = _setup(tmp_path, monkeypatch)
    _write_pdf(raw_pdf_dir / "a.pdf", "Inhalt von Dokument A.")
    _write_pdf(raw_pdf_dir / "b.pdf", "Inhalt von Dokument B.")
    run_ingestion(embeddings=FakeEmbeddings())

    (raw_pdf_dir / "b.pdf").unlink()
    num_chunks = run_ingestion(embeddings=FakeEmbeddings())

    assert num_chunks == 0
    assert _collection_source_files(vectorstore_dir) == {"a.pdf"}
    manifest = json.loads((vectorstore_dir / MANIFEST_FILENAME).read_text())
    assert set(manifest) == {"a.pdf"}


def test_force_rebuilds_even_without_changes(tmp_path, monkeypatch):
    raw_pdf_dir, vectorstore_dir = _setup(tmp_path, monkeypatch)
    _write_pdf(raw_pdf_dir / "a.pdf", "Inhalt von Dokument A.")
    run_ingestion(embeddings=FakeEmbeddings())

    num_chunks = run_ingestion(force=True, embeddings=FakeEmbeddings())

    assert num_chunks == 1
    assert _collection_source_files(vectorstore_dir) == {"a.pdf"}
