"""Tests für die Parser×Chunking×Embedding-Auflösung im Frontend (siehe
app.py, docs/BEST_OF_BREED.md) - für den Demo- und den vollen
Produktiv-Korpus. Laufen vollständig offline: geprüft wird nur die
Routing-/Schätzlogik, kein echter Collection-Build (der würde Netzwerk-/
Rechenzeit brauchen, siehe docs/PARSER.md für Docling/Unstructured auf dem
vollen Korpus).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_manual_bot.ingestion.chunking_backends import CHUNKING_BACKENDS
from rag_manual_bot.ingestion.custom_demo import (
    DEFAULT_CHUNKING,
    DEFAULT_EMBEDDING,
    DEFAULT_PARSER,
    estimate_build_seconds,
    is_known_combination,
    is_parser_cached,
)
from rag_manual_bot.ingestion.embedding_models import EMBEDDING_BACKENDS
from rag_manual_bot.ingestion.parser_backends import PARSER_BACKENDS


def test_defaults_match_production_settings():
    assert DEFAULT_PARSER == "pymupdf4llm"
    assert DEFAULT_CHUNKING == "header_recursive_800"
    assert DEFAULT_EMBEDDING == "text-embedding-3-small"


def test_pure_parser_variation_is_known_on_demo_corpus():
    for parser in PARSER_BACKENDS:
        assert is_known_combination(parser, DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="demo")


def test_pure_chunking_variation_is_known_on_demo_corpus():
    for chunking in CHUNKING_BACKENDS:
        assert is_known_combination(DEFAULT_PARSER, chunking, DEFAULT_EMBEDDING, corpus="demo")


def test_pure_embedding_variation_is_known_on_demo_corpus():
    for embedding in EMBEDDING_BACKENDS:
        assert is_known_combination(DEFAULT_PARSER, DEFAULT_CHUNKING, embedding, corpus="demo")


def test_best_of_breed_combination_is_known_on_demo_corpus():
    assert is_known_combination("unstructured", "semantic", "text-embedding-3-large", corpus="demo")


def test_novel_combination_is_not_known():
    assert not is_known_combination("docling", "semantic", "multilingual-e5-large", corpus="demo")
    assert not is_known_combination("unstructured", "token_based", "bge-m3", corpus="demo")


def test_only_production_default_is_known_on_full_corpus():
    assert is_known_combination(DEFAULT_PARSER, DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="full")
    assert not is_known_combination("docling", DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="full")
    assert not is_known_combination(DEFAULT_PARSER, "semantic", DEFAULT_EMBEDDING, corpus="full")
    assert not is_known_combination(DEFAULT_PARSER, DEFAULT_CHUNKING, "text-embedding-3-large", corpus="full")


def test_full_corpus_estimate_is_much_larger_than_demo_estimate():
    demo_s = estimate_build_seconds("pymupdf4llm", "header_recursive_800", "text-embedding-3-small", corpus="demo")
    full_s = estimate_build_seconds("pymupdf4llm", "header_recursive_800", "text-embedding-3-small", corpus="full")
    assert full_s > demo_s * 20  # ~5100 vs. ~53 Seiten


def test_docling_estimate_is_much_larger_than_pymupdf4llm_on_full_corpus():
    fast_s = estimate_build_seconds("pymupdf4llm", DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="full")
    slow_s = estimate_build_seconds("docling", DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="full")
    assert slow_s > fast_s * 20
    assert slow_s > 3600 * 20  # Docling auf vollem Korpus: mehrere Dutzend Stunden


def test_semantic_chunking_increases_estimate():
    fast_s = estimate_build_seconds(DEFAULT_PARSER, "header_recursive_800", DEFAULT_EMBEDDING, corpus="demo")
    semantic_s = estimate_build_seconds(DEFAULT_PARSER, "semantic", DEFAULT_EMBEDDING, corpus="demo")
    assert semantic_s > fast_s


def test_hf_embedding_increases_estimate_vs_openai():
    openai_s = estimate_build_seconds(DEFAULT_PARSER, DEFAULT_CHUNKING, "text-embedding-3-large", corpus="demo")
    hf_s = estimate_build_seconds(DEFAULT_PARSER, DEFAULT_CHUNKING, "multilingual-e5-large", corpus="demo")
    assert hf_s > openai_s


def test_parser_cached_flag_removes_parse_time_from_estimate():
    with_parse = estimate_build_seconds("docling", DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="full", parser_cached=False)
    without_parse = estimate_build_seconds("docling", DEFAULT_CHUNKING, DEFAULT_EMBEDDING, corpus="full", parser_cached=True)
    assert without_parse < with_parse
    # Docling-Parsing dominiert die Schätzung fast vollständig
    assert without_parse < with_parse * 0.05


def test_is_parser_cached_defaults_false_for_unused_parser():
    # "pdfplumber" wird in dieser Testdatei nie tatsächlich aufgerufen (kein
    # echter Parse/Build), sollte also nicht als gecacht gelten.
    assert not is_parser_cached("pdfplumber", "full")
