"""Tests für den Retrieval-Strategie-Vergleich (siehe docs/RETRIEVAL.md).

Laufen vollständig offline: BM25 (`rank_bm25`) ist eine reine
Python-Implementierung ohne Downloads und wird real getestet. Für "rerank"
wird nur die reine Sortier-Logik (`rerank_by_scores`) getestet, kein echtes
Cross-Encoder-Modell geladen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.documents import Document

from rag_manual_bot.rag.retrieval_backends import (
    RETRIEVAL_BACKENDS,
    _BM25Index,
    reciprocal_rank_fusion,
    rerank_by_scores,
)


def _doc(text: str, page: str) -> Document:
    return Document(page_content=text, metadata={"source_file": "manual.pdf", "page_number": page})


def test_registry_has_four_strategies():
    assert set(RETRIEVAL_BACKENDS) == {"mmr", "similarity", "rerank", "hybrid"}


def test_bm25_index_ranks_matching_document_first():
    docs = [
        _doc("Die Randbedingungen werden im Load-Modul definiert.", "10"),
        _doc("Ein Part wird im Part-Modul erstellt.", "5"),
        _doc("Materialien und Sections gehören zusammen.", "20"),
    ]
    index = _BM25Index(docs)

    top = index.top_k("Wie definiert man Randbedingungen im Load-Modul?", k=2)

    assert len(top) == 2
    assert top[0].metadata["page_number"] == "10"


def test_bm25_index_handles_empty_corpus():
    index = _BM25Index([])
    assert index.top_k("Frage", k=5) == []


def test_reciprocal_rank_fusion_boosts_documents_ranked_high_in_both_lists():
    a = _doc("A", "1")
    b = _doc("B", "2")
    c = _doc("C", "3")

    # b liegt in beiden Rankings vorn -> sollte nach Fusion vorne landen,
    # obwohl a in Ranking 1 auf Platz 1 steht.
    ranking_1 = [a, b, c]
    ranking_2 = [b, c, a]

    fused = reciprocal_rank_fusion([ranking_1, ranking_2], k=3)

    assert fused[0].page_content == "B"
    assert {d.page_content for d in fused} == {"A", "B", "C"}


def test_reciprocal_rank_fusion_deduplicates_same_chunk_across_rankings():
    a = _doc("A", "1")
    a_duplicate_object = _doc("A", "1")  # andere Instanz, gleicher Chunk-Text

    fused = reciprocal_rank_fusion([[a], [a_duplicate_object]], k=5)

    assert len(fused) == 1


def test_reciprocal_rank_fusion_respects_k():
    docs = [_doc(str(i), str(i)) for i in range(5)]
    fused = reciprocal_rank_fusion([docs], k=2)
    assert len(fused) == 2


def test_rerank_by_scores_sorts_descending_and_respects_k():
    docs = [_doc("low", "1"), _doc("high", "2"), _doc("mid", "3")]
    scores = [0.1, 0.9, 0.5]

    top = rerank_by_scores(docs, scores, k=2)

    assert [d.page_content for d in top] == ["high", "mid"]
