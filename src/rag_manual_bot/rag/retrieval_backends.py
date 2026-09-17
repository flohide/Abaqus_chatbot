"""Registry der vergleichbaren Retrieval-Strategien (siehe docs/RETRIEVAL.md).

Anders als beim Parser-/Chunking-/Embedding-Vergleich braucht diese
Dimension **keinen eigenen Demo-Korpus** — alle vier Strategien arbeiten auf
demselben, bereits vorhandenen Vectorstore (Retrieval ist eine reine
Query-Zeit-Entscheidung, keine Ingestion-Entscheidung). Jeder Eintrag ist
eine Factory `(vectorstore) -> Callable[[str], list[Document]]`.

- `mmr` - Produktiv-Baseline (siehe `retriever.py`, MMR über Chroma).
- `similarity` - naive Baseline ohne MMR-Diversität, isoliert deren Wert.
- `rerank` - MMR holt `fetch_k` Kandidaten, ein Cross-Encoder sortiert sie
  nach tatsächlicher Query-Relevanz neu, die Top-`k` werden zurückgegeben.
- `hybrid` - BM25 (Keyword-Suche) + Dense-Similarity-Suche, fusioniert per
  Reciprocal Rank Fusion (RRF) - relevant für exakte Keyword-Treffer (z. B.
  `*STEP`), die Embeddings manchmal schlechter abbilden als Volltextsuche.

`rank_bm25` (für `hybrid`) und `sentence-transformers` (für `rerank`) sind
lazy importierte Abhängigkeiten (Teil von requirements.txt, da `rerank` von
der Best-of-Breed-Pipeline in app.py zur Query-Zeit gebraucht wird) - das
Modul bleibt ohne sie importierbar, solange nur `mmr`/`similarity` genutzt
werden.
"""

import re
from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import settings
from .retriever import build_retriever

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class _BM25Index:
    """Dünner Wrapper um `rank_bm25.BM25Okapi` mit eigener Tokenisierung
    (statt über `langchain_community.retrievers.BM25Retriever` zu gehen, das
    laut Upstream im Sunsetting-Modus ist, siehe `CHUNKING.md`)."""

    def __init__(self, documents: list[Document]):
        self._documents = documents
        if not documents:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi  # BM25Okapi([]) wirft ZeroDivisionError auf leerem Korpus

        self._bm25 = BM25Okapi([_tokenize(d.page_content) for d in documents])

    def top_k(self, query: str, k: int) -> list[Document]:
        if not self._documents:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._documents, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int, rrf_k: int = 60) -> list[Document]:
    """Fusioniert mehrere Rankings zu einem: Score(d) = Summe 1/(rrf_k + rank).
    Dedupliziert über den Chunk-Text (stabiler als Objektidentität, da
    verschiedene Retriever unabhängige `Document`-Instanzen für denselben
    Chunk liefern können)."""
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            doc_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ranked_keys[:k]]


def rerank_by_scores(candidates: list[Document], scores, k: int) -> list[Document]:
    """Sortiert `candidates` absteigend nach `scores` (gleiche Länge, parallel
    indiziert) und gibt die Top-`k` zurück - als reine Funktion ausgelagert,
    damit sie ohne ein echtes Cross-Encoder-Modell testbar ist."""
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]


def _build_mmr(vectorstore: Chroma) -> Callable[[str], list[Document]]:
    return build_retriever(vectorstore).invoke


def _build_similarity(vectorstore: Chroma) -> Callable[[str], list[Document]]:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": settings.retrieval_k})
    return retriever.invoke


def _build_rerank(vectorstore: Chroma) -> Callable[[str], list[Document]]:
    from sentence_transformers import CrossEncoder

    cross_encoder = CrossEncoder(RERANK_MODEL)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retrieval_fetch_k, "fetch_k": settings.retrieval_fetch_k * 2},
    )

    def retrieve(query: str) -> list[Document]:
        candidates = base_retriever.invoke(query)
        if not candidates:
            return []
        scores = cross_encoder.predict([(query, doc.page_content) for doc in candidates])
        return rerank_by_scores(candidates, scores, settings.retrieval_k)

    return retrieve


def _build_hybrid(vectorstore: Chroma) -> Callable[[str], list[Document]]:
    raw = vectorstore.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=meta) for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    bm25 = _BM25Index(documents)
    dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": settings.retrieval_fetch_k})

    def retrieve(query: str) -> list[Document]:
        bm25_docs = bm25.top_k(query, settings.retrieval_fetch_k)
        dense_docs = dense_retriever.invoke(query)
        return reciprocal_rank_fusion([bm25_docs, dense_docs], k=settings.retrieval_k)

    return retrieve


RETRIEVAL_BACKENDS: dict[str, Callable[[Chroma], Callable[[str], list[Document]]]] = {
    "mmr": _build_mmr,
    "similarity": _build_similarity,
    "rerank": _build_rerank,
    "hybrid": _build_hybrid,
}
