"""Löst eine beliebige Parser×Chunking×Embedding-Kombination auf eine
Collection auf (siehe app.py Sidebar) - sowohl für den ~53-Seiten-Demo-Korpus
als auch für den vollen ~5.100-Seiten-Produktiv-Korpus. Für Kombinationen,
die bereits als benannte Vergleichs-Collection existieren (siehe
docs/PARSER.md, docs/CHUNKING.md, docs/EMBEDDING.md, docs/BEST_OF_BREED.md),
wird diese direkt wiederverwendet; für neue, noch nie getestete
Kombinationen wird einmalig eine neue Collection gebaut.

**Wichtig für den vollen Korpus:** Docling/Unstructured (hi_res) sind bei
~5.100 Seiten um Größenordnungen langsamer als bei den 53 Demo-Seiten (siehe
docs/PARSER.md) - ein einzelner Build kann mehrere Stunden bis über einen
Tag dauern und blockiert währenddessen den gesamten Streamlit-Prozess.
`estimate_build_seconds()` liefert eine grobe Vorabschätzung dafür, damit
die Sidebar davor warnen kann, statt den Build blind zu starten.

Retrieval-Strategie und Antwort-LLM sind bewusst **nicht** Teil dieser
Auflösung - beide sind reine Query-Zeit-Entscheidungen (siehe
rag/retrieval_backends.py, rag/llms.py) und wirken unabhängig auf jede
beliebige Collection, brauchen also keine eigene Ingestion.
"""

from pathlib import Path

import pymupdf
from langchain_core.embeddings import Embeddings

from .chunking_backends import CHUNKING_BACKENDS
from .embedding_models import EMBEDDING_BACKENDS
from .loader import PageDocument
from .parser_backends import PARSER_BACKENDS
from .parser_demo import DEMO_PDF_PATH, build_demo_pdf
from ..config import settings
from ..rag.vectorstore import build_vectorstore, load_vectorstore

DEFAULT_PARSER = "pymupdf4llm"
DEFAULT_CHUNKING = "header_recursive_800"
DEFAULT_EMBEDDING = "text-embedding-3-small"

DEMO_PAGE_COUNT = 53
FULL_PAGE_COUNT = 5100  # siehe README.md "Enthaltene Handbücher"

# Kombinationen, die bereits als benannte Vergleichs-Collection existieren
# (jeweils nur eine Dimension weicht vom Produktiv-Default ab, plus der
# Best-of-Breed-Sonderfall) - werden direkt wiederverwendet statt neu gebaut.
# Gilt nur für den Demo-Korpus; für den vollen Korpus gibt es bisher nur die
# eine Produktiv-Collection (siehe _KNOWN_FULL_COLLECTIONS).
_KNOWN_DEMO_COLLECTIONS: dict[tuple[str, str, str], tuple[Path, str]] = {
    ("unstructured", "semantic", "text-embedding-3-large"): (
        settings.embedding_demo_vectorstore_dir,
        "best_of_breed",
    ),
}
for _p in PARSER_BACKENDS:
    _KNOWN_DEMO_COLLECTIONS[(_p, DEFAULT_CHUNKING, DEFAULT_EMBEDDING)] = (
        settings.parser_demo_vectorstore_dir,
        f"parser_demo_{_p}",
    )
for _c in CHUNKING_BACKENDS:
    _KNOWN_DEMO_COLLECTIONS[(DEFAULT_PARSER, _c, DEFAULT_EMBEDDING)] = (
        settings.chunking_demo_vectorstore_dir,
        f"chunking_demo_{_c}",
    )
for _e in EMBEDDING_BACKENDS:
    _KNOWN_DEMO_COLLECTIONS[(DEFAULT_PARSER, DEFAULT_CHUNKING, _e)] = (
        settings.embedding_demo_vectorstore_dir,
        f"embedding_demo_{_e}",
    )

_KNOWN_FULL_COLLECTIONS: dict[tuple[str, str, str], tuple[Path, str]] = {
    (DEFAULT_PARSER, DEFAULT_CHUNKING, DEFAULT_EMBEDDING): (settings.vectorstore_dir, settings.collection_name),
}


def is_known_combination(parser_key: str, chunking_key: str, embedding_key: str, corpus: str = "demo") -> bool:
    """True, wenn für diese Kombination bereits eine benannte Collection
    existiert - andernfalls baut `resolve_collection()` beim ersten Aufruf
    einmalig neu (siehe `estimate_build_seconds()` für eine Zeitschätzung)."""
    known = _KNOWN_FULL_COLLECTIONS if corpus == "full" else _KNOWN_DEMO_COLLECTIONS
    return (parser_key, chunking_key, embedding_key) in known


def _collection_exists(persist_directory: Path, collection_name: str) -> bool:
    if not persist_directory.exists():
        return False
    store = load_vectorstore(persist_directory, collection_name)
    return len(store.get()["ids"]) > 0


def _load_pdf_with_parser(pdf_path: Path, parser_key: str) -> list[PageDocument]:
    """Wie `loader.py::load_pdf`, aber für ein beliebiges Parser-Backend statt
    fest für PyMuPDF4LLM - liest die gedruckten Seiten-Labels weiterhin direkt
    aus dem PDF (siehe `loader.py`), unabhängig von der internen
    Seitenzählung des jeweiligen Parsers."""
    pages_by_index = PARSER_BACKENDS[parser_key](pdf_path)
    doc = pymupdf.open(str(pdf_path))
    try:
        result = []
        for pdf_index_1based, text in pages_by_index.items():
            if not text.strip():
                continue
            printed_label = doc[pdf_index_1based - 1].get_label() or str(pdf_index_1based)
            result.append(
                PageDocument(
                    text=text,
                    source_file=pdf_path.name,
                    page_number=printed_label,
                    pdf_page_index=pdf_index_1based,
                )
            )
        return result
    finally:
        doc.close()


# Parsing ist reine Parser×Korpus-Arbeit, unabhängig von Chunking/Embedding -
# ein Prozess-weiter Cache (Key: (parser_key, corpus)) sorgt dafür, dass ein
# Parser-Wechsel innerhalb derselben App-Sitzung nur einmal bezahlt wird,
# selbst wenn danach mehrere Chunking-/Embedding-Varianten desselben Parsers
# ausprobiert werden - bei Docling/Unstructured auf dem vollen Korpus (siehe
# `estimate_build_seconds()`) ist das der dominante Kostenfaktor. Geht beim
# Neustart des Streamlit-Prozesses verloren (nur In-Memory, nicht auf
# Platte) - die fertige Chroma-Collection selbst bleibt aber wie gewohnt
# persistent (siehe `_collection_exists()`).
_parsed_pages_cache: dict[tuple[str, str], list[PageDocument]] = {}


def is_parser_cached(parser_key: str, corpus: str) -> bool:
    """True, wenn dieser Parser für diesen Korpus in der laufenden Sitzung
    bereits geparst wurde (siehe `_parsed_pages_cache`) - dann entfällt der
    Parse-Schritt beim nächsten Build mit demselben Parser, siehe
    `estimate_build_seconds(..., parser_cached=True)`."""
    return (parser_key, corpus) in _parsed_pages_cache


def _get_parsed_pages(parser_key: str, corpus: str) -> list[PageDocument]:
    cache_key = (parser_key, corpus)
    if cache_key in _parsed_pages_cache:
        return _parsed_pages_cache[cache_key]

    if corpus == "full":
        pdf_paths = sorted(settings.raw_pdf_dir.glob("*.pdf"))
        pages: list[PageDocument] = []
        for pdf_path in pdf_paths:
            pages.extend(_load_pdf_with_parser(pdf_path, parser_key))
    else:
        refs = build_demo_pdf()
        pages_by_index = PARSER_BACKENDS[parser_key](DEMO_PDF_PATH)
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

    _parsed_pages_cache[cache_key] = pages
    return pages


def _build_collection(
    parser_key: str, chunking_key: str, embedding_key: str, corpus: str, persist_directory: Path, collection_name: str
) -> None:
    pages = _get_parsed_pages(parser_key, corpus)
    chunks = CHUNKING_BACKENDS[chunking_key](pages)
    build_vectorstore(
        chunks,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embeddings=EMBEDDING_BACKENDS[embedding_key](),
    )


def resolve_collection(
    parser_key: str, chunking_key: str, embedding_key: str, corpus: str = "demo"
) -> tuple[Path, str, Embeddings | None]:
    """Gibt (persist_directory, collection_name, embeddings) für die gewünschte
    Parser×Chunking×Embedding-Kombination zurück (`corpus`: "demo" oder
    "full"). `embeddings` ist `None`, wenn das Produktiv-Embedding-Modell
    verwendet wird (dann übernimmt `load_vectorstore()` den Produktiv-Default
    automatisch).

    Baut die Collection einmalig neu, falls sie weder als bekannte Collection
    noch schon auf der Platte vorhanden ist. **Für `corpus="full"` mit
    Docling/Unstructured kann das mehrere Stunden dauern** (siehe
    `estimate_build_seconds()`, docs/PARSER.md) - vorher prüfen, nicht blind
    aufrufen."""
    embeddings = None if embedding_key == DEFAULT_EMBEDDING else EMBEDDING_BACKENDS[embedding_key]()
    known_map = _KNOWN_FULL_COLLECTIONS if corpus == "full" else _KNOWN_DEMO_COLLECTIONS
    known = known_map.get((parser_key, chunking_key, embedding_key))

    if known:
        persist_directory, collection_name = known
    elif corpus == "full":
        persist_directory = settings.full_custom_vectorstore_dir
        collection_name = f"full_custom_{parser_key}_{chunking_key}_{embedding_key}"
    else:
        persist_directory = settings.embedding_demo_vectorstore_dir
        collection_name = f"custom_{parser_key}_{chunking_key}_{embedding_key}"

    if not _collection_exists(persist_directory, collection_name):
        _build_collection(parser_key, chunking_key, embedding_key, corpus, persist_directory, collection_name)

    return persist_directory, collection_name, embeddings


# ---------------------------------------------------------------------------
# Grobe Bauzeit-Schätzung, basierend auf gemessenen/dokumentierten Raten
# (siehe docs/PARSER.md, docs/EMBEDDING.md sowie reale Läufe in diesem
# Projekt). Reine Schätzung zur Groborientierung in der UI, keine Garantie -
# reale Laufzeit hängt u. a. von Hardware, Netzwerk und API-Auslastung ab.
# ---------------------------------------------------------------------------
_PARSER_SECONDS_PER_PAGE = {
    "pymupdf4llm": 0.175,  # docs/PARSER.md: 1,05s / 6 Seiten
    "pdfplumber": 0.06,  # docs/PARSER.md: 0,36s / 6 Seiten
    "unstructured": 3.10,  # real gemessen (build_best_of_breed_demo.py): 164,3s / 53 Seiten
    "docling": 22.18,  # docs/PARSER.md: 133,1s / 6 Seiten
}
_HF_EMBEDDING_KEYS = {"multilingual-e5-large", "bge-m3", "paraphrase-multilingual-mpnet", "minilm-l6-en"}
_EMBED_SECONDS_PER_CHUNK_HF = 0.46  # real gemessen: ~80s (abzgl. Parse/Chunk) / 174 Chunks
_EMBED_SECONDS_PER_CHUNK_OPENAI = 0.01  # real gemessen: ~2 Min. / 15.408 Chunks (gebatcht)
_SEMANTIC_CHUNK_SECONDS_PER_PAGE = 0.40  # grobe Hochrechnung aus dem Demo-Korpus-Lauf
_CHUNKS_PER_PAGE = 3.02  # real: 15.408 Chunks / 5.100 Seiten (Produktiv-Ingestion)


def estimate_build_seconds(
    parser_key: str, chunking_key: str, embedding_key: str, corpus: str = "demo", parser_cached: bool | None = None
) -> float:
    """Grobe Schätzung der Bauzeit in Sekunden für eine neue Collection -
    für die UI gedacht (siehe app.py), nicht als exakte Vorhersage.

    `parser_cached`: Wenn `True`, wird der Parse-Schritt nicht mitgerechnet
    (derselbe Parser wurde in dieser Sitzung schon einmal ausgeführt, siehe
    `is_parser_cached()` - bei Docling/Unstructured auf dem vollen Korpus der
    dominante Kostenfaktor). Default `None` prüft `is_parser_cached()`
    automatisch."""
    page_count = FULL_PAGE_COUNT if corpus == "full" else DEMO_PAGE_COUNT
    if parser_cached is None:
        parser_cached = is_parser_cached(parser_key, corpus)
    parse_s = 0.0 if parser_cached else _PARSER_SECONDS_PER_PAGE[parser_key] * page_count
    chunk_s = _SEMANTIC_CHUNK_SECONDS_PER_PAGE * page_count if chunking_key == "semantic" else 1.0
    n_chunks = _CHUNKS_PER_PAGE * page_count
    embed_rate = _EMBED_SECONDS_PER_CHUNK_HF if embedding_key in _HF_EMBEDDING_KEYS else _EMBED_SECONDS_PER_CHUNK_OPENAI
    embed_s = embed_rate * n_chunks
    return parse_s + chunk_s + embed_s
