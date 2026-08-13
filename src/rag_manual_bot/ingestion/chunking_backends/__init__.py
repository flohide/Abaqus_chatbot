"""Registry der vergleichbaren Chunking-Backends (siehe docs/CHUNKING.md).

Jedes Backend exponiert `chunk(pages: list[PageDocument]) -> list[Document]`
mit demselben Metadaten-Schema (`source_file`, `page_number`,
`pdf_page_index`, `section` - leer, wo das Backend keine
Überschriften-Hierarchie kennt).

`header_recursive_800` ist die Produktiv-Chunking-Logik (siehe
`ingestion/chunker.py`); die übrigen Varianten (andere chunk_size, kein
Header-Split, Token- statt Zeichenbasis, Semantic Chunking) existieren nur
für den Chunking-Vergleich (Demo-Korpus, siehe `ingestion/chunking_demo.py`).
"""

from . import header_recursive_backend, recursive_only_backend, semantic_backend, token_backend

CHUNKING_BACKENDS = {
    "header_recursive_400": header_recursive_backend.build(chunk_size=400, chunk_overlap=75),
    "header_recursive_800": header_recursive_backend.build(chunk_size=800, chunk_overlap=150),
    "header_recursive_1600": header_recursive_backend.build(chunk_size=1600, chunk_overlap=300),
    "recursive_only": recursive_only_backend.chunk,
    "token_based": token_backend.chunk,
    "semantic": semantic_backend.chunk,
}
