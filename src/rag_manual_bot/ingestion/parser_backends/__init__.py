"""Registry der vier vergleichbaren PDF-Parser-Backends (siehe docs/PARSER.md).

Jedes Backend exponiert `parse(pdf_path: Path) -> dict[int, str]`, das für
jede Seite (1-basiert, entsprechend der Position in der PDF-Datei) den
extrahierten Text/Markdown liefert. Die eigentliche Zuordnung zu Quelldatei
und gedrucktem Seiten-Label übernimmt separat der Demo-Korpus-Aufbau
(siehe ingestion/parser_demo.py), damit alle vier Parser unabhängig von
ihrer internen Seitenzählung dieselben, korrekten Zitate liefern.
"""

from . import docling_backend, pdfplumber_backend, pymupdf4llm_backend, unstructured_backend

PARSER_BACKENDS = {
    "pymupdf4llm": pymupdf4llm_backend.parse,
    "pdfplumber": pdfplumber_backend.parse,
    "docling": docling_backend.parse,
    "unstructured": unstructured_backend.parse,
}
