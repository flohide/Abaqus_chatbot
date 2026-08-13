"""Baut den Parser-Vergleichs-Demo-Korpus: dieselben ~50 Seiten, einmal mit
jedem der vier in docs/PARSER.md evaluierten Parser eingelesen, jeweils in
eine eigene Chroma-Collection unter vectorstore_parser_demo/.

Die Seitenauswahl deckt bewusst unterschiedliche Inhaltstypen ab, die sich
im Parser-Vergleich als unterschiedlich schwierig erwiesen haben:
- Inhaltsverzeichnis-Tabellen (2-spaltig, punktiert, ohne Gitterlinien)
- eine linienlose Mini-Tabelle ("Degrees of Freedom") in Fließtext
- eine vollständige Schritt-für-Schritt-Anleitung (Modellerstellung)
- ein Analyse-Kapitel (Steps/Loads) aus einem zweiten Handbuch

Jede Seite behält ihr echtes, gedrucktes Seiten-Label (siehe loader.py) —
unabhängig davon, was der jeweilige Parser intern für eine Seitenzählung
verwendet -, damit alle vier Collections bei denselben Fragen exakt
vergleichbare, korrekte Quellenangaben liefern.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import settings
from ..rag.vectorstore import build_vectorstore
from .chunker import chunk_pages
from .loader import PageDocument
from .parser_backends import PARSER_BACKENDS

# (Dateiname in data/raw_pdfs/, 0-basierter Seitenindex im Original-PDF)
DEMO_PAGE_SPECS: list[tuple[str, int]] = (
    # Inhaltsverzeichnis mit Tabellenstruktur
    [("Abaqus2017_GETTINGSTARTED.pdf", i) for i in (9, 10)]
    # Degrees-of-Freedom-Mini-Tabelle + Elementtheorie (inkl. Abbildung)
    + [("Abaqus2017_GETTINGSTARTED.pdf", i) for i in range(213, 216)]
    + [("Abaqus2017_GETTINGSTARTED.pdf", i) for i in range(159, 162)]
    # Vollständiges Tutorial-Kapitel "Creating and Analyzing a Simple Model"
    + [("Abaqus2017_GETTINGSTARTED.pdf", i) for i in range(1060, 1090)]
    # Analyse-Kapitel (Steps/Loads) aus einem zweiten Handbuch
    + [("Abaqus2017_ANALYSIS.pdf", i) for i in range(340, 355)]
)

DEMO_PDF_PATH = Path(__file__).resolve().parents[3] / "data" / "parser_demo_pdfs" / "demo_corpus.pdf"


@dataclass
class DemoPageRef:
    source_file: str
    page_number: str  # gedrucktes Label, siehe loader.py
    pdf_page_index: int  # physische Position IM ORIGINAL-HANDBUCH (nicht in der Demo-PDF!)


def build_demo_pdf() -> list[DemoPageRef]:
    """Baut die kombinierte Demo-PDF aus DEMO_PAGE_SPECS und gibt die
    korrekten (Quelldatei, gedrucktes Label, Original-Seitenindex)-Tripel in
    derselben Reihenfolge zurück. Der Seitenindex bezieht sich bewusst auf
    das Original-Handbuch in data/raw_pdfs/ (Ziel der Deep-Links), nicht auf
    die Position innerhalb der Demo-PDF."""
    DEMO_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = pymupdf.open()
    refs: list[DemoPageRef] = []
    open_docs: dict[str, pymupdf.Document] = {}
    try:
        for filename, page_index in DEMO_PAGE_SPECS:
            if filename not in open_docs:
                open_docs[filename] = pymupdf.open(settings.raw_pdf_dir / filename)
            src_doc = open_docs[filename]
            out.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
            label = src_doc[page_index].get_label() or str(page_index + 1)
            refs.append(
                DemoPageRef(source_file=filename, page_number=label, pdf_page_index=page_index + 1)
            )
        out.save(str(DEMO_PDF_PATH))
    finally:
        out.close()
        for doc in open_docs.values():
            doc.close()
    return refs


def run_parser_demo_ingestion(parsers: list[str] | None = None) -> dict[str, int]:
    """Baut für jedes gewählte Parser-Backend eine eigene Collection.
    Gibt {parser_name: anzahl_chunks} zurück."""
    parsers = parsers or list(PARSER_BACKENDS)
    refs = build_demo_pdf()

    result = {}
    for parser_name in parsers:
        parse = PARSER_BACKENDS[parser_name]
        pages_by_index = parse(DEMO_PDF_PATH)

        page_documents = [
            PageDocument(
                text=pages_by_index.get(i, ""),
                source_file=ref.source_file,
                page_number=ref.page_number,
                pdf_page_index=ref.pdf_page_index,
            )
            for i, ref in enumerate(refs, start=1)
            if pages_by_index.get(i, "").strip()
        ]
        chunks = chunk_pages(page_documents)
        build_vectorstore(
            chunks,
            persist_directory=settings.parser_demo_vectorstore_dir,
            collection_name=f"parser_demo_{parser_name}",
        )
        result[parser_name] = len(chunks)
    return result
