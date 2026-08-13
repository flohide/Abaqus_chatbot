#!/usr/bin/env python
"""Entry Point: Baut den Parser-Vergleichs-Demo-Korpus (siehe docs/PARSER.md).

Erzeugt vier separate Chroma-Collections (eine pro Parser) aus denselben
~50 Demo-Seiten, damit sich die Parser im Streamlit-Frontend bei identischen
Fragen direkt vergleichen lassen.

⚠️ Docling und Unstructured (hi_res) sind sehr langsam (siehe docs/PARSER.md)
— der volle Lauf über alle vier Parser kann mehrere Dutzend Minuten dauern.

Benötigt die Extra-Abhängigkeiten aus requirements-parser-comparison.txt:
    pip install -r requirements-parser-comparison.txt

Nutzung:
    python scripts/build_parser_demo_corpus.py                  # alle 4 Parser
    python scripts/build_parser_demo_corpus.py pymupdf4llm docling  # Auswahl
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.parser_backends import PARSER_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.parser_demo import DEMO_PAGE_SPECS, run_parser_demo_ingestion  # noqa: E402

console = Console()


def main() -> None:
    requested = sys.argv[1:] or list(PARSER_BACKENDS)
    unknown = set(requested) - set(PARSER_BACKENDS)
    if unknown:
        console.print(f"[bold red]Unbekannte Parser:[/bold red] {unknown}")
        console.print(f"Verfügbar: {list(PARSER_BACKENDS)}")
        raise SystemExit(1)

    console.print(f"[bold cyan]Demo-Korpus:[/bold cyan] {len(DEMO_PAGE_SPECS)} Seiten")
    console.print(f"[bold cyan]Parser:[/bold cyan] {requested}")
    console.print(f"[bold cyan]Ziel:[/bold cyan] {settings.parser_demo_vectorstore_dir}\n")

    for parser_name in requested:
        start = time.time()
        with console.status(f"[bold green]Parsing mit {parser_name} ..."):
            counts = run_parser_demo_ingestion(parsers=[parser_name])
        elapsed = time.time() - start
        console.print(
            f"[bold green]{parser_name}:[/bold green] {counts[parser_name]} Chunks "
            f"in {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
