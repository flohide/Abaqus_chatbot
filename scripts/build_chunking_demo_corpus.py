#!/usr/bin/env python
"""Entry Point: Baut den Chunking-Vergleichs-Demo-Korpus (siehe docs/CHUNKING.md).

Erzeugt eine Chroma-Collection pro registrierter Chunking-Strategie aus
demselben Demo-Korpus, den auch der Parser-Vergleich nutzt (Parser fix:
PyMuPDF4LLM), damit sich die Strategien im Streamlit-Frontend bei
identischen Fragen direkt vergleichen lassen.

Benötigt für die "semantic"-Variante die Extra-Abhängigkeit aus
requirements-chunking-comparison.txt:
    pip install -r requirements-chunking-comparison.txt

Nutzung:
    python scripts/build_chunking_demo_corpus.py                        # alle Strategien
    python scripts/build_chunking_demo_corpus.py header_recursive_800 semantic  # Auswahl
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.chunking_backends import CHUNKING_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.chunking_demo import run_chunking_demo_ingestion  # noqa: E402
from rag_manual_bot.ingestion.parser_demo import DEMO_PAGE_SPECS  # noqa: E402

console = Console()


def main() -> None:
    requested = sys.argv[1:] or list(CHUNKING_BACKENDS)
    unknown = set(requested) - set(CHUNKING_BACKENDS)
    if unknown:
        console.print(f"[bold red]Unbekannte Chunking-Strategien:[/bold red] {unknown}")
        console.print(f"Verfügbar: {list(CHUNKING_BACKENDS)}")
        raise SystemExit(1)

    console.print(f"[bold cyan]Demo-Korpus:[/bold cyan] {len(DEMO_PAGE_SPECS)} Seiten (Parser: pymupdf4llm)")
    console.print(f"[bold cyan]Chunking-Strategien:[/bold cyan] {requested}")
    console.print(f"[bold cyan]Ziel:[/bold cyan] {settings.chunking_demo_vectorstore_dir}\n")

    for chunker_name in requested:
        start = time.time()
        with console.status(f"[bold green]Chunking mit {chunker_name} ..."):
            counts = run_chunking_demo_ingestion(chunkers=[chunker_name])
        elapsed = time.time() - start
        console.print(
            f"[bold green]{chunker_name}:[/bold green] {counts[chunker_name]} Chunks "
            f"in {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
