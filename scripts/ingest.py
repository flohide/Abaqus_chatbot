#!/usr/bin/env python
"""Entry Point: Baut die Chroma-Wissensbasis aus den PDFs in data/raw_pdfs/ auf.

Nutzung:
    python scripts/ingest.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.pipeline import run_ingestion  # noqa: E402

console = Console()


def main() -> None:
    console.print(f"[bold cyan]Lese PDFs aus[/bold cyan] {settings.raw_pdf_dir}")
    start = time.time()
    with console.status("[bold green]Parsing, Chunking & Embedding läuft ..."):
        num_chunks = run_ingestion()
    elapsed = time.time() - start
    console.print(
        f"[bold green]Fertig:[/bold green] {num_chunks} Chunks in "
        f"{elapsed:.1f}s in {settings.vectorstore_dir} gespeichert."
    )


if __name__ == "__main__":
    main()
