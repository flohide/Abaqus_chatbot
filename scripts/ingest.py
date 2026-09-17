#!/usr/bin/env python
"""Entry Point: Baut die Chroma-Wissensbasis aus den PDFs in data/raw_pdfs/ auf.

Liest standardmäßig inkrementell ein (siehe `ingestion/pipeline.py`): Nur
neue/geänderte PDFs werden neu geparst,
gechunkt und eingebettet, unveränderte Dateien werden übersprungen.

Nutzung:
    python scripts/ingest.py            # inkrementell (Delta gegenüber dem Manifest)
    python scripts/ingest.py --force    # kompletter Neuaufbau aller Dateien
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.pipeline import run_ingestion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Kompletten Neuaufbau erzwingen statt nur das Delta einzulesen"
    )
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold cyan]Lese PDFs aus[/bold cyan] {settings.raw_pdf_dir}")
    start = time.time()
    status = "[bold green]Kompletter Neuaufbau läuft ..." if args.force else "[bold green]Prüfe auf Änderungen ..."
    with console.status(status):
        num_chunks = run_ingestion(force=args.force)
    elapsed = time.time() - start

    if num_chunks == 0:
        console.print(f"[bold green]Fertig:[/bold green] keine Änderungen erkannt, nichts zu tun ({elapsed:.1f}s).")
    else:
        console.print(
            f"[bold green]Fertig:[/bold green] {num_chunks} Chunks in "
            f"{elapsed:.1f}s in {settings.vectorstore_dir} gespeichert."
        )


if __name__ == "__main__":
    main()
