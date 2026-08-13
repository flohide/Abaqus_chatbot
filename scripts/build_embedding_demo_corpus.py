#!/usr/bin/env python
"""Entry Point: Baut den Embedding-Modell-Vergleichs-Demo-Korpus (siehe docs/EMBEDDING.md).

Erzeugt eine Chroma-Collection pro Embedding-Modell aus demselben
Demo-Korpus, den auch Parser- und Chunking-Vergleich nutzen (Parser fix:
PyMuPDF4LLM, Chunking fix: Produktiv-Default Header+Recursive 800) - isoliert
wird ausschließlich das Embedding-Modell.

Drei Modelle laufen über die Produktiv-API/-Gateway (OpenAI), vier lokal via
`sentence-transformers` (kein API-Key nötig, aber die Modellgewichte werden
beim ersten Aufruf heruntergeladen - je nach Modell mehrere hundert MB bis
~2 GB, siehe docs/EMBEDDING.md). Dafür zusätzlich installieren:
    pip install -r requirements-embedding-comparison.txt

Nutzung:
    python scripts/build_embedding_demo_corpus.py                                    # alle 7 Modelle
    python scripts/build_embedding_demo_corpus.py text-embedding-3-small multilingual-e5-large
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.embedding_demo import run_embedding_demo_ingestion  # noqa: E402
from rag_manual_bot.ingestion.embedding_models import EMBEDDING_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.parser_demo import DEMO_PAGE_SPECS  # noqa: E402

console = Console()


def main() -> None:
    requested = sys.argv[1:] or list(EMBEDDING_BACKENDS)
    unknown = set(requested) - set(EMBEDDING_BACKENDS)
    if unknown:
        console.print(f"[bold red]Unbekannte Embedding-Modelle:[/bold red] {unknown}")
        console.print(f"Verfügbar: {list(EMBEDDING_BACKENDS)}")
        raise SystemExit(1)

    console.print(f"[bold cyan]Demo-Korpus:[/bold cyan] {len(DEMO_PAGE_SPECS)} Seiten (Parser: pymupdf4llm, Chunking: header_recursive_800)")
    console.print(f"[bold cyan]Embedding-Modelle:[/bold cyan] {requested}")
    console.print(f"[bold cyan]Ziel:[/bold cyan] {settings.embedding_demo_vectorstore_dir}\n")

    for model_name in requested:
        start = time.time()
        with console.status(f"[bold green]Embedde mit {model_name} ..."):
            counts = run_embedding_demo_ingestion(models=[model_name])
        elapsed = time.time() - start
        console.print(
            f"[bold green]{model_name}:[/bold green] {counts[model_name]} Chunks "
            f"in {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
