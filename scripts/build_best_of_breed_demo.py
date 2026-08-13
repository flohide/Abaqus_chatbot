#!/usr/bin/env python
"""Entry Point: Baut die "Best-of-Breed"-Demo-Collection (siehe docs/BEST_OF_BREED.md).

Kombiniert die jeweils empirisch beste Option aus den vier unabhängigen
RAGAS-Vergleichen (siehe docs/PARSER.md, docs/CHUNKING.md, docs/EMBEDDING.md) —
Parser: Unstructured (hi_res), Chunking: Semantic, Embedding:
text-embedding-3-large. Die vierte Dimension (Retrieval: Rerank) ist eine
reine Query-Zeit-Entscheidung und wird erst in der Evaluation gewählt
(siehe scripts/evaluate_best_of_breed.py).

Diese Kombination wurde in keinem der vier Einzelvergleiche getestet — dort
variierte jeweils nur eine Dimension, die anderen drei blieben auf dem
Produktiv-Default. Ob das Kombinieren der Einzelsieger tatsächlich die beste
Gesamt-Pipeline ergibt (statt sich durch Wechselwirkungen zu neutralisieren),
ist genau die Frage, die dieser Vergleich beantwortet.

⚠️ Unstructured (hi_res) ist ~70× langsamer als PyMuPDF4LLM (siehe
docs/PARSER.md) — der Aufbau auf dem ~53-Seiten-Demo-Korpus dauert mehrere
Minuten.

Nutzung:
    python scripts/build_best_of_breed_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402

from rag_manual_bot.config import settings  # noqa: E402
from rag_manual_bot.ingestion.chunking_backends import CHUNKING_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.embedding_models import EMBEDDING_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.loader import PageDocument  # noqa: E402
from rag_manual_bot.ingestion.parser_backends import PARSER_BACKENDS  # noqa: E402
from rag_manual_bot.ingestion.parser_demo import DEMO_PDF_PATH, build_demo_pdf  # noqa: E402
from rag_manual_bot.rag.vectorstore import build_vectorstore  # noqa: E402

console = Console()

BEST_OF_BREED_COLLECTION = "best_of_breed"


def main() -> None:
    console.print(
        "[bold cyan]Best-of-Breed Demo-Korpus:[/bold cyan] "
        "Unstructured (Parser) + Semantic (Chunking) + text-embedding-3-large (Embedding)"
    )
    refs = build_demo_pdf()

    t0 = time.time()
    with console.status("[bold green]Parse mit Unstructured (hi_res) — das dauert mehrere Minuten ..."):
        pages_by_index = PARSER_BACKENDS["unstructured"](DEMO_PDF_PATH)
    console.print(f"Parsing abgeschlossen in {time.time() - t0:.1f}s")

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
    console.print(f"{len(pages)} nicht-leere Seiten.")

    t0 = time.time()
    with console.status("[bold green]Chunking mit Semantic Chunker ..."):
        chunks = CHUNKING_BACKENDS["semantic"](pages)
    console.print(f"{len(chunks)} Chunks erzeugt in {time.time() - t0:.1f}s")

    t0 = time.time()
    with console.status("[bold green]Embedde mit text-embedding-3-large + baue Vectorstore ..."):
        build_vectorstore(
            chunks,
            persist_directory=settings.embedding_demo_vectorstore_dir,
            collection_name=BEST_OF_BREED_COLLECTION,
            embeddings=EMBEDDING_BACKENDS["text-embedding-3-large"](),
        )
    console.print(f"[bold green]Fertig[/bold green] in {time.time() - t0:.1f}s: Collection "
                  f"'{BEST_OF_BREED_COLLECTION}' unter {settings.embedding_demo_vectorstore_dir}")


if __name__ == "__main__":
    main()
