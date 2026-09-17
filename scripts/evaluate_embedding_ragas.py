#!/usr/bin/env python
"""Entry Point: RAGAS-Evaluation über die volle Embedding-Modell × 2-LLM-Matrix.

Bewertet Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall und
FactualCorrectness auf dem Embedding-Vergleichs-Demo-Korpus (Parser fix:
PyMuPDF4LLM, Chunking fix: Produktiv-Default Header+Recursive 800,
identisches 12-Fragen-Set wie bei Parser-/Chunking-Vergleich, siehe
docs/EMBEDDING.md).

Richter-LLM ist für alle Kombinationen fest OpenAI (siehe eval/metrics.py)
- unabhängig vom jeweils getesteten Embedding-Modell.

Voraussetzung: die Embedding-Demo-Collections müssen bereits existieren
(python scripts/build_embedding_demo_corpus.py).

Nutzung:
    python scripts/evaluate_embedding_ragas.py                                # volle Matrix (7 Modelle)
    python scripts/evaluate_embedding_ragas.py --models text-embedding-3-small multilingual-e5-large
    python scripts/evaluate_embedding_ragas.py --llms openai
    python scripts/evaluate_embedding_ragas.py --concurrency 2
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

from rag_manual_bot.eval.run import EMBEDDING_COLLECTIONS, LLM_PROVIDERS, run_embedding_evaluation  # noqa: E402

console = Console()

METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
]

RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval_results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(EMBEDDING_COLLECTIONS), default=None)
    parser.add_argument("--llms", nargs="+", choices=LLM_PROVIDERS, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    console.print("[bold cyan]RAGAS-Evaluation: Embedding-Modell-Vergleich[/bold cyan]")
    console.print(f"Embedding-Modelle: {args.models or list(EMBEDDING_COLLECTIONS)}")
    console.print(f"LLMs: {args.llms or LLM_PROVIDERS}\n")

    records = run_embedding_evaluation(
        models=args.models,
        llm_providers=args.llms,
        concurrency=args.concurrency,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"embedding_ragas_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    console.print(f"\n[bold green]Rohdaten gespeichert:[/bold green] {raw_path}")

    summary = df.groupby(["embedding_model", "llm_provider"])[METRIC_COLUMNS].mean().round(3)
    summary_path = RESULTS_DIR / f"embedding_ragas_summary_{timestamp}.csv"
    summary.to_csv(summary_path)
    console.print(f"[bold green]Zusammenfassung gespeichert:[/bold green] {summary_path}\n")

    console.print("[bold]Durchschnittliche Scores pro Kombination:[/bold]")
    console.print(summary.to_string())

    markdown_path = RESULTS_DIR / f"embedding_ragas_summary_{timestamp}.md"
    markdown_path.write_text(summary.reset_index().to_markdown(index=False))
    console.print(f"\n[bold green]Markdown-Tabelle gespeichert:[/bold green] {markdown_path}")


if __name__ == "__main__":
    main()
