#!/usr/bin/env python
"""Entry Point: RAGAS-Vergleich "Best-of-Breed" vs. Produktiv-Baseline
(siehe docs/BEST_OF_BREED.md).

Kombiniert die jeweils empirisch beste Option aus den vier unabhängigen
Vergleichen zu einer Gesamt-Pipeline (Unstructured + Semantic +
text-embedding-3-large + Rerank) und testet, ob das Kombinieren der
Einzelsieger tatsächlich eine bessere Gesamt-Pipeline ergibt als die
Produktiv-Baseline (PyMuPDF4LLM + Header+Recursive 800 +
text-embedding-3-small + MMR) - beide Pipelines laufen im selben Lauf mit
demselben Richter.

Voraussetzung: die Best-of-Breed-Collection muss bereits existieren
(python scripts/build_best_of_breed_demo.py), ebenso die Baseline-Collection
`chunking_demo_header_recursive_800` (python scripts/build_chunking_demo_corpus.py
header_recursive_800). MISTRAL_API_KEY muss in .env gesetzt sein.

Standard-LLMs (ohne --llms): OpenAI, Mistral (`mistral-small-latest`), GPT-4o.

Nutzung:
    python scripts/evaluate_best_of_breed.py
    python scripts/evaluate_best_of_breed.py --llms openai
    python scripts/evaluate_best_of_breed.py --corpus full   # voller Produktivkorpus statt Demo-Korpus
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

from rag_manual_bot.eval.run import ANSWER_LLM_PROVIDERS, run_best_of_breed_evaluation  # noqa: E402

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
    parser.add_argument("--llms", nargs="+", choices=ANSWER_LLM_PROVIDERS, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--corpus", choices=["demo", "full"], default="demo")
    args = parser.parse_args()

    console.print("[bold cyan]RAGAS-Vergleich: Best-of-Breed vs. Produktiv-Baseline[/bold cyan]")
    console.print(f"LLMs: {args.llms or ANSWER_LLM_PROVIDERS}")
    console.print(f"Korpus: {args.corpus}\n")

    records = run_best_of_breed_evaluation(
        llm_providers=args.llms,
        concurrency=args.concurrency,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        corpus=args.corpus,
    )

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = "_full" if args.corpus == "full" else ""
    raw_path = RESULTS_DIR / f"best_of_breed{suffix}_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    console.print(f"\n[bold green]Rohdaten gespeichert:[/bold green] {raw_path}")

    summary = df.groupby(["pipeline", "llm_provider"])[METRIC_COLUMNS].mean().round(3)
    summary_path = RESULTS_DIR / f"best_of_breed{suffix}_summary_{timestamp}.csv"
    summary.to_csv(summary_path)
    console.print(f"[bold green]Zusammenfassung gespeichert:[/bold green] {summary_path}\n")

    console.print("[bold]Durchschnittliche Scores pro Pipeline × LLM:[/bold]")
    console.print(summary.to_string())

    markdown_path = RESULTS_DIR / f"best_of_breed{suffix}_summary_{timestamp}.md"
    markdown_path.write_text(summary.reset_index().to_markdown(index=False))
    console.print(f"\n[bold green]Markdown-Tabelle gespeichert:[/bold green] {markdown_path}")


if __name__ == "__main__":
    main()
