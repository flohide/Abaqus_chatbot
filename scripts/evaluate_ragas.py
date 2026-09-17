#!/usr/bin/env python
"""Entry Point: RAGAS-Evaluation über die volle 4-Parser × 2-LLM-Matrix.

Bewertet Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall und
FactualCorrectness auf dem ~53-Seiten-Demo-Korpus (identisches Fragenset für
alle 8 Kombinationen, siehe docs/PARSER.md, Framework-Hintergrund in docs/RAGAS.md).

Richter-LLM ist für alle Kombinationen fest OpenAI (siehe eval/metrics.py)
— vermeidet Self-Preference-Bias beim Modellvergleich.

Voraussetzung: die 4 Parser-Demo-Collections müssen bereits existieren
(python scripts/build_parser_demo_corpus.py).

Nutzung:
    python scripts/evaluate_ragas.py                              # volle Matrix
    python scripts/evaluate_ragas.py --parsers pymupdf4llm docling
    python scripts/evaluate_ragas.py --llms openai
    python scripts/evaluate_ragas.py --concurrency 2
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

from rag_manual_bot.eval.run import LLM_PROVIDERS, PARSER_COLLECTIONS, run_full_evaluation  # noqa: E402

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
    parser.add_argument("--parsers", nargs="+", choices=list(PARSER_COLLECTIONS), default=None)
    parser.add_argument("--llms", nargs="+", choices=LLM_PROVIDERS, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    console.print("[bold cyan]RAGAS-Evaluation[/bold cyan]")
    console.print(f"Parser: {args.parsers or list(PARSER_COLLECTIONS)}")
    console.print(f"LLMs: {args.llms or LLM_PROVIDERS}\n")

    records = run_full_evaluation(
        parsers=args.parsers,
        llm_providers=args.llms,
        concurrency=args.concurrency,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"ragas_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    console.print(f"\n[bold green]Rohdaten gespeichert:[/bold green] {raw_path}")

    summary = df.groupby(["parser", "llm_provider"])[METRIC_COLUMNS].mean().round(3)
    summary_path = RESULTS_DIR / f"ragas_summary_{timestamp}.csv"
    summary.to_csv(summary_path)
    console.print(f"[bold green]Zusammenfassung gespeichert:[/bold green] {summary_path}\n")

    console.print("[bold]Durchschnittliche Scores pro Kombination:[/bold]")
    console.print(summary.to_string())

    markdown_path = RESULTS_DIR / f"ragas_summary_{timestamp}.md"
    markdown_path.write_text(summary.reset_index().to_markdown(index=False))
    console.print(f"\n[bold green]Markdown-Tabelle gespeichert:[/bold green] {markdown_path}")


if __name__ == "__main__":
    main()
