#!/usr/bin/env python
"""Entry Point: RAGAS-Evaluation über den Antwort-LLM-Vergleich (siehe docs/LLM.md).

Bewertet Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall und
FactualCorrectness auf der Produktiv-äquivalenten Chunking-Demo-Collection
(Parser fix: PyMuPDF4LLM, Chunking fix: Header+Recursive 800, Embedding fix:
text-embedding-3-small, Retrieval fix: MMR, identisches 12-Fragen-Set wie
bei den anderen Vergleichen) - variiert wird ausschließlich, welches LLM
die Antwort generiert: OpenAI (`gpt-4o-mini`), Mistral
(`mistral-small-latest`) und Qwen (`Qwen2.5-32B-Instruct-AWQ`, offenes
Gewichts-Modell über denselben Hub/Key wie OpenAI, kein separater API-Key
nötig).

Richter-LLM ist für alle Kombinationen fest OpenAI (siehe eval/metrics.py)
- unabhängig vom jeweils getesteten Antwort-LLM.

Voraussetzung: die Chunking-Demo-Collection `chunking_demo_header_recursive_800`
muss bereits existieren (python scripts/build_chunking_demo_corpus.py
header_recursive_800) und MISTRAL_API_KEY muss in .env gesetzt sein.

Nutzung:
    python scripts/evaluate_llm_ragas.py                          # alle 3 LLMs
    python scripts/evaluate_llm_ragas.py --llms openai qwen
    python scripts/evaluate_llm_ragas.py --concurrency 2
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

from rag_manual_bot.eval.run import ANSWER_LLM_PROVIDERS, run_llm_evaluation  # noqa: E402

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
    args = parser.parse_args()

    console.print("[bold cyan]RAGAS-Evaluation: Antwort-LLM-Vergleich[/bold cyan]")
    console.print(f"LLMs: {args.llms or ANSWER_LLM_PROVIDERS}\n")

    records = run_llm_evaluation(
        llm_providers=args.llms,
        concurrency=args.concurrency,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"llm_ragas_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    console.print(f"\n[bold green]Rohdaten gespeichert:[/bold green] {raw_path}")

    summary = df.groupby(["llm_provider"])[METRIC_COLUMNS].mean().round(3)
    summary_path = RESULTS_DIR / f"llm_ragas_summary_{timestamp}.csv"
    summary.to_csv(summary_path)
    console.print(f"[bold green]Zusammenfassung gespeichert:[/bold green] {summary_path}\n")

    console.print("[bold]Durchschnittliche Scores pro LLM:[/bold]")
    console.print(summary.to_string())

    markdown_path = RESULTS_DIR / f"llm_ragas_summary_{timestamp}.md"
    markdown_path.write_text(summary.reset_index().to_markdown(index=False))
    console.print(f"\n[bold green]Markdown-Tabelle gespeichert:[/bold green] {markdown_path}")


if __name__ == "__main__":
    main()
