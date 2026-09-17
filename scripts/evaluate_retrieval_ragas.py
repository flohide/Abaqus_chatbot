#!/usr/bin/env python
"""Entry Point: RAGAS-Evaluation über die volle Retrieval-Strategie × 2-LLM-Matrix.

Bewertet Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall und
FactualCorrectness auf der Produktiv-äquivalenten Chunking-Demo-Collection
(Parser fix: PyMuPDF4LLM, Chunking fix: Header+Recursive 800, Embedding fix:
text-embedding-3-small, identisches 12-Fragen-Set wie bei den anderen
Vergleichen, siehe docs/RETRIEVAL.md). Anders als beim Parser-/Chunking-/
Embedding-Vergleich ist **kein separater Demo-Korpus-Build nötig** -
Retrieval-Strategien wirken nur zur Query-Zeit.

Richter-LLM ist für alle Kombinationen fest OpenAI (siehe eval/metrics.py).

Voraussetzung: die Chunking-Demo-Collection `chunking_demo_header_recursive_800`
muss bereits existieren (python scripts/build_chunking_demo_corpus.py);
"rerank"/"hybrid" brauchen `rank_bm25`/`sentence-transformers`, die bereits
Teil von requirements.txt sind.

Nutzung:
    python scripts/evaluate_retrieval_ragas.py                          # volle Matrix
    python scripts/evaluate_retrieval_ragas.py --strategies mmr rerank
    python scripts/evaluate_retrieval_ragas.py --llms openai
    python scripts/evaluate_retrieval_ragas.py --concurrency 2
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

from rag_manual_bot.eval.run import LLM_PROVIDERS, run_retrieval_evaluation  # noqa: E402
from rag_manual_bot.rag.retrieval_backends import RETRIEVAL_BACKENDS  # noqa: E402

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
    parser.add_argument("--strategies", nargs="+", choices=list(RETRIEVAL_BACKENDS), default=None)
    parser.add_argument("--llms", nargs="+", choices=LLM_PROVIDERS, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    console.print("[bold cyan]RAGAS-Evaluation: Retrieval-Strategie-Vergleich[/bold cyan]")
    console.print(f"Retrieval-Strategien: {args.strategies or list(RETRIEVAL_BACKENDS)}")
    console.print(f"LLMs: {args.llms or LLM_PROVIDERS}\n")

    records = run_retrieval_evaluation(
        strategies=args.strategies,
        llm_providers=args.llms,
        concurrency=args.concurrency,
        on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
    )

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"retrieval_ragas_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    console.print(f"\n[bold green]Rohdaten gespeichert:[/bold green] {raw_path}")

    summary = df.groupby(["retrieval_strategy", "llm_provider"])[METRIC_COLUMNS].mean().round(3)
    summary_path = RESULTS_DIR / f"retrieval_ragas_summary_{timestamp}.csv"
    summary.to_csv(summary_path)
    console.print(f"[bold green]Zusammenfassung gespeichert:[/bold green] {summary_path}\n")

    console.print("[bold]Durchschnittliche Scores pro Kombination:[/bold]")
    console.print(summary.to_string())

    markdown_path = RESULTS_DIR / f"retrieval_ragas_summary_{timestamp}.md"
    markdown_path.write_text(summary.reset_index().to_markdown(index=False))
    console.print(f"\n[bold green]Markdown-Tabelle gespeichert:[/bold green] {markdown_path}")


if __name__ == "__main__":
    main()
