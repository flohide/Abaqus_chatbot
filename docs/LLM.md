# Antwort-LLM-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md),
[`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md),
[`EMBEDDING.md`](EMBEDDING.md) und [`RETRIEVAL.md`](RETRIEVAL.md): Bisher
tauchte das Antwort-LLM in jedem der vier Einzelvergleiche nur als gpt-4o-mini auf. Dieser Vergleich zieht die
LLM-Wahl als eigene, isolierte Dimension heraus und ergänzt sie um zwei
weitere Modelle: **GPT-4o** (volle Modellgröße) und **Mistral**
(`mistral-small-latest`).

## Kandidaten

Parser, Chunking, Embedding und Retrieval bleiben für alle drei Varianten
fix (PyMuPDF4LLM, Header+Recursive 800, `text-embedding-3-small`, MMR) —
verglichen wird ausschließlich das Antwort-LLM.

| Modell | Provider | Kosten | Bemerkung |
|---|---|---|---|
| `gpt-4o-mini` | OpenAI (über FH-SWF-Hub) | Hub-Kontingent | **Produktiv-Default** |
| `mistral-small-latest` | Mistral AI | eigener Tier/Kontingent | eigener API-Key nötig |
| `gpt-4o` | OpenAI (über denselben FH-SWF-Hub) | Hub-Kontingent, teurer pro Token als `gpt-4o-mini` | läuft über denselben Client/Key wie `openai` (`rag/llms.py::get_llm("gpt4o")`), nur ein anderer Modellname — isoliert die Modellgröße als einzige Variable gegenüber dem Produktiv-Default |

## Methodik

Kein Demo-Korpus-Build nötig (wie beim Retrieval-Vergleich, siehe
`RETRIEVAL.md`) - direkt RAGAS-Evaluation (identisches 12-Fragen-Set, siehe
[`RAGAS.md`](RAGAS.md)). Da hier drei Antwort-LLMs unterschiedlicher
Anbieter direkt gegeneinander antreten, wird als Richter bewusst ein von
allen Kandidaten unabhängiges Modell eingesetzt (**Claude Sonnet 5**,
Anthropic, `eval/metrics.py::build_judge_metrics(provider="anthropic")`)
statt eines Richters aus derselben Modellfamilie wie zwei der drei
Kandidaten (Self-Preference-Bias-Risiko):

```bash
pip install -r requirements-eval.txt
python scripts/build_chunking_demo_corpus.py header_recursive_800   # falls noch nicht geschehen
python scripts/evaluate_llm_ragas.py --llms openai mistral gpt4o
```

Ergebnisse landen in `eval_results/llm_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored). Im Streamlit-Frontend (`app.py`) ist das Antwort-LLM dagegen
**nicht** frei wählbar — es ist pro Pipeline fest vorgegeben (`gpt-4o-mini`
bei Produktiv, GPT-4o bei Best-of-Breed, siehe `README.md`/`DOKUMENTATION.md`
Abschnitt 12).

## Ergebnisse

Lauf: 3 LLMs × 12 Fragen = 36 Instanzen × 5 Metriken = 180
Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/llm_ragas_raw_20260915_094357.csv`](../eval_results/llm_ragas_raw_20260915_094357.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_llm_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|
| 1 | **OpenAI (gpt-4o-mini)** | **0.927** | 0.788 | 0.708 | 0.792 | 0.492 | **0.741** |
| 2 | GPT-4o | 0.913 | 0.764 | 0.708 | 0.792 | **0.516** | 0.739 |
| 3 | Mistral (mistral-small-latest) | 0.696 | 0.730 | 0.708 | 0.792 | 0.384 | 0.662 |

ContextPrecision/ContextRecall sind über alle drei identisch (0.708/0.792) —
erwartungsgemäß, da Parser/Chunking/Embedding/Retrieval für alle drei LLMs
fix sind und diese beiden Metriken nur vom Retrieval abhängen, nicht vom
antwortenden Modell.
