# Best-of-Breed: Kombination der vier Einzelsieger

Abschließendes Experiment zu den vier unabhängigen RAGAS-Vergleichen
([`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md),
[`EMBEDDING.md`](EMBEDDING.md), [`RETRIEVAL.md`](RETRIEVAL.md)): Jeder
Vergleich hat isoliert genau eine Dimension variiert, die anderen drei
blieben auf dem Produktiv-Default fixiert — kombiniert getestet wurden die
vier Einzelsieger bisher nicht. Dieser Vergleich testet, ob das Kombinieren
der Einzelsieger tatsächlich die beste Gesamt-Pipeline ergibt, oder ob sich
Wechselwirkungen zwischen den Dimensionen gegenseitig neutralisieren.

## Die zwei Pipelines

| Dimension | Produktiv-Baseline | Best-of-Breed |
|---|---|---|
| Parser | PyMuPDF4LLM | **Unstructured** (hi_res) |
| Chunking | Header+Recursive 800 | **Semantic** |
| Embedding | text-embedding-3-small | **text-embedding-3-large** |
| Retrieval | MMR | **Rerank** (Cross-Encoder) |


Beide Pipelines laufen im selben RAGAS-Lauf mit demselben Richter und
demselben 12-Fragen-Set, damit der Vergleich sauber ist — die Baseline wird
dafür frisch neu generiert statt alte Zahlen aus `CHUNKING.md`/`RETRIEVAL.md`
wiederzuverwenden. Da hier drei Antwort-LLMs unterschiedlicher Anbieter
direkt gegeneinander antreten, ist der Richter bewusst ein von allen
Kandidaten unabhängiges Modell (**Claude Sonnet 5**, Anthropic,
`eval/metrics.py::build_judge_metrics(provider="anthropic")`) statt eines
Richters aus derselben Modellfamilie wie zwei der drei Kandidaten
(Self-Preference-Bias-Risiko) — anders als bei den vier Einzeldimensions-Vergleichen
(`PARSER.md`, `CHUNKING.md`, `EMBEDDING.md`, `RETRIEVAL.md`), wo das
Antwort-LLM je Studie konstant bleibt und ein fester `gpt-4o-mini`-Richter
daher unproblematisch ist.

**Wichtiger Vorbehalt:** "Beste Einzeloption" bedeutet hier den höchsten
RAGAS-Gesamtdurchschnitt, nicht zwangsläufig die pragmatischste
Produktiv-Wahl — Unstructured ist ~70× langsamer als PyMuPDF4LLM und deutlich
größer in der Installation (siehe `PARSER.md`), was für den vollen
~5.100-Seiten-Produktivkorpus relevant wäre, für dieses Experiment aber
bewusst ignoriert wird.

## Methodik

```bash
python scripts/build_best_of_breed_demo.py   # baut die Best-of-Breed-Collection (dauert mehrere Minuten wegen Unstructured hi_res)
pip install -r requirements-eval.txt
python scripts/evaluate_best_of_breed.py --llms openai mistral gpt4o
```

Ergebnisse landen in `eval_results/best_of_breed_{raw,summary}_<timestamp>.{csv,md}`
(gitignored).

## Ergebnisse

Lauf: 2 Pipelines × 3 LLMs (GPT-4o-mini, Mistral, GPT-4o — siehe
[`LLM.md`](LLM.md)) = 6 Kombinationen × 12 Fragen = 72 Instanzen × 5
Metriken = 360 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/best_of_breed_raw_20260915_094906.csv`](../eval_results/best_of_breed_raw_20260915_094906.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_best_of_breed.py`).

### Gesamtvergleich (Mittelwert über alle 5 Metriken)

| Rang | Pipeline | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|
| 1 | **Best-of-Breed** | **GPT-4o** | **0.992** | 0.914 | 0.785 | **1.000** | **0.624** | **0.863** |
| 2 | **Best-of-Breed** | gpt-4o-mini | 0.896 | **0.944** | 0.785 | **1.000** | 0.479 | 0.821 |
| 3 | **Best-of-Breed** | Mistral | 0.746 | 0.879 | 0.785 | **1.000** | 0.452 | 0.772 |
| 4 | Baseline (Produktiv) |gpt-4o-mini | 0.906 | 0.789 | 0.708 | 0.792 | 0.508 | 0.741 |
| 5 | Baseline (Produktiv) | GPT-4o | 0.837 | 0.774 | 0.708 | 0.792 | 0.522 | 0.727 |
| 6 | Baseline (Produktiv) | Mistral | 0.799 | 0.722 | 0.708 | 0.792 | 0.376 | 0.679 |

**Wichtige Einschränkung:** Diese Auswertung läuft auf dem
53-Seiten-Demo-Korpus, nicht auf dem vollen ~5.100-Seiten-Produktivkorpus.
Die Best-of-Breed-Collection wurde zwar zusätzlich für den vollen Korpus
gebaut (`vectorstore_full_custom/`) und war zeitweise über
`app.py::PIPELINES` im Web-Frontend nutzbar — die Pipeline-Auswahl wurde
seitdem aber entfernt, da `scripts/ingest.py` diese Collection nicht mit
aufbaut und der automatische Nachbau ca. 5 Std. dauert (siehe
`docs/DOKUMENTATION.md`, Abschnitt 12). Eine offizielle RAGAS-Validierung
auf dem vollen Korpus ist zudem bewusst nicht Teil dieser Arbeit
(Laufzeit-/Scope-Gründe).

