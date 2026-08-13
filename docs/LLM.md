# Antwort-LLM-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md),
[`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md),
[`EMBEDDING.md`](EMBEDDING.md) und [`RETRIEVAL.md`](RETRIEVAL.md): Bisher
tauchte das Antwort-LLM in jedem der vier Einzelvergleiche nur als feste
Zweier-Achse (OpenAI vs. Mistral) auf. Dieser Vergleich zieht die
LLM-Wahl als eigene, isolierte Dimension heraus und ergänzt sie um ein
drittes, kostenloses Modell: **Qwen2.5-32B-Instruct-AWQ**, ein offenes
Gewichts-Modell, das der FH-SWF-Hub kostenlos mit anbietet.

## Kandidaten

Parser, Chunking, Embedding und Retrieval bleiben für alle drei Varianten
fix (PyMuPDF4LLM, Header+Recursive 800, `text-embedding-3-small`, MMR) —
verglichen wird ausschließlich das Antwort-LLM.

| Modell | Provider | Kosten | Bemerkung |
|---|---|---|---|
| `gpt-4o-mini` | OpenAI (über FH-SWF-Hub) | Hub-Kontingent | **Produktiv-Default** |
| `mistral-small-latest` | Mistral AI | kostenloser Tier | eigener API-Key nötig |
| `Qwen2.5-32B-Instruct-AWQ` | Alibaba (offene Gewichte, über denselben FH-SWF-Hub) | Hub-Kontingent | **kein eigener API-Key nötig** - läuft über denselben Client/Key wie `openai` (`rag/llms.py::get_llm("qwen")`), nur ein anderer Modellname |

Qwen ist damit die mit Abstand niedrigste Einstiegshürde der drei Modelle:
kein Signup, kein separater Key, keine lokale Rechenleistung - einfach ein
anderer `model`-Parameter an denselben `ChatOpenAI`-Client.

## Methodik

Kein Demo-Korpus-Build nötig (wie beim Retrieval-Vergleich, siehe
`RETRIEVAL.md`) - direkt RAGAS-Evaluation (identisches 12-Fragen-Set, fester
OpenAI-Richter unabhängig vom getesteten Antwort-LLM, siehe
[`EVALUATION.md`](EVALUATION.md)):

```bash
pip install -r requirements-eval.txt
python scripts/build_chunking_demo_corpus.py header_recursive_800   # falls noch nicht geschehen
python scripts/evaluate_llm_ragas.py
```

Ergebnisse landen in `eval_results/llm_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored). Live vergleichbar auch im Streamlit-Frontend (`app.py`): alle
drei Modelle stehen in der Sidebar unter "LLM" zur Auswahl.

## Ergebnisse

Vollständiger Lauf: 3 LLMs × 12 Fragen = 36 Instanzen × 5 Metriken = 180
Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Laufzeit: 20–74 s pro LLM für
die Antwortgenerierung (Qwen deutlich langsamer als OpenAI/Mistral, siehe
Einordnung), ~2,1 Min. für die gesamte Metrik-Berechnung (Concurrency=4).
Rohdaten:
[`eval_results/llm_ragas_raw_20260813_060743.csv`](../eval_results/llm_ragas_raw_20260813_060743.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_llm_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen2.5-32B (kostenlos)** | **0.826** | 0.698 | 0.822 | 0.792 | **0.641** | **0.756** |
| 2 | OpenAI (gpt-4o-mini) | 0.736 | **0.807** | 0.822 | 0.792 | 0.582 | 0.748 |
| 3 | Mistral (mistral-small-latest) | 0.736 | 0.683 | 0.822 | 0.792 | 0.534 | 0.713 |

ContextPrecision/ContextRecall sind über alle drei identisch (0.822/0.792) —
erwartungsgemäß, da Parser/Chunking/Embedding/Retrieval für alle drei LLMs
fix sind und diese beiden Metriken nur vom Retrieval abhängen, nicht vom
antwortenden Modell.

### Einordnung

- **Qwen2.5-32B gewinnt überraschend das Gesamtranking** (⌀ 0.756), knapp
  vor OpenAI (0.748) und deutlich vor Mistral (0.713) — obwohl es das
  einzige der drei Modelle ohne eigenen API-Key/Signup ist und einfach über
  denselben Hub-Client wie OpenAI läuft. Treiber sind die mit Abstand beste
  Faithfulness (0.826 vs. 0.736 für beide anderen — identisch, ein
  auffälliger Gleichstand) und die beste FactualCorrectness (0.641) aller
  drei Modelle.
- **OpenAI bleibt bei AnswerRelevancy klar vorn** (0.807 vs. 0.698 Qwen,
  0.683 Mistral) — dasselbe Muster wie in allen vier vorherigen Vergleichen
  (`EVALUATION.md`, `CHUNKING.md`, `EMBEDDING.md`, `RETRIEVAL.md`): OpenAI
  beantwortet tendenziell fokussierter die tatsächlich gestellte Frage,
  andere Modelle schweifen etwas mehr ab oder beantworten unvollständig.
- **Qwen ist spürbar langsamer** (73,9 s für 12 Fragen vs. 30,1 s OpenAI /
  20,1 s Mistral) — bei diesem konkreten Hub-Hosting rund 2,5–3,7× länger
  pro Anfrage. Für einen interaktiven Chat relevant, für Batch-/
  Offline-Auswertungen unerheblich.
- **Kosten-Nutzen ist hier eindeutig:** Qwen liefert die höchste
  Antwortqualität dieses Vergleichs, ohne zusätzlichen API-Key oder
  Kosten — der einzige Nachteil ist Latenz. Für den Produktivbetrieb wäre
  Qwen auf Basis dieser Zahlen eine ernstzunehmende Alternative zum
  aktuellen Default `gpt-4o-mini`, insbesondere wenn Faithfulness (keine
  Halluzination) wichtiger ist als Antwortgeschwindigkeit.

**Wichtige Einschränkung:** Wie bei den anderen Vergleichen ist n=12 Fragen
eine kleine Stichprobe für 36 LLM-bewertete Instanzen — der Abstand
Qwen vs. OpenAI (0.756 vs. 0.748) ist knapp und sollte nicht als
statistisch gesichert gelten, der Abstand zu Mistral (0.713) ist deutlicher
und konsistent mit dem in allen anderen Vergleichen beobachteten Muster,
dass OpenAI Mistral bei AnswerRelevancy übertrifft.
