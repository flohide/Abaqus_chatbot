# RAGAS-Evaluation: Framework, Metriken und Einsatz im Code

Gemeinsame Methodik-Referenz für alle sechs Vergleichsstudien dieses
Projekts ([`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md),
[`EMBEDDING.md`](EMBEDDING.md), [`RETRIEVAL.md`](RETRIEVAL.md),
[`LLM.md`](LLM.md), [`BEST_OF_BREED.md`](BEST_OF_BREED.md)) — die
studienspezifischen Kandidaten, Ergebnistabellen und Einordnungen stehen
dort; hier steht, **was** RAGAS ist, **welche Metriken** eingesetzt werden
und **wie** die Evaluierung im Code funktioniert.

## Was ist RAGAS?

[RAGAS](https://docs.ragas.io/) ("Retrieval-Augmented Generation
Assessment") ist ein Open-Source-Framework zur automatisierten Bewertung
von RAG-Systemen. Statt für jede Antwort einen Menschen bewerten zu lassen,
übernimmt ein **LLM-als-Richter** ("LLM-as-Judge") die Bewertung anhand
einer festen Vier-Tupel-Eingabe pro Frage:

- `user_input` — die gestellte Frage
- `response` — die vom RAG-System generierte Antwort
- `retrieved_contexts` — die vom Retriever gelieferten Chunks
- `reference` — eine (hier: manuell verfasste) Referenzantwort

Jede Metrik prüft daraus einen anderen Aspekt der Antwortqualität und gibt
einen Score in `[0, 1]` zurück (höher = besser). Das macht RAGAS geeignet,
um **viele** Pipeline-Varianten (Parser, Chunking-Strategien,
Embedding-Modelle, Retrieval-Strategien, Antwort-LLMs) auf demselben festen
Fragenset **konsistent und reproduzierbar** gegeneinander zu vergleichen,
ohne für jede Studie erneut manuell durch Dutzende Antworten zu lesen.

**Wichtige Einschränkung, die für alle sechs Studien gilt:** Da ein LLM
selbst die Bewertung übernimmt, ist RAGAS kein objektives, "wahres" Maß —
es misst, wie ein bestimmtes Richter-LLM die Antwortqualität einschätzt.
Für **relative** Vergleiche innerhalb einer Studie (gleicher Richter, gleiche
Fragen) ist das aussagekräftig; absolute Scores sollten nicht unstudienübergreifend
mit unterschiedlichen Richter-LLMs verglichen werden (siehe
[Richter-LLM-Strategie](#richter-llm-strategie) unten).

## Die fünf Metriken

Alle Metriken stammen aus der
[RAGAS-Metrik-Sammlung](https://docs.ragas.io/en/latest/concepts/metrics/)
(`ragas.metrics.collections`):

| Metrik (Code-Klasse) | Braucht Referenz? | Was wird gemessen? |
|---|---|---|
| `Faithfulness` | Nein | Ist die Antwort durch den abgerufenen Kontext gedeckt (keine Halluzination)? |
| `AnswerRelevancy` | Nein | Beantwortet die Antwort tatsächlich die gestellte Frage? |
| `ContextPrecisionWithReference` | Ja | Sind die abgerufenen Chunks relevant für die (referenz-)korrekte Antwort? |
| `ContextRecall` | Ja | Deckt der abgerufene Kontext ab, was für die Referenzantwort nötig wäre? |
| `FactualCorrectness` | Ja | Wie viele Fakten der Antwort stimmen mit der Referenz überein (F1)? |

`ContextPrecision`/`ContextRecall` hängen ausschließlich vom Retrieval ab
(welche Chunks kamen zurück), nicht vom Antwort-LLM — deshalb sind sie in
Studien mit festem Retrieval, aber variierendem Antwort-LLM (z. B.
[`LLM.md`](LLM.md)) über alle Varianten identisch. `FactualCorrectness` ist
erfahrungsgemäß über alle sechs Studien hinweg die strengste, niedrigste
Metrik.

## Das feste Fragenset

Alle sechs Studien laufen auf demselben, festen **12-Fragen-Set**
([`eval/dataset.py`](../src/rag_manual_bot/eval/dataset.py), Auflistung zum
Nachlesen in [`EVAL_FRAGEN.md`](EVAL_FRAGEN.md)) — nur so sind die Scores
zwischen den verglichenen Varianten direkt vergleichbar. Die Fragen decken
bewusst unterschiedliche Inhaltstypen ab (prozedurale Anleitungen,
Tabellen-Lookup, Theorie/Konzepte, spezifische Keywords, TOC-Navigation) und
sind ausschließlich aus dem 53-Seiten-Parser-Vergleichs-Demo-Korpus (siehe
[`PARSER.md`](PARSER.md)) beantwortbar. Die Referenzantworten sind manuell
anhand des tatsächlichen Handbuchtexts verfasst (kein automatisch
generiertes Ground Truth) — Grundlage für `ContextRecall` und
`FactualCorrectness`.

## Richter-LLM-Strategie

Nicht alle sechs Studien nutzen denselben Richter — die Wahl hängt davon ab,
ob das Antwort-LLM innerhalb der Studie konstant bleibt oder selbst die
verglichene Variable ist (`eval/metrics.py::build_judge_metrics(provider=...)`):

- **Fester `gpt-4o-mini`-Richter** (`provider="openai"`, Default) für die
  vier isolierten Dimensions-Vergleiche
  ([`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md),
  [`EMBEDDING.md`](EMBEDDING.md), [`RETRIEVAL.md`](RETRIEVAL.md)): Das
  Antwort-LLM ist dort für alle verglichenen Varianten identisch, ein
  Self-Preference-Bias (LLMs bewerten eigene Formulierungen tendenziell
  günstiger) würde alle Varianten gleichermaßen betreffen und die
  **relative** Rangfolge kaum verzerren.
- **Unabhängiger Richter, Claude Sonnet 5** (`provider="anthropic"`) für
  [`LLM.md`](LLM.md) und [`BEST_OF_BREED.md`](BEST_OF_BREED.md): Dort treten
  OpenAI-Modelle direkt als Kandidaten gegen Mistral an — ein Richter aus der
  OpenAI-Familie wäre hier Kandidat und Richter zugleich. Claude gehört zu
  keiner der bewerteten Antwort-LLM-Familien und ist selbst nie Kandidat.
  Kostenabwägung: Claude Sonnet 5 kostet pro Token ca. 20–25× mehr als
  `gpt-4o-mini` — deshalb bewusst nur für die zwei Studien mit echtem
  Bias-Risiko, nicht global eingesetzt.

Die Embeddings für `AnswerRelevancy` (Kosinus-Ähnlichkeit zwischen
generierter Pseudo-Frage und Original-Frage) bleiben in allen Fällen bei
OpenAI, unabhängig vom Richter-Provider — dort geht es um reine
Vektor-Ähnlichkeit, keine qualitative Bewertung, also kein
Self-Preference-Risiko.

## Einsatz im Code

Pfad: [`src/rag_manual_bot/eval/`](../src/rag_manual_bot/eval/)

- **`dataset.py`** — `EVAL_QUESTIONS: list[EvalQuestion]`, das feste
  12-Fragen-Set (siehe oben).
- **`metrics.py`** — `EvalMetrics`-Dataclass (eine Instanz pro der fünf
  Metrik-Objekte) und `build_judge_metrics(provider, cache_dir=None)`, die
  je nach Provider einen OpenAI- oder Anthropic-Richter-Client aufbaut und
  cached (`ragas.cache.DiskCacheBackend`, Verzeichnis
  `.ragas_cache` bzw. `.ragas_cache_<judge_model>` — pro Richter-Modell
  getrennt, da RAGAS' Cache-Key nicht die Richter-Instanz selbst
  einschließt und ein geteilter Cache sonst stillschweigend alte Urteile
  eines anderen Richters zurückliefern würde). `score_sample(metrics,
  question, answer, contexts, reference)` berechnet alle fünf Metriken für
  eine einzelne Instanz — jede Metrik einzeln in einem eigenen `try/except`,
  damit ein Fehler bei einer Metrik (z. B. ein Parsing-Fehler beim Richter)
  nicht den gesamten Lauf abbricht, sondern nur diese Zelle als `NaN` plus
  `<metrik>_error`-Spalte markiert.
- **`run.py`** — sechs Orchestrierungs-Funktionen, eine pro Vergleichsstudie
  (`run_full_evaluation` (Parser), `run_chunking_evaluation`,
  `run_embedding_evaluation`, `run_retrieval_evaluation`,
  `run_llm_evaluation`, `run_best_of_breed_evaluation`). Jede lädt die
  passende(n) Vectorstore-Collection(s), baut über `build_rag_chain()` die
  RAG-Kette für jede zu vergleichende Variante, lässt sie alle 12 Fragen
  beantworten (`_generate_answers()`) und bewertet anschließend alle
  gesammelten Instanzen parallel (`_score_all()`, `asyncio` mit einem
  `Semaphore` zur Concurrency-Begrenzung). `_invoke_with_retry()` legt sich
  mit Backoff vor `chain.invoke()`, um transiente API-Fehler (v. a.
  429-Rate-Limits bei Mistral) nicht sofort den ganzen Lauf abbrechen zu
  lassen.
- **`_ragas_compat.py`** — Kompatibilitäts-Stub: `ragas` importiert beim
  Laden unbedingt `langchain_community.chat_models.vertexai`
  (`ChatVertexAI`), das im Zuge des "Sunsetting" von `langchain-community`
  entfernt wurde — unabhängig davon, ob Vertex AI je genutzt wird, bricht
  sonst schon der reine `import ragas`. Ein minimaler Platzhalter genügt und
  muss vor jedem `ragas`-Import geladen werden.
- **CLI-Skripte** (`scripts/evaluate_ragas.py`,
  `evaluate_chunking_ragas.py`, `evaluate_embedding_ragas.py`,
  `evaluate_retrieval_ragas.py`, `evaluate_llm_ragas.py`,
  `evaluate_best_of_breed.py`) — dünne Entry-Points, die die passende
  `run_*_evaluation()`-Funktion aufrufen und Ergebnisse als Roh-CSV (eine
  Zeile pro Frage×Variante) sowie aggregierte Zusammenfassung (CSV +
  Markdown, nach der verglichenen Dimension gruppiert) in `eval_results/`
  (git-ignored) speichern.

## Ergebnisse der einzelnen Studien

| Studie | Verglichene Dimension | Dokument |
|---|---|---|
| Parser | PyMuPDF4LLM, pdfplumber, Docling, Unstructured | [`PARSER.md`](PARSER.md) |
| Chunking | 6 Chunking-Strategien | [`CHUNKING.md`](CHUNKING.md) |
| Embedding | 7 Embedding-Modelle | [`EMBEDDING.md`](EMBEDDING.md) |
| Retrieval | 4 Retrieval-Strategien | [`RETRIEVAL.md`](RETRIEVAL.md) |
| Antwort-LLM | 3 LLMs (OpenAI-mini, Mistral, GPT-4o) | [`LLM.md`](LLM.md) |
| Best-of-Breed | Kombination der vier Einzelsieger vs. Produktiv-Baseline | [`BEST_OF_BREED.md`](BEST_OF_BREED.md) |
