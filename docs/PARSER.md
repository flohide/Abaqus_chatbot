# PDF-Parser-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md): Vier
gängige Python-PDF-Parser wurden auf demselben 53-Seiten-Demo-Korpus
quantitativ mit RAGAS verglichen, um die Wahl von **PyMuPDF4LLM** als
Produktiv-Parser empirisch zu begründen.

## Kandidaten

| Parser | Version | Ansatz |
|---|---|---|
| **PyMuPDF4LLM** | 1.28.0 | Regelbasiert, nutzt PyMuPDFs Textlayer + Layout-Heuristiken; Tesseract-OCR nur als Fallback für Bildseiten |
| **pdfplumber** | 0.11.10 | Regelbasiert (pdfminer.six-Backend), Textextraktion + optionale Linien-basierte Tabellenerkennung |
| **Docling** (IBM) | 2.118.0 | Deep-Learning-Layoutanalyse (Objekterkennung) + TableFormer-Modell für Tabellenstruktur + RapidOCR |
| **Unstructured** | 0.25.2 (+ `unstructured-inference` 1.6.13) | `hi_res`-Modus: YOLO-Layout-Modell + Tabellenmodell + Tesseract-OCR |

Implementierung: [`ingestion/parser_backends/`](../src/rag_manual_bot/ingestion/parser_backends/),
jedes Backend exponiert `parse(pdf_path: Path) -> dict[int, str]` (Seiten-Index,
1-basiert entsprechend der Position in der PDF-Datei, → extrahierter
Text/Markdown). Die Zuordnung zu Quelldatei und gedrucktem Seiten-Label
übernimmt separat der Demo-Korpus-Aufbau
([`ingestion/parser_demo.py`](../src/rag_manual_bot/ingestion/parser_demo.py)),
damit alle vier Parser unabhängig von ihrer internen Seitenzählung dieselben,
korrekten Zitate liefern.

## Methodik

Aufbau des Demo-Korpus (53 Seiten: TOC-Seiten, eine linienlose
DOF-Mini-Tabelle, ein vollständiges 30-seitiges Tutorial-Kapitel sowie ein
15-seitiges Analyse-Kapitel — pro Parser eine eigene Chroma-Collection
`parser_demo_<name>` unter `vectorstore_parser_demo/`):

```bash
pip install -r requirements-parser-comparison.txt
python scripts/build_parser_demo_corpus.py
```

Systematische, metrikbasierte Auswertung mit dem
[RAGAS](https://docs.ragas.io/)-Framework — alle 4 Parser (Antwort-LLM fest
`gpt-4o-mini`, Produktiv-Default) auf
demselben 53-Seiten-Demo-Korpus, demselben 12-Fragen-Set
([`eval/dataset.py`](../src/rag_manual_bot/eval/dataset.py)) und fünf
Metriken (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
FactualCorrectness), ausgewertet mit einem fest gehaltenen
`gpt-4o-mini`-Richter (Metrik-/Richter-Details, RAGAS-Framework-Hintergrund
und der `_ragas_compat`-Kompatibilitäts-Stub: siehe [`RAGAS.md`](RAGAS.md)):

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_ragas.py
python scripts/evaluate_ragas.py --parsers pymupdf4llm docling   # optional eingeschränkt
```

Ergebnisse landen als Roh-CSV (eine Zeile pro Frage×Parser) und aggregierte
Zusammenfassung (CSV + Markdown) in `eval_results/` (git-ignored). Interaktiv
vergleichbar über [`docs/dashboard.html`](dashboard.html) (Tab "Parser").

## Ergebnisse
Vollständiger Lauf: 4 Parser × 12 Fragen = 48 Instanzen × 5 Metriken =
240 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/ragas_raw_20260913_172832.csv`](../eval_results/ragas_raw_20260913_172832.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Parser | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|
| 1 | Unstructured | 0.898 | 0.784 | 0.864 | 0.875 | 0.585 | **0.801** |
| 2 | PyMuPDF4LLM (Produktiv) | 0.838 | 0.729 | 0.901 | 0.917 | 0.595 | **0.796** |
| 3 | Docling | 0.727 | 0.706 | 0.702 | 0.750 | 0.611 | **0.699** |
| 4 | pdfplumber | 0.857 | 0.525 | 0.801 | 0.625 | 0.411 | **0.644** |

`ContextPrecision`/`ContextRecall` hängen ausschließlich vom Retrieval ab —
PyMuPDF4LLM hat hier mit 0.901/0.917 die besten Werte aller vier Parser,
obwohl Unstructured im Gesamtranking knapp vorn liegt (dort zieht die
bessere Faithfulness/AnswerRelevancy den Ausschlag).

## Fazit

**Entscheidung für dieses Projekt:** PyMuPDF4LLM bleibt der Produktiv-Parser
(siehe [`DOKUMENTATION.md`, Abschnitt 4.1](DOKUMENTATION.md#41-loader-loaderpy)) —
bei praktisch gleichauf liegender Antwortqualität ist es die klar
pragmatischere Wahl gegenüber dem deutlich langsameren und
installationsaufwändigeren Unstructured. Unstructured (`hi_res`) ist trotzdem
Teil der **Best-of-Breed**-Pipeline (siehe [`BEST_OF_BREED.md`](BEST_OF_BREED.md))
— dort wird der Laufzeit-Nachteil für dieses Experiment bewusst zugunsten des
höchsten RAGAS-Einzelscores in Kauf genommen.
