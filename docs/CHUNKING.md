# Chunking-Techniken-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md) und
[`PARSER.md`](PARSER.md): Sechs Chunking-Strategien wurden auf demselben
~53-Seiten-Demo-Korpus (siehe [`PARSER.md`, Methodik](PARSER.md#methodik))
quantitativ mit RAGAS verglichen, um die Wahl des
**Header+Recursive-Hybrid-Chunkers** (chunk_size=800) als Produktiv-Strategie
empirisch zu begründen.

## Kandidaten

Der Parser ist für alle sechs Varianten fix auf **PyMuPDF4LLM**
(Produktiv-Parser) gesetzt — verglichen wird ausschließlich die
Chunking-Strategie, nicht zusätzlich der Parser (siehe
[`ingestion/chunking_demo.py`](../src/rag_manual_bot/ingestion/chunking_demo.py)).

| Backend | Ansatz | Header-aware | Grenzeinheit |
|---|---|---|---|
| `header_recursive_400` | MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter-Fallback | ✅ | Zeichen (400) |
| `header_recursive_800` | wie oben — **Produktiv-Default** (`ingestion/chunker.py`) | ✅ | Zeichen (800) |
| `header_recursive_1600` | wie oben | ✅ | Zeichen (1600) |
| `recursive_only` | RecursiveCharacterTextSplitter über den gesamten Seitentext | ❌ | Zeichen (800) |
| `token_based` | wie `recursive_only`, aber `tiktoken`-Encoder (`cl100k_base`) statt Zeichenanzahl | ❌ | Tokens (200) |
| `semantic` | `langchain_experimental.text_splitter.SemanticChunker` — embedding-basierte Breakpoints zwischen Sätzen | ❌ | semantische Distanz (Perzentil-Schwellwert) |

Implementierung: [`ingestion/chunking_backends/`](../src/rag_manual_bot/ingestion/chunking_backends/),
jedes Backend exponiert `chunk(pages: list[PageDocument]) -> list[Document]`
mit identischem Metadaten-Schema (`source_file`, `page_number`,
`pdf_page_index`, `section` — leer bei den drei nicht header-aware
Backends).

**Vergleichspaare, die jeweils genau eine Variable isolieren:**
- `header_recursive_800` vs. `recursive_only` → Wert der
  Header-Awareness/Abschnittsstruktur.
- `recursive_only` vs. `token_based` → Zeichen- vs. Token-basierte
  Chunkgrenzen.
- `header_recursive_400`/`_800`/`_1600` → Einfluss der Chunkgröße bei
  ansonsten identischer Strategie.
- `recursive_only`/`token_based` vs. `semantic` → feste Grenzen vs.
  inhaltsbasierte (embedding-getriebene) Grenzen — genau die in
  [`DOKUMENTATION.md`, §3](DOKUMENTATION.md#3-tech-stack--designentscheidungen)
  als "unnötig teuer" verworfene Alternative.

## Methodik

Aufbau des Demo-Korpus (pro Chunking-Strategie eine eigene Chroma-Collection
`chunking_demo_<name>` unter `vectorstore_chunking_demo/`):

```bash
pip install -r requirements-chunking-comparison.txt   # nur für "semantic" nötig
python scripts/build_chunking_demo_corpus.py
```

RAGAS-Evaluation (identisches 12-Fragen-Set wie beim Parser-Vergleich, siehe
[`eval/dataset.py`](../src/rag_manual_bot/eval/dataset.py), fester
`gpt-4o-mini`-Richter, siehe [`RAGAS.md`](RAGAS.md)):

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_chunking_ragas.py
```

Ergebnisse landen in `eval_results/chunking_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored) — Rohdaten pro Frage×Chunking-Strategie (Antwort-LLM fest auf
`gpt-4o-mini`) sowie eine nach `chunking`
gruppierte Zusammenfassung.

Interaktiv vergleichbar über [`docs/dashboard.html`](dashboard.html) (Tab
"Chunking"). Im Streamlit-Frontend (`app.py`) sind die sechs
Chunking-Strategien dagegen nicht live wählbar — dort läuft ausschließlich
die feste Produktiv-Pipeline, siehe `README.md`.

## Ergebnisse

Vollständiger Lauf: 6 Chunking-Strategien × 12 Fragen = 72 Instanzen × 5
Metriken = 360 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/chunking_ragas_raw_20260914_043622.csv`](../eval_results/chunking_ragas_raw_20260914_043622.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_chunking_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Chunking-Strategie | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|
| 1 | Semantic | 0.797 | 0.778 | 0.896 | 0.958 | 0.642 | **0.814** |
| 2 | Header+Recursive 800 (Produktiv) | 0.854 | 0.735 | 0.901 | 0.917 | 0.607 | **0.803** |
| 3 | Recursive-only | 0.850 | 0.735 | 0.906 | 0.833 | 0.538 | **0.772** |
| 4 | Header+Recursive 1600 | 0.771 | 0.780 | 0.797 | 0.875 | 0.581 | **0.761** |
| 5 | Token-basiert | 0.731 | 0.731 | 0.896 | 0.750 | 0.504 | **0.722** |
| 6 | Header+Recursive 400 | 0.815 | 0.674 | 0.758 | 0.667 | 0.528 | **0.688** |

