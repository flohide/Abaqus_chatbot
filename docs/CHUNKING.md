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
OpenAI-Richter, siehe [`EVALUATION.md`](EVALUATION.md)):

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_chunking_ragas.py
```

Ergebnisse landen in `eval_results/chunking_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored) — Rohdaten pro Frage×Chunking×LLM sowie eine nach
`(chunking, llm_provider)` gruppierte Zusammenfassung.

Live vergleichbar auch im Streamlit-Frontend (`app.py`): alle sechs Varianten
stehen in der Sidebar unter "Vergleich Chunking: …" zur Auswahl.

## Ergebnisse

Vollständiger Lauf: 6 Chunking-Strategien × 2 LLMs = 12 Kombinationen × 12
Fragen = 144 Instanzen × 5 Metriken = 720 Einzel-Scores, **0 Fehler, 0
fehlende Werte**. Laufzeit: ~17–24 s pro Kombination für die
Antwortgenerierung, ~9,4 Min. für die gesamte Metrik-Berechnung
(Concurrency=4). Rohdaten:
[`eval_results/chunking_ragas_raw_20260812_172708.csv`](../eval_results/chunking_ragas_raw_20260812_172708.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_chunking_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Chunking-Strategie | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|
| 1 | Semantic | OpenAI | 0.823 | 0.785 | 0.896 | 0.958 | 0.645 | **0.821** |
| 2 | Semantic | Mistral | 0.823 | 0.696 | 0.896 | 0.958 | 0.558 | **0.786** |
| 3 | Recursive-only | OpenAI | 0.860 | 0.735 | 0.906 | 0.833 | 0.552 | **0.777** |
| 4 | Header+Recursive 800 | OpenAI | 0.736 | 0.814 | 0.822 | 0.792 | 0.586 | **0.750** |
| 5 | Recursive-only | Mistral | 0.706 | 0.700 | 0.906 | 0.833 | 0.527 | **0.734** |
| 6 | Header+Recursive 800 | Mistral | 0.789 | 0.736 | 0.822 | 0.792 | 0.532 | **0.734** |
| 7 | Token-basiert | OpenAI | 0.743 | 0.734 | 0.896 | 0.750 | 0.515 | **0.728** |
| 8 | Header+Recursive 1600 | OpenAI | 0.854 | 0.730 | 0.694 | 0.792 | 0.560 | **0.726** |
| 9 | Token-basiert | Mistral | 0.671 | 0.626 | 0.896 | 0.750 | 0.569 | **0.702** |
| 10 | Header+Recursive 1600 | Mistral | 0.767 | 0.644 | 0.694 | 0.792 | 0.598 | **0.699** |
| 11 | Header+Recursive 400 | OpenAI | 0.889 | 0.674 | 0.794 | 0.625 | 0.503 | **0.697** |
| 12 | Header+Recursive 400 | Mistral | 0.549 | 0.666 | 0.794 | 0.625 | 0.531 | **0.633** |

### Retrieval-Qualität pro Chunking-Strategie (LLM-unabhängig)

`context_precision`/`context_recall` hängen ausschließlich vom Retrieval ab,
nicht vom antwortenden LLM — analog zum Parser-Vergleich
([`EVALUATION.md`](EVALUATION.md#retrieval-qualität-pro-parser-llm-unabhängig))
fallen sie hier zwischen OpenAI- und Mistral-Läufen **identisch** aus (gutes
internes Konsistenz-Signal):

| Chunking-Strategie | ContextPrecision | ContextRecall |
|---|---|---|
| Recursive-only | **0.906** | 0.833 |
| Semantic | 0.896 | **0.958** |
| Token-basiert | 0.896 | 0.750 |
| Header+Recursive 800 | 0.822 | 0.792 |
| Header+Recursive 400 | 0.794 | 0.625 |
| Header+Recursive 1600 | 0.694 | 0.792 |

### Einordnung

- **Semantic Chunking gewinnt in dieser Stichprobe klar** (⌀ 0.821 mit
  OpenAI) — vor allem getrieben durch die mit Abstand beste ContextRecall
  (0.958). Das widerspricht der ursprünglichen Einschätzung in
  [`DOKUMENTATION.md`, §3](DOKUMENTATION.md#3-tech-stack--designentscheidungen)
  ("unnötig teuer"): Auf diesem gut strukturierten Handbuch-Demo-Korpus
  liefert die embedding-basierte Breakpoint-Erkennung tatsächlich
  messbar besseres Retrieval als alle festen Chunk-Grenzen. Der Preis dafür
  bleibt real — ein Embedding-Call pro Satz bei der Ingestion (deutlich
  mehr API-Calls als bei den übrigen Strategien) sowie die Abhängigkeit von
  `langchain-experimental`, das laut Upstream im Sunsetting-Modus ist (siehe
  `requirements-chunking-comparison.txt`).
- **Größenvarianten des Hybrid-Chunkers bestätigen den Produktiv-Default
  800 als guten Kompromiss:** `header_recursive_400` hat zwar noch
  akzeptable Precision (0.794), aber die schwächste ContextRecall aller
  sechs Strategien (0.625) — zu kleine Chunks reißen den für die Antwort
  nötigen Kontext auseinander. `header_recursive_1600` hat dagegen die
  **schlechteste ContextPrecision aller Strategien** (0.694) — zu große,
  thematisch verwässerte Chunks schaden der Treffergenauigkeit. `800` liegt
  bei beiden Metriken im Mittelfeld und damit im besten Gleichgewicht.
- **Überraschung: `recursive_only` (keine Header-Awareness) schlägt
  `header_recursive_800` bei ContextPrecision (0.906 vs. 0.822) und
  ContextRecall (0.833 vs. 0.792).** Das war das Vergleichspaar, das gezielt
  den Wert der Header-Struktur isolieren sollte — auf diesem Demo-Korpus
  zahlt sich Header-Awareness beim reinen Retrieval nicht aus. Eine
  plausible Erklärung: Header-Splitting erzeugt stark unterschiedlich große
  Chunks (kurze Abschnitte vs. lange, rekursiv nachgesplittete Abschnitte),
  während `recursive_only` gleichmäßigere Chunk-Größen liefert, was MMR
  (siehe `DOKUMENTATION.md`, §5.1) begünstigen könnte. Header-Awareness
  bleibt aber weiterhin wertvoll für die `section`-Metadaten (Quellenanzeige
  im Chat, siehe §5.4) — das ist in dieser rein retrieval-basierten Metrik
  nicht erfasst.
- **Token- vs. zeichenbasierte Chunkgrenzen machen kaum einen Unterschied:**
  `token_based` (0.896/0.750) liegt nur leicht hinter `recursive_only`
  (0.906/0.833) — beide ohne Header-Awareness, gleiche Chunk-Anzahl-Größenordnung.
  Die isolierte Variable aus diesem Vergleichspaar hat auf diesem Korpus
  also deutlich weniger Einfluss als die Chunk-Größe oder die
  Chunking-Methode (fix vs. semantisch) selbst.
- **OpenAI übertrifft Mistral bei jeder der 6 Chunking-Strategien im
  Gesamtmittel** — durchgängiges Muster, konsistent mit dem Parser-Vergleich
  ([`EVALUATION.md`](EVALUATION.md)).
- **FactualCorrectness ist wie beim Parser-Vergleich über alle Kombinationen
  hinweg die niedrigste Metrik** (0.50–0.65) — typisch für dieses RAGAS-Maß,
  nicht direkt mit den anderen Metriken vergleichbar.

**Wichtige Einschränkung:** Wie beim Parser-Vergleich ist n=12 Fragen auf
einem 53-Seiten-Demo-Korpus eine kleine Stichprobe für 144 LLM-bewertete
Instanzen — geeignet, um klare Tendenzen sichtbar zu machen (z. B. den
ContextRecall-Vorsprung von Semantic Chunking oder die Schwäche von
`header_recursive_400`/`_1600` an den jeweiligen Größenrändern), aber nicht,
um knappe Unterschiede (z. B. Rang 5 vs. 6, praktisch gleichauf) als
statistisch gesichert zu interpretieren.
