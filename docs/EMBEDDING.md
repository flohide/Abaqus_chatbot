# Embedding-Modell-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md),
[`PARSER.md`](PARSER.md) und [`CHUNKING.md`](CHUNKING.md): Sieben
Embedding-Modelle — drei von OpenAI, vier lokal via `sentence-transformers`
von HuggingFace — wurden auf demselben ~53-Seiten-Demo-Korpus quantitativ mit
RAGAS verglichen, um die Wahl von `text-embedding-3-small` als
Produktiv-Embedding empirisch zu begründen.

## Kandidaten

Parser (PyMuPDF4LLM) und Chunking (Header+Recursive 800, Produktiv-Default)
sind für alle sieben Varianten fix — verglichen wird ausschließlich das
Embedding-Modell (siehe
[`ingestion/embedding_demo.py`](../src/rag_manual_bot/ingestion/embedding_demo.py)).

| Modell | Provider | Dimensionen | Bemerkung |
|---|---|---|---|
| `text-embedding-3-small` | OpenAI (API) | 1536 | **Produktiv-Default** (`config.py`) |
| `text-embedding-3-large` | OpenAI (API) | 3072 | teurer, laut OpenAI höhere Retrieval-Qualität |
| `text-embedding-ada-002` | OpenAI (API) | 1536 | Vorgänger-Generation |
| `intfloat/multilingual-e5-large` | HuggingFace (lokal) | 1024 | 560M Parameter, starkes multilingual Retrieval-Modell |
| `BAAI/bge-m3` | HuggingFace (lokal) | 1024 | aktuelles multilingual Modell, für gemischte Sprachen/lange Kontexte |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | HuggingFace (lokal) | 768 | 278M Parameter, etablierte multilingual Baseline |
| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace (lokal) | 384 | 22M Parameter, nur englisch trainiert — bewusst schwacher Kontrast |

Der Korpus ist englisch (Abaqus-Handbücher), Fragen/Antworten sind deutsch
(siehe `DOKUMENTATION.md`, §5.2) — die vier HuggingFace-Kandidaten sind daher
bis auf MiniLM (bewusste Baseline) explizit multilingual-fähige Modelle.

## Architektur: Provider-Abstraktion statt Modellname

Anders als bei Parsern/Chunking-Strategien (und anders als in einer früheren
Fassung dieses Vergleichs, die nur OpenAI-Modelle abdeckte) reicht ein reiner
Modellname hier nicht mehr aus — HuggingFace-Modelle sind keine
`OpenAIEmbeddings`-Instanzen. `ingestion/embedding_models.py::EMBEDDING_BACKENDS`
ist deshalb eine Registry von **Factories** (`() -> Embeddings`), nicht von
Modellnamen-Strings:

- Die drei OpenAI-Einträge bauen `OpenAIEmbeddings` mit dem Produktiv-Client
  (`config.py`); die Modell-ID wird über `_resolve_openai_model()` aufgelöst
  — sie übernimmt automatisch denselben Gateway-Prefix (z. B. `openai/` beim
  FH-SWF-Hub) wie `settings.embedding_model` in `.env`, statt ihn hart zu
  kodieren.
- Die vier HuggingFace-Einträge laden das Modell lokal über
  `langchain_huggingface.HuggingFaceEmbeddings` (siehe
  `requirements-embedding-comparison.txt`) — der Import passiert lazy
  innerhalb der Factory, damit das Modul auch ohne die schweren HF/Torch-
  Abhängigkeiten importierbar bleibt, solange nur OpenAI-Einträge genutzt
  werden (gleiches Prinzip wie beim `semantic`-Chunking-Backend, siehe
  `CHUNKING.md`).

**E5-Prefix-Konvention:** `multilingual-e5-large` wurde mit `"query: "`- bzw.
`"passage: "`-Prefixen vor jedem eingebetteten Text trainiert
([Modellkarte](https://huggingface.co/intfloat/multilingual-e5-large)) — ohne
diese Prefixe fällt seine Retrieval-Qualität spürbar ab, ein leicht
übersehener Stolperstein. `_PrefixedHuggingFaceEmbeddings` in
`embedding_models.py` kapselt das: Queries bekommen `query_prefix`, Chunks
beim Ingestion `passage_prefix` vorangestellt, bevor sie ans zugrunde
liegende Modell gehen. Für die übrigen drei HF-Modelle (keine
Prefix-Konvention) bleiben beide Prefixe leer (No-Op).

**Wichtiger technischer Punkt, unabhängig vom Provider:** Ein Vectorstore
muss mit demselben Embeddings-Objekt **geladen** werden, mit dem er
**aufgebaut** wurde — unterschiedliche Modelle haben unterschiedliche
Vektorraum-Dimensionalität (384 bis 3072) und liefern selbst bei gleicher
Dimension inkompatible Vektorräume. Deshalb tragen sowohl
`rag/vectorstore.py::build_vectorstore()` als auch `load_vectorstore()` einen
optionalen `embeddings`-Parameter (eine fertige `Embeddings`-Instanz, kein
Modellname), und `eval/run.py::_generate_answers` sowie `app.py::_get_chain`
reichen ihn beim Embedding-Vergleich konsistent durch.

## Methodik

Aufbau des Demo-Korpus (pro Modell eine eigene Chroma-Collection
`embedding_demo_<model>` unter `vectorstore_embedding_demo/`):

```bash
pip install -r requirements-embedding-comparison.txt   # nur für die 4 HF-Modelle nötig
python scripts/build_embedding_demo_corpus.py                                  # alle 7 Modelle
python scripts/build_embedding_demo_corpus.py text-embedding-3-small multilingual-e5-large   # Auswahl
```

Die vier HF-Modelle laden beim ersten Aufruf ihre Gewichte vom HuggingFace
Hub herunter (lokal gecacht, insgesamt mehrere GB) und laufen danach
vollständig offline; die Embedding-Berechnung selbst läuft lokal auf der CPU
und ist dadurch spürbar langsamer als der API-Call zu OpenAI.

RAGAS-Evaluation (identisches 12-Fragen-Set wie bei Parser-/Chunking-Vergleich,
fester OpenAI-Richter unabhängig vom getesteten Embedding-Modell, siehe
[`EVALUATION.md`](EVALUATION.md)):

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_embedding_ragas.py
```

Ergebnisse landen in `eval_results/embedding_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored) — Rohdaten pro Frage×Embedding-Modell×LLM sowie eine nach
`(embedding_model, llm_provider)` gruppierte Zusammenfassung.

Live vergleichbar auch im Streamlit-Frontend (`app.py`): alle sieben Modelle
stehen in der Sidebar unter "Vergleich Embedding: …" zur Auswahl (die vier
HF-Einträge sind als "lokal/HF" gekennzeichnet).

## Ergebnisse

Vollständiger Lauf: 7 Embedding-Modelle × 2 LLMs = 14 Kombinationen × 12
Fragen = 168 Instanzen × 5 Metriken = 840 Einzel-Scores, **0 Fehler, 0
fehlende Werte**. Laufzeit: ~17–26 s pro Kombination für die
Antwortgenerierung (bei den vier HF-Modellen inkl. lokalem Laden der
Gewichte aus dem Cache), ~9,0 Min. für die gesamte Metrik-Berechnung
(Concurrency=4). Rohdaten:
[`eval_results/embedding_ragas_raw_20260812_190653.csv`](../eval_results/embedding_ragas_raw_20260812_190653.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_embedding_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Embedding-Modell | Embedding-Provider | Antwort-LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | text-embedding-3-large | OpenAI | OpenAI | 0.888 | 0.806 | 0.871 | 0.917 | 0.617 | **0.820** |
| 2 | text-embedding-3-large | OpenAI | Mistral | 0.888 | 0.676 | 0.871 | 0.917 | 0.565 | **0.783** |
| 3 | multilingual-e5-large | HF (lokal) | OpenAI | 0.765 | 0.796 | 0.958 | 0.792 | 0.539 | **0.770** |
| 4 | text-embedding-ada-002 | OpenAI | OpenAI | 0.912 | 0.735 | 0.753 | 0.875 | 0.515 | **0.758** |
| 5 | text-embedding-3-small | OpenAI | OpenAI | 0.778 | 0.814 | 0.822 | 0.792 | 0.582 | **0.758** |
| 6 | multilingual-e5-large | HF (lokal) | Mistral | 0.800 | 0.665 | 0.958 | 0.792 | 0.537 | **0.750** |
| 7 | minilm-l6-en | HF (lokal) | OpenAI | 0.856 | 0.726 | 0.649 | 0.833 | 0.614 | **0.736** |
| 8 | text-embedding-ada-002 | OpenAI | Mistral | 0.709 | 0.727 | 0.753 | 0.875 | 0.552 | **0.723** |
| 9 | bge-m3 | HF (lokal) | OpenAI | 0.917 | 0.652 | 0.724 | 0.792 | 0.460 | **0.709** |
| 10 | text-embedding-3-small | OpenAI | Mistral | 0.771 | 0.690 | 0.822 | 0.792 | 0.413 | **0.698** |
| 11 | bge-m3 | HF (lokal) | Mistral | 0.749 | 0.592 | 0.724 | 0.792 | 0.518 | **0.675** |
| 12 | paraphrase-multilingual-mpnet | HF (lokal) | OpenAI | 0.704 | 0.672 | 0.681 | 0.750 | 0.539 | **0.669** |
| 13 | minilm-l6-en | HF (lokal) | Mistral | 0.599 | 0.637 | 0.649 | 0.833 | 0.496 | **0.643** |
| 14 | paraphrase-multilingual-mpnet | HF (lokal) | Mistral | 0.658 | 0.521 | 0.681 | 0.750 | 0.465 | **0.615** |

### Retrieval-Qualität pro Embedding-Modell (LLM-unabhängig)

`context_precision`/`context_recall` hängen ausschließlich vom Retrieval ab,
nicht vom antwortenden LLM — fallen wie bei den anderen Vergleichen zwischen
OpenAI- und Mistral-Läufen **identisch** aus:

| Embedding-Modell | Provider | Dimensionen | ContextPrecision | ContextRecall |
|---|---|---|---|---|
| multilingual-e5-large | HF (lokal) | 1024 | **0.958** | 0.792 |
| text-embedding-3-large | OpenAI | 3072 | 0.871 | **0.917** |
| text-embedding-3-small | OpenAI | 1536 | 0.822 | 0.792 |
| text-embedding-ada-002 | OpenAI | 1536 | 0.753 | 0.875 |
| bge-m3 | HF (lokal) | 1024 | 0.724 | 0.792 |
| paraphrase-multilingual-mpnet | HF (lokal) | 768 | 0.681 | 0.750 |
| minilm-l6-en | HF (lokal) | 384 | 0.649 | 0.833 |

### Einordnung

- **`multilingual-e5-large` ist die eigentliche Überraschung dieses
  Vergleichs:** Es erzielt die **höchste ContextPrecision aller sieben
  Modelle** (0.958) — noch vor `text-embedding-3-large` (0.871) — und landet
  im Gesamtranking auf Platz 3, vor `ada-002` und dem Produktiv-Default
  `-3-small`. Ein lokales, kostenloses, offen gewichtetes Modell schlägt hier
  zwei der drei bezahlten OpenAI-Modelle. Wichtiger Vorbehalt: Dieses
  Ergebnis hängt direkt an der korrekten `"query: "`/`"passage: "`-Prefix-
  Konvention, mit der E5-Modelle trainiert wurden
  (`_PrefixedHuggingFaceEmbeddings`, siehe oben) — ohne sie ist ein
  spürbarer Qualitätsabfall zu erwarten; das ist ein leicht zu übersehender
  Stolperstein bei E5-Modellen.
- **`text-embedding-3-large` bleibt Gesamtsieger** (⌀ über beide LLMs:
  0.802) mit der besten ContextRecall (0.917) und der besten Kombination aus
  beiden Retrieval-Metriken gleichzeitig — konsistent mit dem reinen
  OpenAI-Dreiervergleich.
- **`bge-m3` bleibt deutlich hinter seinem Ruf zurück** (⌀ 0.692, mittleres
  Feld) — vermutlicher Grund: BGE-M3 ist für hybrides Retrieval (Dense +
  Sparse + ColBERT-artige Multi-Vektor-Repräsentation) konzipiert;
  `HuggingFaceEmbeddings`/`sentence-transformers` nutzt hier nur die reine
  Dense-Repräsentation. Ein BGE-M3-natives Retrieval-Setup könnte deutlich
  besser abschneiden — dieser Vergleich bewertet also die einfache
  Dense-Einbettung, nicht das volle Potenzial des Modells.
- **`paraphrase-multilingual-mpnet` schneidet am schlechtesten ab** (⌀
  0.642) — plausibel, da es für symmetrische Satzähnlichkeit (Paraphrasing,
  STS) trainiert wurde, nicht für asymmetrisches Query-→-Passage-Retrieval
  wie E5 oder die OpenAI-Modelle.
- **`minilm-l6-en` (bewusste englisch-only-Baseline) schlägt `bge-m3` und
  `paraphrase-mpnet` trotz nur 22M Parametern** (⌀ 0.689 vs. 0.692/0.642 —
  praktisch gleichauf mit bge-m3) — der Korpus ist englisch (Abaqus-
  Handbücher), nur die Fragen sind deutsch; die fehlende Mehrsprachigkeit
  wirkt sich hier also nur auf die Query-Seite aus, nicht auf die
  (englischen) Passagen.
- **API vs. lokal:** Die drei stärksten Modelle verteilen sich auf beide
  Welten (`-3-large` API, `e5-large` lokal) — "lokal" ist also keine
  automatische Qualitätseinbuße. Der Preis für die lokalen Modelle ist
  Rechenzeit statt Geld: HF-Modelle brauchten in diesem Lauf mit
  warmem Cache ähnlich lange wie die API-Calls (~17–26 s je Kombination),
  aber ohne Cache zusätzlich den einmaligen Download (siehe Methodik).
- **OpenAI übertrifft Mistral bei allen 7 Embedding-Modellen im
  Gesamtmittel** — durchgängiges Muster über alle vier Vergleichsdimensionen
  (Parser, Chunking, Embedding) hinweg.
- **FactualCorrectness bleibt über alle Kombinationen hinweg die niedrigste
  Metrik** (0.41–0.62) — typisch für dieses RAGAS-Maß.

**Wichtige Einschränkung:** Wie bei den anderen Vergleichen ist n=12 Fragen
auf einem 53-Seiten-Demo-Korpus eine kleine Stichprobe für 168
LLM-bewertete Instanzen — geeignet, um klare Tendenzen sichtbar zu machen
(den `e5-large`-Precision-Vorsprung, die Schwäche von `paraphrase-mpnet`),
aber nicht, um knappe Unterschiede (z. B. Rang 4 vs. 5, praktisch gleichauf)
als statistisch gesichert zu interpretieren.
