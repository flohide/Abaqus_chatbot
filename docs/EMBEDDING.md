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
fester `gpt-4o-mini`-Richter unabhängig vom getesteten Embedding-Modell, siehe
[`RAGAS.md`](RAGAS.md)):

```bash
pip install -r requirements-eval.txt
python scripts/evaluate_embedding_ragas.py
```

Ergebnisse landen in `eval_results/embedding_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored) — Rohdaten pro Frage×Embedding-Modell (Antwort-LLM fest auf
`gpt-4o-mini`) sowie eine nach `embedding_model` gruppierte Zusammenfassung.

Interaktiv vergleichbar über [`docs/dashboard.html`](dashboard.html) (Tab
"Embedding"). Im Streamlit-Frontend (`app.py`) sind die sieben
Embedding-Modelle dagegen nicht live wählbar — dort läuft ausschließlich
die feste Produktiv-Pipeline, siehe `README.md`.

## Ergebnisse

Vollständiger Lauf: 7 Embedding-Modelle × 12 Fragen = 84 Instanzen × 5
Metriken = 420 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/embedding_ragas_raw_20260914_044025.csv`](../eval_results/embedding_ragas_raw_20260914_044025.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_embedding_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Embedding-Modell | Provider | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|
| 1 | text-embedding-3-small (Produktiv) | OpenAI | 0.840 | 0.729 | 0.901 | 0.917 | 0.596 | **0.797** |
| 2 | multilingual-e5-large | HF (lokal) | 0.882 | 0.736 | 0.882 | 0.792 | 0.622 | **0.783** |
| 3 | text-embedding-ada-002 | OpenAI | 0.809 | 0.741 | 0.781 | 0.875 | 0.672 | **0.776** |
| 4 | text-embedding-3-large | OpenAI | 0.799 | 0.817 | 0.850 | 0.792 | 0.547 | **0.761** |
| 5 | bge-m3 | HF (lokal) | 0.810 | 0.720 | 0.801 | 0.792 | 0.596 | **0.744** |
| 6 | minilm-l6-en | HF (lokal) | 0.758 | 0.575 | 0.493 | 0.583 | 0.565 | **0.595** |
| 7 | paraphrase-multilingual-mpnet | HF (lokal) | 0.622 | 0.520 | 0.667 | 0.667 | 0.397 | **0.575** |