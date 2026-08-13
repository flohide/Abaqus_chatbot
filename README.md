# Abaqus Handbuch-Chatbot (RAG)

Ein Retrieval-Augmented-Generation-System, das Fragen zur Bedienung der
FEA-Software Abaqus auf Basis der offiziellen PDF-Handbücher beantwortet —
mit Seiten-genauem Source-Tracking und Terminal-Chat.

Ausführliche technische Dokumentation (Architektur, Designentscheidungen,
Verifikationsergebnisse): [`docs/DOKUMENTATION.md`](docs/DOKUMENTATION.md)

PDF-Parser-Vergleich (PyMuPDF4LLM vs. Docling vs. pdfplumber vs. Unstructured):
[`docs/PARSER.md`](docs/PARSER.md)

Chunking-Techniken-Vergleich (Header+Recursive in 3 Größen, Recursive-only,
Token-basiert, Semantic Chunking) inkl. RAGAS-Evaluation:
[`docs/CHUNKING.md`](docs/CHUNKING.md)

Embedding-Modell-Vergleich (3 OpenAI-Modelle + 4 lokale HuggingFace-Modelle)
inkl. RAGAS-Evaluation: [`docs/EMBEDDING.md`](docs/EMBEDDING.md)

Retrieval-Strategie-Vergleich (MMR vs. Similarity vs. Cross-Encoder-Rerank
vs. BM25-Hybrid) inkl. RAGAS-Evaluation: [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md)

Best-of-Breed: Kombination der vier Einzelsieger gegen die
Produktiv-Baseline: [`docs/BEST_OF_BREED.md`](docs/BEST_OF_BREED.md)

Antwort-LLM-Vergleich (OpenAI vs. Mistral vs. Qwen2.5-32B, kostenlos über
denselben Hub) inkl. RAGAS-Evaluation: [`docs/LLM.md`](docs/LLM.md)

Quantitative RAGAS-Evaluation (4 Parser × 2 LLMs, 5 Metriken, 480 Scores):
[`docs/EVALUATION.md`](docs/EVALUATION.md)

## Architekturentscheidungen

| Bereich | Wahl | Begründung |
|---|---|---|
| Vektor-DB | [Chroma](https://www.trychroma.com/) (lokal, persistent) | Keine Server-Infrastruktur nötig, gute LangChain-Integration, Metadaten-Filterung möglich. |
| PDF-Parser | [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) | Wandelt PDFs seitenweise in Markdown um und erhält dabei Überschriften, Tabellen und Seitenzahlen — ideal für strukturierte Handbücher. |
| Chunking | Markdown-Header-Splitting + Recursive-Splitting (Hybrid) | Hält zusammengehörige Abschnitte (Kapitel/Unterkapitel) zusammen; nur überlange Abschnitte werden zusätzlich zeichenbasiert gesplittet. |
| Embeddings | `text-embedding-3-small` | Guter Kompromiss aus Kosten und Qualität. |
| LLM | `gpt-4o-mini` (konfigurierbar) | Günstig, schnell, für Handbuch-Q&A ausreichend präzise. |
| RAG-Kette | Reine [LCEL](https://python.langchain.com/docs/concepts/lcel/)-Runnables | LangChain ≥1.0 hat `langchain.chains` (`create_retrieval_chain` etc.) entfernt; die Kette wird hier transparent aus Runnables zusammengesetzt statt einen Compat-Layer zu nutzen. |
| Retrieval | MMR (Max Marginal Relevance), k=5 | Reduziert redundante Chunks aus derselben Seite/Sektion. |
| Memory | Einfache Nachrichtenliste in der Chat-Session | Ausreichend für ein Single-User-Tool, kein Session-Store nötig. |
| Frontend | [Streamlit](https://streamlit.io/) (`st.chat_message`/`st.chat_input`) | Klassische Chat-Optik mit wenig Code, Sidebar für Dropdown-Steuerung (LLM/Wissensbasis). |
| LLM-Vergleich | `ChatOpenAI` (Hub) vs. `ChatMistralAI` austauschbar | Gleiche RAG-Kette, gleiche Wissensbasis, nur das LLM wechselt — sauberer Vergleich der Antwortqualität zweier Modelle. |
| Parser-Vergleich | 4 separate Chroma-Collections auf ~50-Seiten-Demo-Korpus | Docling/Unstructured sind 70–130× langsamer als PyMuPDF4LLM (siehe `docs/PARSER.md`) — voller Korpus für alle 4 Parser wäre nicht praktikabel; Demo-Korpus macht den Vergleich trotzdem live im Frontend möglich. |

## Projektstruktur

```
Abaqus_chatbot/
├── data/
│   ├── raw_pdfs/                    # Quell-PDFs Produktiv-Korpus (git-ignored)
│   └── parser_demo_pdfs/            # Generierter ~50-Seiten-Demo-Korpus (git-ignored)
├── vectorstore/                      # Produktiv-Vectorstore (git-ignored)
├── vectorstore_parser_demo/          # 4 Parser-Vergleichs-Collections (git-ignored)
├── vectorstore_chunking_demo/        # 6 Chunking-Vergleichs-Collections (git-ignored)
├── vectorstore_embedding_demo/       # 7 Embedding-Modell-Vergleichs-Collections (git-ignored)
├── src/rag_manual_bot/
│   ├── config.py                     # pydantic-settings (.env)
│   ├── ingestion/                    # PDF -> Markdown -> Chunks -> Vectorstore
│   │   ├── loader.py                 # Produktiv-Loader (PyMuPDF4LLM + Page-Labels)
│   │   ├── chunker.py                # Produktiv-Chunking (Wrapper um header_recursive_backend)
│   │   ├── pipeline.py
│   │   ├── parser_demo.py            # Demo-Korpus-Aufbau für Parser-Vergleich
│   │   ├── parser_backends/          # 4 austauschbare Parser (siehe docs/PARSER.md)
│   │   │   ├── pymupdf4llm_backend.py
│   │   │   ├── docling_backend.py
│   │   │   ├── pdfplumber_backend.py
│   │   │   └── unstructured_backend.py
│   │   ├── chunking_demo.py          # Demo-Korpus-Aufbau für Chunking-Vergleich
│   │   ├── chunking_backends/        # 6 austauschbare Chunking-Strategien (siehe docs/CHUNKING.md)
│   │   │   ├── header_recursive_backend.py
│   │   │   ├── recursive_only_backend.py
│   │   │   ├── token_backend.py
│   │   │   └── semantic_backend.py
│   │   ├── embedding_demo.py         # Demo-Korpus-Aufbau für Embedding-Modell-Vergleich
│   │   ├── embedding_models.py       # 7 Embedding-Modelle: 3 OpenAI + 4 HF lokal (siehe docs/EMBEDDING.md)
│   │   └── custom_demo.py            # app.py-Sidebar: beliebige Parser×Chunking×Embedding-Kombination auflösen/on-demand bauen
│   ├── rag/                          # Retrieval + LCEL-Kette
│   │   ├── vectorstore.py
│   │   ├── retriever.py              # Produktiv-Retriever (MMR)
│   │   ├── retrieval_backends.py     # 4 austauschbare Retrieval-Strategien (siehe docs/RETRIEVAL.md)
│   │   ├── prompts.py
│   │   ├── chain.py
│   │   └── llms.py                   # LLM-Provider-Auswahl (OpenAI/Mistral)
│   ├── eval/                         # RAGAS-Evaluation (siehe docs/EVALUATION.md, docs/CHUNKING.md)
│   │   ├── _ragas_compat.py          # Kompatibilitäts-Stub für ragas + LangChain >=1.0
│   │   ├── dataset.py                # 12 Eval-Fragen + Referenzantworten
│   │   ├── metrics.py                # 5 RAGAS-Metriken, fester OpenAI-Richter
│   │   └── run.py                    # Orchestriert alle vier Vergleichs-Matrizen
│   ├── citations.py                  # Klickbare file://-Links zu Original-PDFs
│   └── cli/
│       └── chat.py                   # Terminal-Chat (rich)
├── scripts/
│   ├── ingest.py                         # Produktiv-Ingestion-Entry-Point
│   ├── build_parser_demo_corpus.py       # Demo-Korpus für Parser-Vergleich aufbauen
│   ├── build_chunking_demo_corpus.py     # Demo-Korpus für Chunking-Vergleich aufbauen
│   ├── build_embedding_demo_corpus.py    # Demo-Korpus für Embedding-Modell-Vergleich aufbauen
│   ├── evaluate_ragas.py                 # RAGAS-Evaluationsmatrix (Parser × LLM) ausführen
│   ├── evaluate_chunking_ragas.py        # RAGAS-Evaluationsmatrix (Chunking × LLM) ausführen
│   ├── evaluate_embedding_ragas.py       # RAGAS-Evaluationsmatrix (Embedding-Modell × LLM) ausführen
│   ├── evaluate_retrieval_ragas.py       # RAGAS-Evaluationsmatrix (Retrieval-Strategie × LLM) ausführen — kein Demo-Korpus-Build nötig
│   ├── build_best_of_breed_demo.py       # Best-of-Breed-Demo-Collection aufbauen (Unstructured+Semantic+3-large)
│   ├── evaluate_best_of_breed.py         # Best-of-Breed vs. Produktiv-Baseline vergleichen
│   └── evaluate_llm_ragas.py             # RAGAS-Evaluationsmatrix (Antwort-LLM) ausführen — kein Demo-Korpus-Build nötig
├── eval_results/                     # RAGAS-Ergebnisse, CSV + Markdown (git-ignored)
├── main.py                           # Terminal-Chat-Entry-Point
├── app.py                            # Streamlit-Frontend-Entry-Point
└── tests/
```

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` im Projektroot anlegen (siehe `.env.example`):

```bash
cp .env.example .env
# dann OPENAI_API_KEY in .env eintragen
```

Bei Nutzung eines OpenAI-kompatiblen Gateways (z. B. Hochschul-Hub) zusätzlich
`OPENAI_BASE_URL` sowie ggf. provider-präfixierte Modellnamen setzen:

```bash
OPENAI_BASE_URL=https://hub.ki.fh-swf.de/v1
LLM_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Für den LLM-Vergleich im Frontend optional einen Mistral-Key ergänzen
(kostenloser Tier: https://console.mistral.ai/) — ohne Key ist die
Mistral-Option im Frontend deaktiviert:

```bash
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-small-latest
```

## Nutzung

1. Wissensbasis aufbauen (einmalig, bzw. erneut bei geänderten PDFs in `data/raw_pdfs/`):

   ```bash
   python scripts/ingest.py
   ```

2. Chatbot starten — als Terminal-Chat:

   ```bash
   python main.py
   ```

   Befehle im Chat: `/reset` (Verlauf zurücksetzen), `/help`, `/exit`.

   ...oder als Web-Frontend:

   ```bash
   streamlit run app.py
   ```

   Sidebar erlaubt fünf unabhängige Auswahlmöglichkeiten: **Parser** (4),
   **Chunking** (6), **Embedding** (7), **Retrieval-Strategie** (4,
   orthogonal, siehe `docs/RETRIEVAL.md`) und **Antwort-LLM** (3, siehe
   `docs/LLM.md`) — zusammen 2.016 frei kombinierbare Konfigurationen. Ein
   Umschalter wählt zwischen dem vollen ~5.100-Seiten-Produktivkorpus und
   dem ~53-Seiten-Demo-Korpus — Parser/Chunking/Embedding sind in **beiden**
   Modi frei wählbar. Für bereits getestete Kombinationen (siehe
   docs/PARSER.md, docs/CHUNKING.md, docs/EMBEDDING.md,
   docs/BEST_OF_BREED.md) wird die passende Collection direkt
   wiederverwendet; neue, noch nie getestete Kombinationen baut die App beim
   ersten Absenden einer Frage einmalig selbst
   (`ingestion/custom_demo.py::resolve_collection()`) — mit einer live
   berechneten Bauzeit-Schätzung in der Sidebar
   (`estimate_build_seconds()`), bevor der Build startet: auf dem
   Demo-Korpus meist unter einer Minute bis wenige Minuten, auf dem vollen
   Korpus je nach Parser von ~15 Minuten (PyMuPDF4LLM/pdfplumber) bis über
   30 Stunden (Docling — die Sidebar zeigt ab ~30 Min. geschätzter Bauzeit
   eine eskalierte Warnung). Der Parse-Schritt (bei Docling/Unstructured auf
   dem vollen Korpus der dominante Kostenfaktor) wird pro Sitzung
   Parser-weise gecacht — ein Parser-Wechsel wird also nur einmal bezahlt,
   selbst wenn danach mehrere Chunking-/Embedding-Varianten mit demselben
   Parser ausprobiert werden.

## Parser-Vergleich im Frontend

Um die im Frontend wählbaren Parser-Vergleichs-Collections zu befüllen,
zusätzlich die schweren Parser-Abhängigkeiten installieren und den
~50-Seiten-Demo-Korpus einmalig einlesen (siehe auch `docs/PARSER.md`):

```bash
pip install -r requirements-parser-comparison.txt
python scripts/build_parser_demo_corpus.py          # alle 4 Parser (Docling/Unstructured dauern je ca. 10–20 Min.)
python scripts/build_parser_demo_corpus.py pymupdf4llm pdfplumber   # nur die schnellen Parser
```

## Chunking-Vergleich im Frontend

Um die im Frontend wählbaren Chunking-Vergleichs-Collections zu befüllen
(Parser fix: PyMuPDF4LLM, siehe auch `docs/CHUNKING.md`):

```bash
pip install -r requirements-chunking-comparison.txt   # nur für die "semantic"-Strategie nötig
python scripts/build_chunking_demo_corpus.py                          # alle 6 Strategien
python scripts/build_chunking_demo_corpus.py header_recursive_800 semantic   # Auswahl
```

## Embedding-Modell-Vergleich im Frontend

Um die im Frontend wählbaren Embedding-Modell-Vergleichs-Collections zu
befüllen (Parser/Chunking fix: PyMuPDF4LLM/Header+Recursive 800, siehe auch
`docs/EMBEDDING.md`). Die 4 lokalen HuggingFace-Modelle brauchen zusätzlich
`requirements-embedding-comparison.txt` (kein API-Key, aber Download der
Modellgewichte beim ersten Aufruf):

```bash
pip install -r requirements-embedding-comparison.txt   # nur für die 4 HF-Modelle nötig
python scripts/build_embedding_demo_corpus.py                                  # alle 7 Modelle
python scripts/build_embedding_demo_corpus.py text-embedding-3-small multilingual-e5-large   # Auswahl
```

## Retrieval-Vergleich im Frontend

Braucht **keinen eigenen Demo-Korpus** — Retrieval-Strategien wirken nur zur
Query-Zeit auf der bereits vorhandenen Chunking-Demo-Collection (siehe auch
`docs/RETRIEVAL.md`). Für "Rerank"/"Hybrid" zusätzlich installieren:

```bash
pip install -r requirements-retrieval-comparison.txt
```

## Quantitative Evaluation (RAGAS)

Bewertet alle 4 Parser × 2 LLMs (8 Kombinationen), alle 6 Chunking-Strategien
× 2 LLMs (12 Kombinationen), alle 7 Embedding-Modelle × 2 LLMs (14
Kombinationen) sowie alle 4 Retrieval-Strategien × 2 LLMs (8 Kombinationen)
auf demselben Demo-Korpus mit einem festen Fragenset und 5 RAGAS-Metriken
(Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
FactualCorrectness). Details und Ergebnisse:
[`docs/EVALUATION.md`](docs/EVALUATION.md) (Parser),
[`docs/CHUNKING.md`](docs/CHUNKING.md) (Chunking),
[`docs/EMBEDDING.md`](docs/EMBEDDING.md) (Embedding-Modell) und
[`docs/RETRIEVAL.md`](docs/RETRIEVAL.md) (Retrieval-Strategie).

```bash
pip install -r requirements-eval.txt
python scripts/build_parser_demo_corpus.py    # falls noch nicht geschehen
python scripts/evaluate_ragas.py

# Chunking-Vergleich (Parser fix: PyMuPDF4LLM):
python scripts/build_chunking_demo_corpus.py    # falls noch nicht geschehen
python scripts/evaluate_chunking_ragas.py

# Embedding-Modell-Vergleich (Parser/Chunking fix: PyMuPDF4LLM/Header+Recursive 800):
python scripts/build_embedding_demo_corpus.py    # falls noch nicht geschehen
python scripts/evaluate_embedding_ragas.py

# Retrieval-Vergleich (Parser/Chunking/Embedding fix, kein Demo-Korpus-Build nötig):
pip install -r requirements-retrieval-comparison.txt
python scripts/evaluate_retrieval_ragas.py
```

## Enthaltene Handbücher

Standardmäßig sind vier Abaqus-2017-Handbücher in `data/raw_pdfs/` hinterlegt
(~5.100 Seiten, 15.408 Chunks):

- `Abaqus2017_GETTINGSTARTED.pdf` — Einstieg/Bedienung von Abaqus/CAE
- `Abaqus2017_ANALYSIS.pdf` — Analyse-Verfahren
- `Abaqus2017_THEORY.pdf` — mathematisch-theoretischer Hintergrund
- `Abaqus2017_KEYWORDS.pdf` — Keyword-Referenz (Input-Deck-Syntax)

Weitere PDFs können einfach in denselben Ordner gelegt und per
`python scripts/ingest.py` neu eingelesen werden — dabei wird die
bestehende Collection vollständig neu aufgebaut.

## Tests

```bash
python -m pytest tests/ -v
```

Die Tests laufen offline (deterministische Fake-Embeddings, kein OpenAI-Call).
