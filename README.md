# Abaqus Handbuch-Chatbot (RAG)

Ein Retrieval-Augmented-Generation-System, das Fragen zur Bedienung der
FEA-Software Abaqus auf Basis der offiziellen PDF-Handbücher beantwortet —
mit Seiten-genauem Source-Tracking und Terminal-Chat.

**Interaktives Vergleichs-Dashboard** (alle 6 RAGAS-Studien, Parser ×
Chunking × Embedding × Retrieval × LLM × Best-of-Breed, mit Diagrammen und
Rohdaten-Tabellen): [`docs/dashboard.html`](docs/dashboard.html) — lokal
herunterladen und im Browser öffnen, keine Installation nötig.

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

Antwort-LLM-Vergleich (OpenAI-mini vs. Mistral (`mistral-small-latest`) vs. GPT-4o, über denselben
Hub wie OpenAI) inkl. RAGAS-Evaluation: [`docs/LLM.md`](docs/LLM.md)

Quantitative RAGAS-Evaluation (4 Parser, Antwort-LLM fest `gpt-4o-mini`,
5 Metriken, 240 Scores): [`docs/PARSER.md`](docs/PARSER.md) (Framework/Metriken/
Code-Einsatz: [`docs/RAGAS.md`](docs/RAGAS.md))

## Architekturentscheidungen

| Bereich | Wahl | Begründung |
|---|---|---|
| Vektor-DB | [Chroma](https://www.trychroma.com/) (lokal, persistent) | Keine Server-Infrastruktur nötig, gute LangChain-Integration, Metadaten-Filterung möglich. |
| Memory | Einfache Nachrichtenliste in der Chat-Session | Ausreichend für ein Single-User-Tool, kein Session-Store nötig. |
| Frontend | [Streamlit](https://streamlit.io/) (`st.chat_message`/`st.chat_input`) | Klassische Chat-Optik mit wenig Code, bietet bewusst nur die Produktiv-Pipeline an (kein Pipeline-Auswahlfeld, siehe "Nutzung"). |
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
│   │   └── custom_demo.py            # Parser×Chunking×Embedding-Kombination auflösen/on-demand bauen (u.a. von app.py::PIPELINES genutzt)
│   ├── rag/                          # Retrieval + LCEL-Kette
│   │   ├── vectorstore.py
│   │   ├── retriever.py              # Produktiv-Retriever (MMR)
│   │   ├── retrieval_backends.py     # 4 austauschbare Retrieval-Strategien (siehe docs/RETRIEVAL.md)
│   │   ├── prompts.py
│   │   ├── chain.py
│   │   └── llms.py                   # LLM-Provider-Auswahl (OpenAI/Mistral)
│   ├── eval/                         # RAGAS-Evaluation (siehe docs/RAGAS.md, docs/CHUNKING.md)
│   │   ├── _ragas_compat.py          # Kompatibilitäts-Stub für ragas + LangChain >=1.0
│   │   ├── dataset.py                # 12 Eval-Fragen + Referenzantworten
│   │   ├── metrics.py                # 5 RAGAS-Metriken, Richter je Studie: gpt-4o-mini oder Claude Sonnet 5
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

Für Mistral als Antwort-LLM in den Vergleichsstudien (nicht für die
Chat-App selbst nötig, siehe `docs/LLM.md`, `docs/BEST_OF_BREED.md`)
optional einen Mistral-Key ergänzen (kostenloser Tier:
https://console.mistral.ai/):

```bash
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-small-latest
```

> **Für Gutachter/innen:** Die fertig gebaute Vectorstore-Collection
> (`vectorstore/`) ist **nicht** Teil dieses Repos — ihr HNSW-Index allein
> ist ~295 MB groß und würde GitHubs 100-MB-Datei-Limit sprengen. Die
> Quell-PDFs (`data/raw_pdfs/`, zusammen ~52 MB) sind dagegen enthalten.
> Einmalig `python scripts/ingest.py` laufen lassen (siehe "Nutzung" unten,
> ~13 Min.), dann ist die Anwendung (Terminal-Chat und Web-Frontend)
> nutzbar.

## Nutzung

1. Wissensbasis aufbauen — **einmalig nötig** (die Produktiv-Collection ist
   nicht Teil des Repos, siehe oben), danach nur erneut bei geänderten
   PDFs in `data/raw_pdfs/`:

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

   Das Frontend bietet bewusst **nur die Produktiv-Pipeline** an
   (PyMuPDF4LLM + Header+Recursive 800 + `text-embedding-3-small` + MMR +
   `gpt-4o-mini`, voller ~5.100-Seiten-Korpus) — kein Pipeline-Auswahlfeld
   mehr. Die früher zusätzlich wählbare **Best-of-Breed**-Pipeline (siehe
   `docs/BEST_OF_BREED.md`) wurde aus `app.py::PIPELINES` entfernt: Sie
   bräuchte `vectorstore_full_custom/`, das `scripts/ingest.py` nicht mit
   aufbaut und für das es kein schnelles Setup-Skript gibt — ohne
   vorherigen manuellen Build hätte die Auswahl im Frontend einen
   automatischen, aber **ca. 5-stündigen** Unstructured-`hi_res`-Build
   ausgelöst (siehe `docs/PARSER.md`). Der freie Wechsel zwischen den drei
   Antwort-LLMs (OpenAI-mini, GPT-4o, Mistral (`mistral-small-latest`)) findet ausschließlich in der
   RAGAS-Studie statt, siehe `docs/LLM.md`. Die einzelnen
   Demo-Korpus-Vergleichsstudien (Parser/Chunking/Embedding/Retrieval, siehe
   unten) sind über die App ebenfalls **nicht** live auswählbar — sie laufen
   ausschließlich über die jeweiligen `scripts/evaluate_*_ragas.py`-Skripte
   bzw. das `docs/dashboard.html`.

## Parser-Vergleich (Demo-Korpus für die RAGAS-Studie)

Um die Parser-Vergleichs-Collections für `scripts/evaluate_ragas.py` zu
befüllen, zusätzlich die schweren Parser-Abhängigkeiten installieren und den
~50-Seiten-Demo-Korpus einmalig einlesen (siehe auch `docs/PARSER.md`):

```bash
pip install -r requirements-parser-comparison.txt
python scripts/build_parser_demo_corpus.py          # alle 4 Parser (Docling/Unstructured dauern je ca. 10–20 Min.)
python scripts/build_parser_demo_corpus.py pymupdf4llm pdfplumber   # nur die schnellen Parser
```

## Chunking-Vergleich (Demo-Korpus für die RAGAS-Studie)

Um die Chunking-Vergleichs-Collections für `scripts/evaluate_chunking_ragas.py`
zu befüllen (Parser fix: PyMuPDF4LLM, siehe auch `docs/CHUNKING.md`):

```bash
pip install -r requirements-chunking-comparison.txt   # nur für die "semantic"-Strategie nötig
python scripts/build_chunking_demo_corpus.py                          # alle 6 Strategien
python scripts/build_chunking_demo_corpus.py header_recursive_800 semantic   # Auswahl
```

## Embedding-Modell-Vergleich (Demo-Korpus für die RAGAS-Studie)

Um die Embedding-Modell-Vergleichs-Collections für
`scripts/evaluate_embedding_ragas.py` zu befüllen (Parser/Chunking fix:
PyMuPDF4LLM/Header+Recursive 800, siehe auch `docs/EMBEDDING.md`). Die 4
lokalen HuggingFace-Modelle brauchen zusätzlich
`requirements-embedding-comparison.txt` (kein API-Key, aber Download der
Modellgewichte beim ersten Aufruf):

```bash
pip install -r requirements-embedding-comparison.txt   # nur für die 4 HF-Modelle nötig
python scripts/build_embedding_demo_corpus.py                                  # alle 7 Modelle
python scripts/build_embedding_demo_corpus.py text-embedding-3-small multilingual-e5-large   # Auswahl
```

## Retrieval-Vergleich (für die RAGAS-Studie)

Braucht **keinen eigenen Demo-Korpus** — Retrieval-Strategien wirken nur zur
Query-Zeit auf der bereits vorhandenen Chunking-Demo-Collection (siehe auch
`docs/RETRIEVAL.md`). "Rerank"/"Hybrid" brauchen `rank_bm25`/
`sentence-transformers`, die bereits Teil von `requirements.txt` sind.

## Quantitative Evaluation (RAGAS)

Bewertet alle 4 Parser, alle 6 Chunking-Strategien, alle 7 Embedding-Modelle
sowie alle 4 Retrieval-Strategien (Antwort-LLM jeweils fest auf
`gpt-4o-mini`, der eigentliche LLM-Vergleich läuft separat, siehe unten)
auf demselben Demo-Korpus mit einem festen Fragenset und 5 RAGAS-Metriken
(Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
FactualCorrectness). Details und Ergebnisse:
[`docs/PARSER.md`](docs/PARSER.md) (Parser, siehe auch [`docs/RAGAS.md`](docs/RAGAS.md)),
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
