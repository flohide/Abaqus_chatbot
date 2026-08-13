# Retrieval-Strategie-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md),
[`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md) und
[`EMBEDDING.md`](EMBEDDING.md): Vier Retrieval-Strategien wurden auf
derselben Wissensbasis quantitativ mit RAGAS verglichen, um die Wahl von MMR
(Max Marginal Relevance) als Produktiv-Retriever empirisch zu begründen —
und um den in `DOKUMENTATION.md` §13 offen notierten Punkt ("Kein
Re-Ranking / keine Contextual Compression") zu schließen.

## Kandidaten

| Strategie | Ansatz |
|---|---|
| `mmr` | **Produktiv-Baseline** (`rag/retriever.py`) — MMR über Chroma, `k=5`, `fetch_k=20` |
| `similarity` | naive Baseline ohne MMR-Diversität — reine Top-`k`-Ähnlichkeitssuche |
| `rerank` | MMR holt `fetch_k=20` Kandidaten, ein Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, multilingual) sortiert sie nach tatsächlicher Query-Relevanz neu, Top-`k=5` |
| `hybrid` | BM25 (Keyword-Suche, `rank_bm25`) + Dense-Similarity-Suche, je `fetch_k=20` Kandidaten, fusioniert per Reciprocal Rank Fusion zu Top-`k=5` |

**Vergleichspaare:**
- `mmr` vs. `similarity` → Wert der MMR-Diversität selbst.
- `mmr`/`similarity` vs. `rerank` → Wert eines nachgeschalteten
  Relevanz-Rerankings.
- `mmr` vs. `hybrid` → Wert von Keyword-Signal zusätzlich zu Embeddings —
  bei Abaqus mit exakten Keyword-Referenzen (`*STEP`,
  `*COUPLED TEMPERATURE-DISPLACEMENT`) potenziell relevant, da Embeddings
  exakte Syntax-Treffer manchmal schlechter abbilden als Volltextsuche.

## Architektur: keine eigene Ingestion nötig

Anders als Parser-, Chunking- und Embedding-Vergleich ist Retrieval eine
reine **Query-Zeit**-Entscheidung, keine Ingestion-Entscheidung — es gibt
keinen eigenen Demo-Korpus-Build-Schritt. Alle vier Strategien laufen auf
derselben, bereits vorhandenen Produktiv-äquivalenten Collection
(`chunking_demo_header_recursive_800`: PyMuPDF4LLM + Header+Recursive 800 +
`text-embedding-3-small`, siehe `CHUNKING.md`).

`rag/retrieval_backends.py::RETRIEVAL_BACKENDS` ist eine Registry von
Factories `(vectorstore) -> Callable[[str], list[Document]]`, analog zu
`CHUNKING_BACKENDS`/`EMBEDDING_BACKENDS`. `rag/chain.py::build_rag_chain()`
bekam dafür einen `retrieval_strategy`-Parameter (Default `"mmr"` =
unverändertes Produktiv-Verhalten).

Zwei Implementierungsdetails:
- **BM25 ohne `langchain_community`:** `_BM25Index` kapselt `rank_bm25`
  direkt (eigene Tokenisierung), statt über
  `langchain_community.retrievers.BM25Retriever` zu gehen — das Paket ist
  laut Upstream im Sunsetting-Modus (gleiches Argument wie beim
  `_ragas_compat`-Stub, siehe `EVALUATION.md`).
- **Reciprocal Rank Fusion (RRF):** `hybrid` kombiniert BM25- und
  Dense-Ranking über `Score(d) = Σ 1/(rrf_k + rank)` (`rrf_k=60`, Standard
  aus der RRF-Literatur) statt eine LangChain-`EnsembleRetriever`-Abhängigkeit
  einzuführen — eigenständige, leicht testbare Implementierung.

## Methodik

Kein Demo-Korpus-Build nötig (siehe oben) — direkt RAGAS-Evaluation
(identisches 12-Fragen-Set, fester OpenAI-Richter, siehe
[`EVALUATION.md`](EVALUATION.md)):

```bash
pip install -r requirements-retrieval-comparison.txt   # nur für "rerank"/"hybrid" nötig
pip install -r requirements-eval.txt
python scripts/build_chunking_demo_corpus.py header_recursive_800   # falls noch nicht geschehen
python scripts/evaluate_retrieval_ragas.py
```

Ergebnisse landen in `eval_results/retrieval_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored). Live vergleichbar auch im Streamlit-Frontend (`app.py`): alle
vier Strategien stehen in der Sidebar unter "Retrieval-Strategie" zur
Auswahl — orthogonal zur Wissensbasis, wirkt also auf jede beliebige
gewählte Collection.

## Ergebnisse

Vollständiger Lauf: 4 Retrieval-Strategien × 2 LLMs = 8 Kombinationen × 12
Fragen = 96 Instanzen × 5 Metriken = 480 Einzel-Scores, **0 Fehler, 0
fehlende Werte**. Laufzeit: ~18–27 s pro Kombination für die
Antwortgenerierung (`rerank` beim ersten Aufruf zusätzlich ein einmaliger
Cross-Encoder-Download), ~4,7 Min. für die gesamte Metrik-Berechnung
(Concurrency=4). Rohdaten:
[`eval_results/retrieval_ragas_raw_20260812_200254.csv`](../eval_results/retrieval_ragas_raw_20260812_200254.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_retrieval_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Retrieval-Strategie | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|
| 1 | Rerank | OpenAI | 0.898 | 0.876 | 0.826 | **1.000** | 0.713 | **0.863** |
| 2 | Rerank | Mistral | 0.854 | 0.794 | 0.826 | **1.000** | 0.704 | **0.836** |
| 3 | Similarity | OpenAI | 0.940 | 0.877 | 0.825 | 0.875 | 0.642 | **0.832** |
| 4 | Hybrid | OpenAI | 0.931 | 0.796 | 0.809 | 0.875 | 0.644 | **0.811** |
| 5 | Similarity | Mistral | 0.844 | 0.695 | 0.825 | 0.875 | 0.664 | **0.781** |
| 6 | MMR (Produktiv) | OpenAI | 0.750 | 0.813 | 0.822 | 0.792 | 0.578 | **0.751** |
| 7 | Hybrid | Mistral | 0.717 | 0.673 | 0.809 | 0.875 | 0.613 | **0.737** |
| 8 | MMR (Produktiv) | Mistral | 0.796 | 0.655 | 0.822 | 0.792 | 0.482 | **0.709** |

### Retrieval-Qualität pro Strategie (LLM-unabhängig)

`context_precision`/`context_recall` hängen ausschließlich vom Retrieval ab,
nicht vom antwortenden LLM — fallen wie bei den anderen Vergleichen zwischen
OpenAI- und Mistral-Läufen **identisch** aus:

| Retrieval-Strategie | ContextPrecision | ContextRecall |
|---|---|---|
| Rerank | **0.826** | **1.000** |
| Similarity | 0.825 | 0.875 |
| MMR (Produktiv) | 0.822 | 0.792 |
| Hybrid | 0.809 | 0.875 |

### Einordnung

- **Rerank gewinnt klar und ohne Trade-off:** beste ContextPrecision *und*
  perfekte ContextRecall (1.000 — alle für die Referenzantworten nötigen
  Chunks wurden in jeder der 12 Fragen gefunden) gleichzeitig. Der
  Cross-Encoder bewertet die 20 MMR-Kandidaten nach tatsächlicher
  Query-Relevanz neu, statt sich (wie MMR) zusätzlich um Diversität zu
  kümmern — auf diesem Demo-Korpus zahlt sich das eindeutig aus. Schließt
  den in `DOKUMENTATION.md` §13 offen notierten Punkt.
- **Überraschung: `similarity` (keine MMR-Diversität) schlägt `mmr`
  (Produktiv) bei beiden Retrieval-Metriken** (Precision 0.825 vs. 0.822,
  Recall 0.875 vs. 0.792) — das isolierte Vergleichspaar zeigt, dass MMRs
  Diversitäts-Optimierung auf diesem kompakten 53-Seiten-Korpus tendenziell
  eher schadet als hilft: Sie tauscht gelegentlich einen direkt relevanten
  Chunk gegen einen "diverseren", aber weniger relevanten. **MMR hat die
  schlechteste ContextRecall aller vier Strategien** (0.792) — ausgerechnet
  die Produktiv-Baseline. Bei einem großen, thematisch redundanteren
  Produktivkorpus (~5.100 Seiten) könnte sich das Bild umkehren, da dort das
  Risiko vieler nahezu identischer Chunks aus derselben Seite/demselben
  Abschnitt deutlich höher ist.
- **Hybrid (BM25 + Dense) hilft hier nicht:** trotz des Arguments für exakte
  Keyword-Treffer (`*STEP` etc.) liegt Hybrid bei ContextPrecision am
  niedrigsten aller vier Strategien (0.809) — die getesteten 12 Fragen
  enthalten wenige exakte Keyword-Lookups, das BM25-Signal fügt in der RRF-
  Fusion hier eher Rauschen als Präzision hinzu. Ein Vergleichspaar, dessen
  erwarteter Vorteil sich in dieser Stichprobe nicht bestätigt — ein
  Fragenset mit mehr Keyword-Referenz-Fragen (z. B. `*STEP`-Parameter,
  Keyword-Syntax) wäre ein faireres Testfeld für Hybrid-Retrieval.
- **OpenAI übertrifft Mistral bei allen 4 Retrieval-Strategien im
  Gesamtmittel** — durchgängiges Muster über alle vier Vergleichsdimensionen
  hinweg (Parser, Chunking, Embedding, Retrieval).
- **FactualCorrectness bleibt über alle Kombinationen hinweg die niedrigste
  Metrik** (0.48–0.71) — typisch für dieses RAGAS-Maß; hier allerdings mit
  klarem Zusammenhang zur Retrieval-Strategie: Rerank hat mit 0.70–0.71
  auch bei FactualCorrectness die höchsten Werte aller vier Strategien —
  bessere Kontexte führen offenbar auch zu faktisch korrekteren Antworten.

**Wichtige Einschränkung:** Wie bei den anderen Vergleichen ist n=12 Fragen
auf einem 53-Seiten-Demo-Korpus eine kleine Stichprobe für 96
LLM-bewertete Instanzen — geeignet, um klare Tendenzen sichtbar zu machen
(den Rerank-Vorsprung, die MMR-Recall-Schwäche), aber nicht, um knappe
Unterschiede (z. B. Rang 3 vs. 4) als statistisch gesichert zu
interpretieren. Insbesondere die perfekte ContextRecall von Rerank sollte
nicht als Garantie für größere Korpora gelesen werden — bei mehr
Kandidaten und mehr potenziell relevanten Chunks pro Frage wird perfekte
Recall schwerer zu erreichen.
