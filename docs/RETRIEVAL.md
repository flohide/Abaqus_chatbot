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
  `_ragas_compat`-Stub, siehe `RAGAS.md`).
- **Reciprocal Rank Fusion (RRF):** `hybrid` kombiniert BM25- und
  Dense-Ranking über `Score(d) = Σ 1/(rrf_k + rank)` (`rrf_k=60`, Standard
  aus der RRF-Literatur) statt eine LangChain-`EnsembleRetriever`-Abhängigkeit
  einzuführen — eigenständige, leicht testbare Implementierung.

## Methodik

Kein Demo-Korpus-Build nötig (siehe oben) — direkt RAGAS-Evaluation
(identisches 12-Fragen-Set, fester `gpt-4o-mini`-Richter, siehe
[`RAGAS.md`](RAGAS.md)):

```bash
pip install -r requirements-eval.txt
python scripts/build_chunking_demo_corpus.py header_recursive_800   # falls noch nicht geschehen
python scripts/evaluate_retrieval_ragas.py
```

Ergebnisse landen in `eval_results/retrieval_ragas_{raw,summary}_<timestamp>.{csv,md}`
(gitignored). Interaktiv vergleichbar über [`docs/dashboard.html`](dashboard.html)
(Tab "Retrieval").

## Ergebnisse

Vollständiger Lauf: 4 Retrieval-Strategien × 12 Fragen = 48 Instanzen × 5
Metriken = 240 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/retrieval_ragas_raw_20260914_044238.csv`](../eval_results/retrieval_ragas_raw_20260914_044238.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_retrieval_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Retrieval-Strategie | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|
| 1 | Rerank | 0.938 | 0.880 | 0.840 | **1.000** | 0.686 | **0.869** |
| 2 | Similarity | 0.899 | 0.876 | 0.864 | **1.000** | 0.619 | **0.852** |
| 3 | Hybrid | 0.781 | 0.809 | 0.855 | **1.000** | 0.704 | **0.830** |
| 4 | MMR (Produktiv) | 0.856 | 0.735 | **0.901** | 0.917 | 0.596 | **0.801** |

