# Best-of-Breed: Kombination der vier Einzelsieger

Abschließendes Experiment zu den vier unabhängigen RAGAS-Vergleichen
([`PARSER.md`](PARSER.md), [`CHUNKING.md`](CHUNKING.md),
[`EMBEDDING.md`](EMBEDDING.md), [`RETRIEVAL.md`](RETRIEVAL.md)): Jeder
Vergleich hat isoliert genau eine Dimension variiert, die anderen drei
blieben auf dem Produktiv-Default fixiert — kombiniert getestet wurden die
vier Einzelsieger bisher nicht. Dieser Vergleich testet, ob das Kombinieren
der Einzelsieger tatsächlich die beste Gesamt-Pipeline ergibt, oder ob sich
Wechselwirkungen zwischen den Dimensionen gegenseitig neutralisieren.

## Die zwei Pipelines

| Dimension | Produktiv-Baseline | Best-of-Breed | Einzelsieger laut |
|---|---|---|---|
| Parser | PyMuPDF4LLM | **Unstructured** (hi_res) | `EVALUATION.md`, Rang 1 (⌀ 0.785) |
| Chunking | Header+Recursive 800 | **Semantic** | `CHUNKING.md`, Rang 1 (⌀ 0.821 mit Produktiv-Embedding) |
| Embedding | text-embedding-3-small | **text-embedding-3-large** | `EMBEDDING.md`, Rang 1 (⌀ 0.820) |
| Retrieval | MMR | **Rerank** (Cross-Encoder) | `RETRIEVAL.md`, Rang 1 (⌀ 0.863, perfekte ContextRecall) |

Beide Pipelines laufen im selben RAGAS-Lauf mit demselben Richter
(fester OpenAI-Judge, siehe `eval/metrics.py`) und demselben 12-Fragen-Set,
damit der Vergleich sauber ist — die Baseline wird dafür frisch neu
generiert statt alte Zahlen aus `CHUNKING.md`/`RETRIEVAL.md`
wiederzuverwenden.

**Wichtiger Vorbehalt:** "Beste Einzeloption" bedeutet hier den höchsten
RAGAS-Gesamtdurchschnitt, nicht zwangsläufig die pragmatischste
Produktiv-Wahl — Unstructured ist ~70× langsamer als PyMuPDF4LLM und deutlich
größer in der Installation (siehe `PARSER.md`), was für den vollen
~5.100-Seiten-Produktivkorpus relevant wäre, für dieses Experiment aber
bewusst ignoriert wird.

## Methodik

```bash
python scripts/build_best_of_breed_demo.py   # baut die Best-of-Breed-Collection (dauert mehrere Minuten wegen Unstructured hi_res)
pip install -r requirements-eval.txt
python scripts/evaluate_best_of_breed.py
```

Ergebnisse landen in `eval_results/best_of_breed_{raw,summary}_<timestamp>.{csv,md}`
(gitignored).

## Ergebnisse

Vollständiger Lauf: 2 Pipelines × 3 LLMs (OpenAI, Mistral, Qwen — siehe
[`LLM.md`](LLM.md)) = 6 Kombinationen × 12 Fragen = 72 Instanzen × 5
Metriken = 360 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Rohdaten:
[`eval_results/best_of_breed_raw_20260813_063342.csv`](../eval_results/best_of_breed_raw_20260813_063342.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_best_of_breed.py`).
(Ein früherer Lauf mit nur 2 LLMs — bevor Qwen als dritte, kostenlose
Option ergänzt wurde — hatte 1 vereinzelten Judge-Timeout; dieser Lauf ist
vollständig fehlerfrei.)

### Gesamtvergleich (Mittelwert über alle 5 Metriken)

| Rang | Pipeline | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|
| 1 | **Best-of-Breed** | **Qwen (kostenlos)** | 0.866 | 0.851 | 0.912 | **1.000** | 0.665 | **0.859** |
| 2 | **Best-of-Breed** | OpenAI | 0.833 | 0.878 | 0.912 | **1.000** | 0.643 | **0.853** |
| 3 | **Best-of-Breed** | Mistral | 0.881 | 0.812 | 0.912 | **1.000** | 0.542 | **0.829** |
| 4 | Baseline (Produktiv) | Qwen | 0.826 | 0.698 | 0.822 | 0.792 | 0.641 | 0.756 |
| 5 | Baseline (Produktiv) | OpenAI | 0.764 | 0.813 | 0.822 | 0.792 | 0.571 | 0.752 |
| 6 | Baseline (Produktiv) | Mistral | 0.733 | 0.696 | 0.822 | 0.792 | 0.590 | 0.727 |

**Best-of-Breed gewinnt bei jeder einzelnen der 5 Metriken, für alle drei
LLMs, ohne Ausnahme.** Gesamtdurchschnitt über alle drei LLMs:
Best-of-Breed **0.847** vs. Baseline **0.745** — eine Verbesserung um
**+0.102 (+13,7 %)**, praktisch identisch zum ursprünglichen 2-LLM-Befund
(+14,4 %).

**Wichtiger Gegencheck — kein neuer Gesamtsieger:** Best-of-Breed + Qwen
(0.859) ist zwar die beste Kombination *dieser* Tabelle, bleibt aber knapp
**hinter** der besten Einzelkombination des gesamten Projekts:
**Retrieval: Rerank + OpenAI aus `RETRIEVAL.md` (⌀ 0.863)** — dort wurde nur
die Retrieval-Strategie auf der sonst unveränderten Produktiv-Pipeline
getauscht, ohne Parser/Chunking/Embedding zu ändern. Das Kombinieren *aller
vier* Einzelsieger schlägt zwar die Baseline klar, schlägt aber **nicht**
die einfachere "nur Rerank hinzufügen"-Variante — die Hypothese "je mehr
Einzelsieger kombiniert, desto besser" bestätigt sich hier also nicht
uneingeschränkt (siehe unten).

### Einordnung

- **Stacking aller vier Einzelsieger übertrifft nicht die einfachere
  "nur Rerank"-Verbesserung** — ein ernüchternder, aber wichtiger Befund:
  Auf vergleichbarer 2-LLM-Basis (OpenAI/Mistral) liegt Best-of-Breed bei ⌀
  0.841, reines Rerank-auf-Produktiv-Pipeline bei ⌀ 0.850 (aus
  `RETRIEVAL.md`). Ein plausibler Grund: Unstructured (hi_res) erkannte auf
  dem Demo-Korpus nur 40 von 53 Seiten als nicht-leer (siehe Build-Log) —
  der Parser-Wechsel verliert also Korpusabdeckung, was die Gewinne aus
  besserem Chunking/Embedding/Retrieval teilweise wieder auffrisst. Mehr
  Einzelverbesserungen zu stapeln ist hier kein Selbstläufer.
- **Qwen bestätigt seinen Vorsprung aus `LLM.md` auch auf der
  Best-of-Breed-Pipeline** — dort gewann es bereits das reine
  LLM-Ranking (⌀ 0.756 vs. 0.748 OpenAI), hier gewinnt es zusätzlich mit
  der besten AnswerRelevancy aller drei LLMs auf dieser Pipeline (0.851 —
  auf der Baseline war AnswerRelevancy noch die schwächste Metrik von
  Qwen). Ein besserer Kontext scheint Qwens tendenziell abschweifende
  Antworten (siehe `LLM.md`) spürbar zu fokussieren.
- **Best-of-Breed schlägt trotzdem klar die Baseline:** ContextPrecision
  steigt auf 0.912 — höher als der beste Einzelwert, den
  `text-embedding-3-large` (0.871) oder Rerank allein auf der
  Produktiv-Pipeline (0.826) jeweils erreicht hatten (nur
  `multilingual-e5-large` allein lag mit 0.958 noch höher). Die
  Kombination hilft also gegenüber der Baseline messbar — nur eben nicht
  mehr als die einfachste Einzelverbesserung (Rerank) allein.
- **ContextRecall ist wie beim reinen Retrieval-Vergleich perfekt (1.000)**
  für alle drei LLMs — der Rerank-Effekt trägt unverändert durch,
  unabhängig von Parser/Chunking/Embedding/LLM-Wechsel.
- **Mistral zeigt eine Anomalie bei FactualCorrectness:** Als einziges LLM
  sinkt dieser Wert leicht von Baseline zu Best-of-Breed (0.590 → 0.542),
  während alle anderen 4 Metriken sich für Mistral deutlich verbessern
  (Faithfulness 0.733 → 0.881, AnswerRelevancy 0.696 → 0.812). Trotzdem
  bleibt Mistrals Gesamtdurchschnitt klar besser (0.727 → 0.829).
- **Praktische Empfehlung, angesichts des Gegenchecks oben:** Die volle
  Vier-Dimensionen-Kombination lohnt sich hier nicht gegenüber der
  einfacheren Alternative — Unstructured (hi_res, ~70× langsamer als
  PyMuPDF4LLM, siehe `PARSER.md`), Semantic Chunking (Embedding-Call pro
  Satz bei der Ingestion, siehe `CHUNKING.md`) und `text-embedding-3-large`
  (teurer pro Token) kosten spürbar mehr, ohne auf dieser Stichprobe einen
  Vorteil gegenüber "nur Rerank" zu bringen. Die durch diese Zahlen
  empirisch am besten begründbare Kombination ist stattdessen die
  **schlankere**: Produktiv-Parser/-Chunking/-Embedding beibehalten, nur
  **Rerank** ergänzen (⌀ 0.850, `RETRIEVAL.md`) und als LLM **Qwen**
  verwenden (kostenlos, kein separater Key, siehe `LLM.md`) — beide
  Verbesserungen sind günstiger als der volle Best-of-Breed-Umbau und
  liegen in derselben Qualitätsliga.

**Wichtige Einschränkung:** n=12 Fragen auf einem reduzierten Demo-Korpus
(Unstructured erkannte nur 40 von 53 Seiten als nicht-leer, siehe
Build-Log) ist eine kleine Stichprobe für 72 LLM-bewertete Instanzen —
geeignet, um die klare Gesamttendenz sichtbar zu machen, aber eine einzelne
Zahl (+13,7 %) sollte nicht als präzise, reproduzierbare Effektgröße auf dem
vollen Produktivkorpus gelesen werden.
