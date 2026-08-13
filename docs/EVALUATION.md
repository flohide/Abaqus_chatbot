# RAGAS-Evaluation: Parser × LLM

Quantitative Auswertung der Antwortqualität über die volle Vergleichsmatrix
aus vier PDF-Parsern (siehe [`PARSER.md`](PARSER.md)) und zwei LLMs (siehe
[`DOKUMENTATION.md`, Abschnitt 12.1](DOKUMENTATION.md#121-llm-vergleich-openai-vs-mistral))
mit dem [RAGAS](https://docs.ragas.io/)-Framework.

## Methodik

### Korpus & Fragenset

Alle 8 Kombinationen werden auf demselben ~53-Seiten-Demo-Korpus
(`ingestion/parser_demo.py`, siehe `PARSER.md`) mit demselben, festen
12-Fragen-Set (`eval/dataset.py`) evaluiert — nur so sind die Ergebnisse
zwischen den Varianten direkt vergleichbar. Die Fragen decken bewusst
unterschiedliche Inhaltstypen ab (prozedurale Anleitungen, Tabellen-Lookup,
Theorie/Konzepte, spezifische Keywords, TOC-Navigation) und wurden anhand
des tatsächlichen Handbuchtexts formuliert; die Referenzantworten sind
manuell verfasst (kein automatisch generiertes Ground Truth).

### Metriken

Alle fünf Metriken werden aus der [RAGAS-Metrik-Sammlung](https://docs.ragas.io/en/latest/concepts/metrics/)
(`ragas.metrics.collections`) bezogen:

| Metrik | Braucht Referenz? | Was wird gemessen? |
|---|---|---|
| **Faithfulness** | Nein | Ist die Antwort durch den abgerufenen Kontext gedeckt (keine Halluzination)? |
| **AnswerRelevancy** | Nein | Beantwortet die Antwort tatsächlich die gestellte Frage? |
| **ContextPrecisionWithReference** | Ja | Sind die abgerufenen Chunks relevant für die (referenz-)korrekte Antwort? |
| **ContextRecall** | Ja | Deckt der abgerufene Kontext ab, was für die Referenzantwort nötig wäre? |
| **FactualCorrectness** | Ja | Wie viele Fakten der Antwort stimmen mit der Referenz überein (F1)? |

Alle Werte liegen im Bereich [0, 1], höher ist besser.

### Fester Richter (Judge-LLM)

Für **alle** 8 Kombinationen wird derselbe Richter verwendet: `gpt-4o-mini`
über das Hochschul-Gateway (`eval/metrics.py`). Würde man stattdessen je
nach getesteter Variante auch als Richter dasselbe Modell einsetzen (also
Mistral-Antworten von Mistral bewerten lassen), entstünde ein
**Self-Preference-Bias** — LLMs bewerten ihre eigenen Formulierungen
tendenziell günstiger. Ein fixer, unabhängiger Richter macht die 8 Scores
untereinander fair vergleichbar.

### Technischer Hinweis: ragas 0.4.3 + LangChain ≥1.0

`ragas` importiert beim Laden unbedingt `langchain_community.chat_models
.vertexai` (`ChatVertexAI`) — dieses Submodul wurde im Zuge des
"Sunsetting" von `langchain-community` entfernt (vgl. bereits das
`langchain.chains`-Problem in `DOKUMENTATION.md`, Abschnitt 3). Ein
minimaler Kompatibilitäts-Stub (`eval/_ragas_compat.py`, wird vor jedem
`ragas`-Import geladen) behebt das, ohne Vertex AI tatsächlich zu benötigen
oder zu nutzen.

### Ausführung

```bash
pip install -r requirements-eval.txt
python scripts/build_parser_demo_corpus.py   # falls noch nicht geschehen
python scripts/evaluate_ragas.py
```

Optional eingeschränkt:

```bash
python scripts/evaluate_ragas.py --parsers pymupdf4llm docling --llms openai
```

Ergebnisse landen als Roh-CSV (eine Zeile pro Frage×Parser×LLM) und
aggregierte Zusammenfassung (CSV + Markdown) in `eval_results/`
(git-ignored).

## Ergebnisse

Vollständiger Lauf: 8 Kombinationen × 12 Fragen = 96 Instanzen × 5 Metriken
= 480 Einzel-Scores, **0 Fehler, 0 fehlende Werte**. Laufzeit: ~23–72 s pro
Kombination für die Antwortgenerierung, ~8,6 Min. für die gesamte
Metrik-Berechnung (Concurrency=4). Rohdaten:
[`eval_results/ragas_raw_20260811_124845.csv`](../eval_results/ragas_raw_20260811_124845.csv)
(git-ignored — lokal reproduzierbar über `scripts/evaluate_ragas.py`).

### Gesamtranking (Mittelwert über alle 5 Metriken)

| Rang | Parser | LLM | Faithfulness | AnswerRelevancy | ContextPrecision | ContextRecall | FactualCorrectness | ⌀ |
|---|---|---|---|---|---|---|---|---|
| 1 | Unstructured | OpenAI | 0.860 | 0.806 | 0.864 | 0.833 | 0.564 | **0.785** |
| 2 | PyMuPDF4LLM | OpenAI | 0.794 | 0.812 | 0.822 | 0.792 | 0.597 | **0.763** |
| 3 | pdfplumber | OpenAI | 0.903 | 0.719 | 0.828 | 0.875 | 0.419 | **0.749** |
| 4 | PyMuPDF4LLM | Mistral | 0.808 | 0.713 | 0.822 | 0.792 | 0.499 | **0.727** |
| 5 | Unstructured | Mistral | 0.568 | 0.755 | 0.864 | 0.833 | 0.587 | **0.721** |
| 6 | pdfplumber | Mistral | 0.731 | 0.639 | 0.828 | 0.875 | 0.488 | **0.712** |
| 7 | Docling | OpenAI | 0.660 | 0.711 | 0.667 | 0.792 | 0.567 | **0.679** |
| 8 | Docling | Mistral | 0.658 | 0.438 | 0.667 | 0.792 | 0.448 | **0.601** |

### Retrieval-Qualität pro Parser (LLM-unabhängig)

`context_precision`/`context_recall` hängen ausschließlich vom Retrieval
ab, nicht vom antwortenden LLM — bei Fragen ohne Gesprächsverlauf (wie hier)
überspringt die Kette den History-aware-Contextualize-Schritt vollständig
(Abschnitt 5.3 in `DOKUMENTATION.md`), sodass die Retrieval-Metriken
zwischen OpenAI- und Mistral-Läufen **identisch** ausfallen — ein gutes
internes Konsistenz-Signal für die Korrektheit der Pipeline.

| Parser | ContextPrecision | ContextRecall |
|---|---|---|
| Unstructured | **0.864** | 0.833 |
| pdfplumber | 0.828 | **0.875** |
| PyMuPDF4LLM | 0.822 | 0.792 |
| Docling | 0.667 | 0.792 |

### Einordnung

- **Docling schneidet in beiden LLM-Kombinationen am schlechtesten ab**
  (Rang 7 und 8) — konsistent mit dem strukturellen Tabellenverlust aus dem
  qualitativen Vergleich (`PARSER.md`): niedrigste ContextPrecision (0.667)
  aller vier Parser bestätigt quantitativ, dass die zerstörte Tabellenform
  tatsächlich zu weniger präzise treffendem Retrieval führt — nicht nur zu
  einem kosmetischen Problem.
- **Unstructured (hi_res) erzielt trotz OCR-Problemen bei der TOC-Tabelle
  (siehe `PARSER.md`) die beste ContextPrecision/-Recall.** Die Fragen
  dieses Evals betreffen nur teilweise die TOC-Seite; bei den übrigen
  Inhalten scheint Unstructureds Layout-Modell brauchbare Chunk-Grenzen zu
  liefern.
- **OpenAI (gpt-4o-mini) übertrifft Mistral (mistral-small-latest) bei
  AnswerRelevancy in jeder einzelnen der 4 Parser-Kombinationen** — ein
  durchgängiges Muster, keine Ausreißer. Bei Faithfulness ist das Bild
  gemischter (Mistral gewinnt bei PyMuPDF4LLM knapp), aber bei Docling und
  Unstructured fällt Mistral deutlich ab (0.658 bzw. 0.568) — ein Hinweis,
  dass Mistral empfindlicher auf strukturell beschädigten Kontext reagiert
  als GPT-4o-mini.
- **FactualCorrectness ist über alle Kombinationen hinweg die
  niedrigste Metrik** (0.42–0.60) — das ist für dieses RAGAS-Maß typisch
  (strikte atomare Fakten-F1 gegen manuell formulierte Referenzantworten,
  die anders formuliert sein können als die Modellantwort) und sollte nicht
  direkt mit den anderen Metriken auf derselben Skala verglichen werden.
- **Gesamtsieger dieser Stichprobe:** Unstructured+OpenAI (⌀ 0.785),
  dicht gefolgt von PyMuPDF4LLM+OpenAI (⌀ 0.763). Angesichts des
  Geschwindigkeits- und Installationsaufwand-Unterschieds (siehe
  `PARSER.md`: Unstructured `hi_res` ~3,25 s/Seite vs. PyMuPDF4LLM
  ~0,02 s/Seite, plus 1,8 GB zusätzliche Abhängigkeiten) bleibt PyMuPDF4LLM
  trotz des knapp zweiten Platzes die pragmatisch bessere Wahl für den
  Produktivbetrieb — die Qualitätsdifferenz zu Unstructured ist hier klein
  (0.022), der Ressourcen-Unterschied groß.

**Wichtige Einschränkung:** n=12 Fragen auf einem 53-Seiten-Demo-Korpus ist
eine kleine Stichprobe für 96 LLM-bewertete Instanzen — geeignet, um klare
Tendenzen sichtbar zu machen (wie den Docling-Rückstand), aber nicht, um
kleine Unterschiede (z. B. Rang 4 vs. 6) als statistisch gesichert zu
interpretieren. Für eine belastbarere Aussage in der Ausarbeitung wäre ein
größeres Fragenset und/oder mehrere Wiederholungsläufe sinnvoll.
