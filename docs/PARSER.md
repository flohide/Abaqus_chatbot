# PDF-Parser-Vergleich für technische Handbücher

Ergänzende Evaluierung zu [`DOKUMENTATION.md`](DOKUMENTATION.md): Vier gängige
Python-PDF-Parser wurden am selben Ausschnitt eines Abaqus-Handbuchs getestet,
um die Wahl von **PyMuPDF4LLM** als Produktiv-Parser empirisch zu begründen.

## Kandidaten

| Parser | Version | Ansatz |
|---|---|---|
| **PyMuPDF4LLM** | 1.28.0 | Regelbasiert, nutzt PyMuPDFs Textlayer + Layout-Heuristiken; Tesseract-OCR nur als Fallback für Bildseiten |
| **pdfplumber** | 0.11.10 | Regelbasiert (pdfminer.six-Backend), Textextraktion + optionale Linien-basierte Tabellenerkennung |
| **Docling** (IBM) | 2.118.0 | Deep-Learning-Layoutanalyse (Objekterkennung) + TableFormer-Modell für Tabellenstruktur + RapidOCR |
| **Unstructured** | 0.25.2 (+ `unstructured-inference` 1.6.13) | Wahlweise `fast` (pdfminer, wie pdfplumber) oder `hi_res` (YOLO-Layout-Modell + Tabellenmodell + Tesseract-OCR) |

## Methodik

Aus `Abaqus2017_GETTINGSTARTED.pdf` wurden 6 repräsentative Seiten extrahiert
und als eigenständiges Test-PDF gespeichert:

- **2 Inhaltsverzeichnis-Seiten** (dichter, punktierter Zwei-Spalten-Text mit
  rechtsbündigen Seitenzahlen — kein echtes Grid, keine sichtbaren Linien)
- **2 Fließtext-Seiten** mit einer **linienlosen Mini-Tabelle**
  ("Degrees of Freedom", Spalten: Nummer + Beschreibung)
- **2 weitere Fließtext-Seiten** mit Aufzählungen und einer Abbildung

Alle vier Parser wurden mit Standardeinstellungen (bzw. den für Tabellen
empfohlenen Optionen) auf exakt demselben 6-Seiten-PDF ausgeführt, auf einem
Apple-Silicon-Mac **ohne GPU-Beschleunigung** (PyTorch lief auf CPU).

⚠️ **Wichtige Einschränkung dieses Tests:** Die getesteten Seiten stammen aus
einem born-digital, bereits gut strukturierten technischen Handbuch. Doclings
und Unstructureds Deep-Learning-Ansätze sind primär für stark visuell
komplexe oder gescannte Dokumente konzipiert — dort dürften sie relativ zu
regelbasierten Parsern deutlich besser abschneiden als hier. Auf einer GPU
wären die Laufzeiten der ML-basierten Parser ebenfalls signifikant niedriger.

## Ergebnisse

| Kriterium | PyMuPDF4LLM | pdfplumber | Docling | Unstructured `fast` | Unstructured `hi_res` |
|---|---|---|---|---|---|
| **Laufzeit (6 Seiten)** | **1,05 s** | **0,36 s** | 133,1 s | 4,75 s | 73,9 s |
| **Hochgerechnet auf 2.500 Seiten** | ~7 Min (real gemessen) | ~2,5 Min | ~15 Std. | ~33 Min | ~8,5 Std. |
| **TOC-Tabelle (2-spaltig, punktiert)** | Saubere Markdown-Tabelle, korrekte Seitenzahlen | Als Fließtext (keine Tabellenerkennung, `extract_tables()` liefert 0 Treffer) | Als Tabelle erkannt, aber Zeilen/Spalten teils falsch verschmolzen | Ein einziger Textblock (`UncategorizedText`), keine Struktur | Als Tabelle erkannt, **aber Zellinhalt OCR-verstümmelt** (z. B. Seitenzahl „1048" → „104s", Text durch Punktraster-Fehlinterpretation korrumpiert) |
| **DOF-Mini-Tabelle (linienlos)** | Korrekte Markdown-Tabelle (`\|4\|Rotation about the 1-axis\|...`) | Korrekter Fließtext, aber keine Tabellenstruktur | **Zerfällt in unstrukturierte Bullet-Liste**, Tabellenform geht komplett verloren | Stark fragmentiert: Zahlen und Text als separate, teils falsch klassifizierte Elemente | Sauber als `ListItem`-Elemente erkannt (zufällig korrekt, da Layout-Modell hier keine Tabelle detektierte) |
| **Bullet-Zeichen (•)** | Korrekt als `-` gerendert | **Kaputt**: `(cid:129)` (unaufgelöster Font-Glyph) | Korrekt gerendert | **Kaputt**: `(cid:129)` (gleiches pdfminer-Problem wie pdfplumber) | Korrekt gerendert |
| **Überschriften-Struktur** | `#`/`##`/`###`, zuverlässig | Keine (nur Fließtext) | `#`/`##`, gut, aber gelegentlich zerrissene Aufzählungen | Kategorisiert (`Title`/`Header`/`NarrativeText`), aber keine Hierarchie-Ebenen | Wie `fast`, plus bessere Listenerkennung |
| **Seiten-Metadaten** | Nativ pro Seite (`page_chunks=True`) | Nativ pro Seite (Iteration über `pdf.pages`) | Nur global exportierbar; Seitenzuordnung erfordert zusätzliche Arbeit über `item.prov` | Pro Element vorhanden (`metadata.page_number`) | Pro Element vorhanden |
| **OCR-Fallback für Bildseiten** | Tesseract, zuverlässig (im echten Ingestion-Lauf bestätigt) | Kein OCR | RapidOCR, schlug bei denselben Bildinhalten teils fehl (`returned empty result`) | Kein OCR (`fast`) | Tesseract, aber s.o. Tabellen-OCR-Qualität fraglich |
| **Installationsgröße (venv)** | Schlank (~50 MB) | Schlank (~10 MB) | **+1,8 GB** (PyTorch, Transformers, Layout-/Tabellenmodelle) | +~600 MB (`fast`), zusätzlich Layout-/Tabellenmodelle für `hi_res` | s. `fast` |
| **Nebenwirkung bei Installation** | — | — | — | Downgrade von `numpy` 2.5.1 → 2.4.6 im gemeinsamen venv (Dependency-Konflikt-Risiko bei paralleler Nutzung mit anderen ML-Bibliotheken) | s. `fast` |

## Beispiel: dieselbe Mini-Tabelle, vier Parser

Quelle: "Characterizing elements", Seite 158 (Degrees-of-Freedom-Tabelle ohne
sichtbare Gitterlinien).

**PyMuPDF4LLM** (korrekt):
```
|4|Rotation about the 1-axis|
|---|---|
|5|Rotation about the 2-axis|
|6|Rotation about the 3-axis|
|7|Warping in open-section beam elements|
```

**Docling** (Struktur verloren):
```
- Rotation about the 1-axis 4 Rotation about the 2-axis 5 Rotation about
  the 3-axis 6 Warping in open-section beam elements 7 ...
```

**pdfplumber** (kein Tabellen-Match, aber lesbarer Fließtext):
```
4 Rotation about the 1-axis
5 Rotation about the 2-axis
6 Rotation about the 3-axis
```

**Unstructured `hi_res`** (korrekte Listenelemente, aber keine Tabellenform):
```
[ListItem] 4 Rotation about the 1-axis
[ListItem] 5 Rotation about the 2-axis
[ListItem] 6 Rotation about the 3-axis
```

## Beispiel: TOC-Tabelle, Unstructured `hi_res` (OCR-Korruption)

Zum Vergleich ein Ausschnitt aus der von Unstructured per Tabellenmodell +
OCR erkannten HTML-Tabelle — sie erkennt korrekt eine Tabellenstruktur,
liest den Inhalt aber fehlerhaft (Punktraster wird als Buchstabenfolge
fehlinterpretiert):

```html
<tr><td colspan="2">Circuit board Crop teSt.............ccccccccccceeeeeeeeeenneeeeeeeeeeeceeeceenaaeeeeeeeeeesecceeaaeeeeeeeeeeeeesseeeeeeeeeeeeeeeseeeeaas 104s</td></tr>
```

(Korrekter Text: *„Circuit board drop test.....................1049"*)

## Fazit

Für dieses konkrete, textbasierte Handbuch-Korpus ist **PyMuPDF4LLM** allen
getesteten Alternativen sowohl in **Geschwindigkeit** (bis zu 127× schneller
als Docling) als auch in **inhaltlicher Genauigkeit** bei den getesteten
Tabellen überlegen — trotz des Rufs von Docling und Unstructured `hi_res`
als besonders tabellenstarke Parser. Zwei plausible Gründe:

1. Die Abaqus-Handbücher sind **born-digital** mit sauberem Textlayer;
   PyMuPDF4LLM kann diesen direkt auslesen, während die ML-basierten Parser
   auf Bild-Rendering + Objekterkennung/OCR ausweichen — ein Umweg, der bei
   bereits digital vorliegendem Text eher Fehler einführt als behebt.
2. Die Tabellen in diesen Handbüchern sind **linienlos** (kein sichtbares
   Grid), was pdfplumbers Default-Strategie komplett scheitern lässt und
   auch die Deep-Learning-Layoutmodelle vor die schwierigere Aufgabe stellt,
   Tabellengrenzen rein visuell zu erschließen — anstatt (wie PyMuPDF4LLM)
   einfach die zugrunde liegende Textposition/-ausrichtung auszuwerten.

Für stark visuelle oder gescannte Dokumente (z. B. handschriftliche Formulare,
komplexe mehrspaltige Layouts mit echten Gitterlinien-Tabellen) wäre eine
erneute Evaluierung sinnvoll — dort dürften Docling und Unstructured
`hi_res` ihre Stärken eher ausspielen können.

**Entscheidung für dieses Projekt:** PyMuPDF4LLM bleibt der Produktiv-Parser
(siehe [`DOKUMENTATION.md`, Abschnitt 4.1](DOKUMENTATION.md#41-loader-loaderpy)).

## Nachtrag: Live-Vergleich im Frontend (~50-Seiten-Demo-Korpus)

Für einen interaktiven Vergleich wurden alle vier Parser zusätzlich auf
einem größeren, festen 53-Seiten-Demo-Korpus ausgeführt (TOC-Seiten,
DOF-Tabelle, ein vollständiges 30-seitiges Tutorial-Kapitel sowie ein
15-seitiges Analyse-Kapitel) und jeweils in eine eigene Chroma-Collection
gespeichert (`scripts/build_parser_demo_corpus.py`). Im Streamlit-Frontend
(`app.py`) lässt sich die Wissensbasis live umschalten, um dieselbe Frage
gegen alle vier Parser-Varianten zu stellen.

### Korrektur der Laufzeit-Hochrechnung

Der zweite Lauf zeigte einen wichtigen methodischen Punkt: Docling und
Unstructured laden beim ersten Aufruf Deep-Learning-Modelle (Layout-/
Tabellenmodelle, ggf. inkl. Download) — dieser **einmalige Cold-Start**
dominierte die oben gemessenen 133,1 s bzw. 73,9 s für nur 6 Seiten. Mit
bereits lokal zwischengespeicherten Modellen (zweiter Lauf, 53 Seiten)
lag die tatsächliche Verarbeitungszeit deutlich niedriger:

| Parser | 53 Seiten, warmer Cache | Daraus abgeleitet: ms/Seite | Vorherige Hochrechnung (kalt) |
|---|---|---|---|
| Docling | **49,9 s** | ~0,94 s/Seite | ~15 Std. (auf Basis von 22,2 s/Seite kalt) |
| Unstructured `hi_res` | **172,1 s** | ~3,25 s/Seite | ~8,5 Std. (auf Basis von 12,3 s/Seite kalt) |

Auf den vollen Produktiv-Korpus (~5.100 Seiten) hochgerechnet, ergäbe der
warme Cache **~80 Min. (Docling)** bzw. **~4,6 Std. (Unstructured)** statt
der ursprünglich geschätzten Stunden. Die Kernaussage des Vergleichs
(Genauigkeit bei linienlosen Tabellen, s. o.) bleibt davon unberührt — die
Geschwindigkeits-Hochrechnung war jedoch durch den Cold-Start-Overhead
deutlich zu pessimistisch. **Praktische Konsequenz:** Für einen realistischen
Laufzeit-Vergleich sollte immer ein zweiter, "warmer" Lauf gemessen werden,
nicht nur der erste.

### Überraschung: LLM-Antworten oft robust gegenüber Extraktionsfehlern

Bei zwei gezielten Testfragen gegen alle vier Demo-Collections — u. a.
*"Welcher DOF-Wert steht für Rotation um die 2-Achse?"* (betrifft die bei
Docling zuvor komplett zerfallene Tabellenstruktur) — lieferten **alle vier
Parser dieselbe korrekte Antwort ("6")**. Obwohl Doclings Rohtext die
Tabellenform verliert (`"...Rotation about the 1-axis 4 Rotation about the
2-axis 5..."`), reichte die verbleibende Nähe von Zahl und Beschreibung im
Fließtext für GPT-4o-mini aus, um die Information trotzdem korrekt zu
extrahieren. Das relativiert die Praxisrelevanz einzelner
Strukturverluste: **nicht jeder Extraktionsfehler eines Parsers führt auch
zu einer falschen Chatbot-Antwort** — das LLM kompensiert leichte bis
mittlere Strukturschäden im Kontext erstaunlich gut. Für Fragen, die exakte
tabellarische Mehrfach-Spalten-Zuordnung über viele Zeilen hinweg erfordern
(z. B. "liste alle DOF-Werte mit Beschreibung auf"), ist trotzdem zu
erwarten, dass die sauberere PyMuPDF4LLM-Tabellenstruktur zuverlässiger
bleibt — das wäre eine sinnvolle weitere Testfrage für die Ausarbeitung.
