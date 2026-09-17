# RAGAS-Evaluations-Fragenset

Auflistung der 12 Fragen und Referenzantworten aus
[`src/rag_manual_bot/eval/dataset.py`](../src/rag_manual_bot/eval/dataset.py)
zur manuellen Durchsicht. Dies ist
eine reine Lesehilfe — die Quelle der Wahrheit bleibt `dataset.py`; Änderungen
müssen dort vorgenommen werden, nicht hier.

Alle Fragen sind so gewählt, dass sie ausschließlich aus dem ~53-Seiten
Parser-Vergleichs-Demo-Korpus (siehe `ingestion/parser_demo.py`,
`data/parser_demo_pdfs/demo_corpus.pdf`) beantwortbar sind, damit alle
Vergleichsstudien auf exakt derselben Grundlage laufen.

| # | Frage | Referenzantwort |
|---|---|---|
| 1 | Wie erstellt man im Assembly-Modul eine Instanz eines Parts? | Im Model Tree den Assembly-Container erweitern und auf 'Instances' doppelklicken, um eine Instanz zu erstellen. Im Create Instance-Dialog das gewünschte Part aus der Parts-Liste auswählen und OK klicken. |
| 2 | Welcher DOF-Wert (Degree of Freedom) steht für 'Rotation about the 3-axis'? | Der DOF-Wert 6 steht für Rotation um die 3-Achse. |
| 3 | Welcher DOF-Wert steht für 'Warping in open-section beam elements'? | Der DOF-Wert 7 steht für Warping (Verwölbung) in offenen Balkenquerschnitts-Elementen. |
| 4 | Was unterscheidet lineare (First-Order) Elemente von quadratischen (Second-Order) Elementen bezüglich der Knoten? | Lineare Elemente haben Knoten nur an den Ecken (z. B. der 8-Knoten-Brick C3D8) und nutzen lineare Interpolation in jede Richtung. Quadratische Elemente haben zusätzlich Knoten auf den Kantenmitten (z. B. der 20-Knoten-Brick C3D20) und nutzen quadratische Interpolation. |
| 5 | Bietet Abaqus/Explicit volle oder nur reduzierte Integration an? | Abaqus/Explicit bietet nur reduzierte Integration an. Abaqus/Standard bietet sowohl volle als auch reduzierte Integration an. |
| 6 | In welcher Reihenfolge muss man in Abaqus/CAE vorgehen: zuerst die Assembly oder zuerst die Analysis Steps erstellen? | Man muss zuerst die Assembly erstellen, bevor man Analysis Steps erstellt. Diese Reihenfolge im Model Tree ist fest vorgegeben und kann nicht geändert werden. |
| 7 | Wodurch definiert man die Eigenschaften eines Parts in Abaqus/CAE? | Über Sections (Abschnitte). Nachdem man eine Section erstellt hat, kann sie einem Part oder Bereichen davon zugewiesen werden. |
| 8 | Sind Randbedingungen (Boundary Conditions) und Lasten (Loads) in Abaqus step-abhängig? | Ja, prescribed conditions wie Loads und Boundary Conditions sind step-abhängig (step-dependent). |
| 9 | Welches Keyword definiert in Abaqus einen gekoppelten thermisch-elektrisch-strukturellen Analyseschritt? | \*STEP, gefolgt von \*COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL. |
| 10 | Wofür steht der Randbedingungstyp 'MVP' bei der Eddy-Current-Analyse? | MVP steht für eine uniforme (gleichförmige) Randbedingung des magnetischen Vektorpotentials (magnetic vector potential). MVPNU steht für die nicht-uniforme Variante. |
| 11 | Was wird als typisches Anwendungsbeispiel für Induktionserwärmung (induction heating) im Kontext der Eddy-Current-Analyse genannt? | Ein Induktionskocher (induction cooker) wird als Beispiel genannt. |
| 12 | Auf welcher Seite im Inhaltsverzeichnis beginnt der Abschnitt 'Creating and positioning an instance of the pin'? | Seite 1161. |

## Herkunft der Fragen (Seitenbereiche im Demo-Korpus)

Zur Einordnung, aus welchem Teil des Demo-Korpus (`ingestion/parser_demo.py::DEMO_PAGE_SPECS`)
welche Frage stammt:

| Fragen | Quelle | Seitenbereich |
|---|---|---|
| 12 | Abaqus2017_GETTINGSTARTED.pdf, Inhaltsverzeichnis | S. 9–10 |
| 2, 3, 1 | Abaqus2017_GETTINGSTARTED.pdf, DOF-Tabelle + Instanz-Erstellung | S. 213–215 |
| 4, 5 | Abaqus2017_GETTINGSTARTED.pdf, Elementcharakterisierung | S. 159–161 |
| 6, 7, 8 | Abaqus2017_GETTINGSTARTED.pdf, Tutorial-Kapitel | S. 1060–1089 |
| 9, 10, 11 | Abaqus2017_ANALYSIS.pdf, Analyse-Kapitel (Eddy-Current) | S. 340–354 |
