"""Evaluations-Fragenset für den RAGAS-Vergleich (Parser × LLM).

Alle Fragen sind so gewählt, dass sie ausschließlich aus dem ~53-Seiten
Parser-Vergleichs-Demo-Korpus (siehe ingestion/parser_demo.py) beantwortbar
sind, damit alle 4 Parser-Varianten × 2 LLMs auf exakt derselben Grundlage
verglichen werden. Referenzantworten wurden manuell anhand des tatsächlichen
Handbuchtexts (nicht geraten) formuliert — Grundlage für ContextRecall und
FactualCorrectness.
"""

from dataclasses import dataclass


@dataclass
class EvalQuestion:
    question: str
    reference: str


EVAL_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="Wie erstellt man im Assembly-Modul eine Instanz eines Parts?",
        reference=(
            "Im Model Tree den Assembly-Container erweitern und auf 'Instances' "
            "doppelklicken, um eine Instanz zu erstellen. Im Create Instance-Dialog "
            "das gewünschte Part aus der Parts-Liste auswählen und OK klicken."
        ),
    ),
    EvalQuestion(
        question="Welcher DOF-Wert (Degree of Freedom) steht für 'Rotation about the 3-axis'?",
        reference="Der DOF-Wert 6 steht für Rotation um die 3-Achse.",
    ),
    EvalQuestion(
        question="Welcher DOF-Wert steht für 'Warping in open-section beam elements'?",
        reference="Der DOF-Wert 7 steht für Warping (Verwölbung) in offenen Balkenquerschnitts-Elementen.",
    ),
    EvalQuestion(
        question=(
            "Was unterscheidet lineare (First-Order) Elemente von quadratischen "
            "(Second-Order) Elementen bezüglich der Knoten?"
        ),
        reference=(
            "Lineare Elemente haben Knoten nur an den Ecken (z. B. der 8-Knoten-Brick "
            "C3D8) und nutzen lineare Interpolation in jede Richtung. Quadratische "
            "Elemente haben zusätzlich Knoten auf den Kantenmitten (z. B. der "
            "20-Knoten-Brick C3D20) und nutzen quadratische Interpolation."
        ),
    ),
    EvalQuestion(
        question="Bietet Abaqus/Explicit volle oder nur reduzierte Integration an?",
        reference=(
            "Abaqus/Explicit bietet nur reduzierte Integration an. Abaqus/Standard "
            "bietet sowohl volle als auch reduzierte Integration an."
        ),
    ),
    EvalQuestion(
        question=(
            "In welcher Reihenfolge muss man in Abaqus/CAE vorgehen: zuerst die "
            "Assembly oder zuerst die Analysis Steps erstellen?"
        ),
        reference=(
            "Man muss zuerst die Assembly erstellen, bevor man Analysis Steps "
            "erstellt. Diese Reihenfolge im Model Tree ist fest vorgegeben und kann "
            "nicht geändert werden."
        ),
    ),
    EvalQuestion(
        question="Wodurch definiert man die Eigenschaften eines Parts in Abaqus/CAE?",
        reference=(
            "Über Sections (Abschnitte). Nachdem man eine Section erstellt hat, kann "
            "sie einem Part oder Bereichen davon zugewiesen werden."
        ),
    ),
    EvalQuestion(
        question="Sind Randbedingungen (Boundary Conditions) und Lasten (Loads) in Abaqus step-abhängig?",
        reference=(
            "Ja, prescribed conditions wie Loads und Boundary Conditions sind "
            "step-abhängig (step-dependent)."
        ),
    ),
    EvalQuestion(
        question=(
            "Welches Keyword definiert in Abaqus einen gekoppelten "
            "thermisch-elektrisch-strukturellen Analyseschritt?"
        ),
        reference="*STEP, gefolgt von *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL.",
    ),
    EvalQuestion(
        question="Wofür steht der Randbedingungstyp 'MVP' bei der Eddy-Current-Analyse?",
        reference=(
            "MVP steht für eine uniforme (gleichförmige) Randbedingung des "
            "magnetischen Vektorpotentials (magnetic vector potential). MVPNU steht "
            "für die nicht-uniforme Variante."
        ),
    ),
    EvalQuestion(
        question=(
            "Was wird als typisches Anwendungsbeispiel für Induktionserwärmung "
            "(induction heating) im Kontext der Eddy-Current-Analyse genannt?"
        ),
        reference="Ein Induktionskocher (induction cooker) wird als Beispiel genannt.",
    ),
    EvalQuestion(
        question=(
            "Auf welcher Seite im Inhaltsverzeichnis beginnt der Abschnitt "
            "'Creating and positioning an instance of the pin'?"
        ),
        reference="Seite 1161.",
    ),
]
