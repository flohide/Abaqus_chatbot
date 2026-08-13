import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Bewusst nur dataset.py importieren, nicht eval.metrics/eval.run — diese
# brauchen ragas (requirements-eval.txt), das nicht Teil der
# Standard-Installation ist. dataset.py selbst hat keine ragas-Abhängigkeit.
from rag_manual_bot.eval.dataset import EVAL_QUESTIONS, EvalQuestion


def test_eval_questions_well_formed():
    assert len(EVAL_QUESTIONS) >= 10
    for eq in EVAL_QUESTIONS:
        assert isinstance(eq, EvalQuestion)
        assert eq.question.strip()
        assert eq.reference.strip()
        assert eq.question.endswith("?")


def test_eval_questions_are_unique():
    questions = [eq.question for eq in EVAL_QUESTIONS]
    assert len(questions) == len(set(questions))
