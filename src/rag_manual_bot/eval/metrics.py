"""RAGAS-Metriken-Setup: fester OpenAI-Richter für alle Varianten.

Der Richter (Judge-LLM + Embeddings) wird bewusst für ALLE 8 Parser×LLM-
Kombinationen konstant gehalten (immer OpenAI/Hub), unabhängig davon, ob
die zu bewertende Antwort von OpenAI oder Mistral stammt. Würde man
stattdessen je nach getesteter Variante das jeweils gleiche Modell auch als
Richter einsetzen, entstünde ein Self-Preference-Bias (LLMs bewerten eigene
Ausgaben tendenziell günstiger) — die Ergebnisse zwischen den Varianten
wären dann nicht mehr fair vergleichbar.
"""

from dataclasses import dataclass

from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from ..config import settings


@dataclass
class EvalMetrics:
    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecisionWithReference
    context_recall: ContextRecall
    factual_correctness: FactualCorrectness


def build_judge_metrics(cache_dir: str = ".ragas_cache") -> EvalMetrics:
    """Baut alle 5 Metriken mit einem festen OpenAI-Richter (gecacht)."""
    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    cache = DiskCacheBackend(cache_dir=cache_dir)
    judge = llm_factory(settings.llm_model, provider="openai", client=client, cache=cache)
    embeddings = RagasOpenAIEmbeddings(client=client, model=settings.embedding_model)

    return EvalMetrics(
        faithfulness=Faithfulness(llm=judge),
        answer_relevancy=AnswerRelevancy(llm=judge, embeddings=embeddings),
        context_precision=ContextPrecisionWithReference(llm=judge),
        context_recall=ContextRecall(llm=judge),
        factual_correctness=FactualCorrectness(llm=judge),
    )


async def score_sample(
    metrics: EvalMetrics,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
) -> dict[str, float]:
    """Berechnet alle 5 Metriken für eine einzelne Frage-Antwort-Instanz."""
    results = {}
    try:
        results["faithfulness"] = (
            await metrics.faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            )
        ).value
    except Exception as exc:  # Einzelne Metrik-Fehler nicht den ganzen Lauf abbrechen lassen
        results["faithfulness"] = float("nan")
        results["faithfulness_error"] = str(exc)

    try:
        results["answer_relevancy"] = (
            await metrics.answer_relevancy.ascore(user_input=question, response=answer)
        ).value
    except Exception as exc:
        results["answer_relevancy"] = float("nan")
        results["answer_relevancy_error"] = str(exc)

    try:
        results["context_precision"] = (
            await metrics.context_precision.ascore(
                user_input=question, reference=reference, retrieved_contexts=contexts
            )
        ).value
    except Exception as exc:
        results["context_precision"] = float("nan")
        results["context_precision_error"] = str(exc)

    try:
        results["context_recall"] = (
            await metrics.context_recall.ascore(
                user_input=question, retrieved_contexts=contexts, reference=reference
            )
        ).value
    except Exception as exc:
        results["context_recall"] = float("nan")
        results["context_recall_error"] = str(exc)

    try:
        results["factual_correctness"] = (
            await metrics.factual_correctness.ascore(response=answer, reference=reference)
        ).value
    except Exception as exc:
        results["factual_correctness"] = float("nan")
        results["factual_correctness_error"] = str(exc)

    return results
