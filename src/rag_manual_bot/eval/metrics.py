"""RAGAS-Metriken-Setup: Richter je Studie bewusst unterschiedlich gewaehlt.

Vier der sechs Vergleichsstudien (Parser, Chunking, Embedding, Retrieval)
laufen mit dem festen OpenAI-Richter (gpt-4o-mini) - dort ist das
Antwort-LLM fuer alle verglichenen Varianten ohnehin konstant, ein
Self-Preference-Bias wuerde alle Varianten gleichermassen betreffen und die
relative Rangfolge kaum verzerren (siehe docs/RAGAS.md).

Der Antwort-LLM-Vergleich (docs/LLM.md) und Best-of-Breed
(docs/BEST_OF_BREED.md) laufen dagegen bewusst mit einem unabhaengigen
Claude-Richter (provider="anthropic") - dort treten OpenAI-Modelle direkt
gegen Mistral (mistral-small-latest) an, und ein Richter aus der OpenAI-Familie waere selbst
Kandidat und Richter zugleich (Self-Preference-Bias, empirisch bestaetigt:
siehe Projektverlauf - GPT-4os Vorsprung schrumpfte bzw. kehrte sich unter
dem unabhaengigen Richter um). Claude ist bei keinem der Vergleiche selbst
Antwort-LLM-Kandidat.

Kostenabwaegung: Claude Sonnet 5 kostet pro Token ca. 20-25x mehr als
gpt-4o-mini (bestaetigt: $3/$15 vs. $0.15/$0.60 pro 1M Tokens) - deshalb
bewusst nur fuer die zwei Studien mit echtem Bias-Risiko, nicht global.

Die Embeddings fuer AnswerRelevancy bleiben in allen Faellen bei OpenAI -
dort geht es um reine Kosinus-Aehnlichkeit, keine qualitative Bewertung,
also kein Self-Preference-Risiko.
"""

from dataclasses import dataclass

from anthropic import AsyncAnthropic
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


def _build_openai_judge(cache_dir: str | None):
    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    cache = DiskCacheBackend(cache_dir=cache_dir or ".ragas_cache")
    judge = llm_factory(settings.llm_model, provider="openai", client=client, cache=cache)
    return judge, client


def _build_anthropic_judge(cache_dir: str | None):
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY fehlt in .env - der Claude-Richter wird fuer den "
            "LLM-Vergleich und Best-of-Breed benoetigt, siehe eval/metrics.py."
        )
    if cache_dir is None:
        # Cache-Verzeichnis ist per Richter-Modell namensraumgetrennt statt eines
        # global geteilten ".ragas_cache" - ragas' eigener cacher() generiert den
        # Cache-Key nur aus Funktionsname + Aufruf-Argumenten (Frage/Kontext/
        # Antwort/Referenz), NICHT aus der Richter-Instanz selbst. Ein geteilter
        # Cache-Ordner ueber einen Richter-Wechsel hinweg liefert deshalb
        # stillschweigend alte Urteile des vorherigen Richters zurueck, sobald
        # (Frage, Kontext, [Antwort]) mit einem frueheren Lauf uebereinstimmt.
        # Empirisch beobachtet: 73-93% alte Cache-Treffer bei geteiltem Cache.
        cache_dir = f".ragas_cache_{settings.judge_model}"
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    cache = DiskCacheBackend(cache_dir=cache_dir)
    judge = llm_factory(settings.judge_model, provider="anthropic", client=client, cache=cache)
    # Kompatibilitäts-Fix (analog zu _ragas_compat.py): ragas/instructor
    # schicken standardmäßig temperature/top_p an Messages.create() mit -
    # die installierte anthropic-SDK (1.5.0) nimmt beide Parameter nicht
    # mehr entgegen (TypeError: unexpected keyword argument 'temperature'),
    # ohne diesen Pop schlagen alle 5 Metriken fehl.
    judge.model_args.pop("temperature", None)
    judge.model_args.pop("top_p", None)
    # ragas/instructor-Default (1024) reicht bei Faithfulness/FactualCorrectness
    # nicht immer aus (Claim-Zerlegung bei laengeren Antworten) - beobachtet als
    # "The output is incomplete due to a max_tokens length limit." bei ca. 4% der
    # Instanzen im Best-of-Breed-Demo-Rerun (14/360 Zellen).
    judge.model_args["max_tokens"] = 4096
    return judge, None


def build_judge_metrics(provider: str = "openai", cache_dir: str | None = None) -> EvalMetrics:
    """Baut alle 5 Metriken mit einem festen Richter (gecacht).

    `provider="openai"` (Default): gpt-4o-mini, wie urspruenglich fuer alle
    sechs Studien. `provider="anthropic"`: unabhaengiger Claude-Richter, nur
    fuer den LLM-Vergleich und Best-of-Breed vorgesehen.
    """
    if provider == "anthropic":
        judge, client = _build_anthropic_judge(cache_dir)
    elif provider == "openai":
        judge, client = _build_openai_judge(cache_dir)
    else:
        raise ValueError(f"Unbekannter Richter-Provider: {provider!r} (erwartet 'openai' oder 'anthropic')")

    # Fuer provider="openai" liefert der Richter-Client bereits einen
    # passenden AsyncOpenAI-Client - wiederverwenden statt einen zweiten
    # aufzubauen (anthropic liefert keinen OpenAI-Client, dafuer bleibt der
    # eigene Embeddings-Client noetig).
    embeddings_client = client if provider == "openai" else AsyncOpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url
    )
    embeddings = RagasOpenAIEmbeddings(client=embeddings_client, model=settings.embedding_model)

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
