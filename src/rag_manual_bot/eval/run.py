"""Orchestriert die RAGAS-Evaluationsmatrizen: 4-Parser × 2-LLM (Produktiv-
Chunking fix), Chunking-Strategie × 2-LLM (Parser fix: PyMuPDF4LLM, siehe
docs/CHUNKING.md), Embedding-Modell × 2-LLM (Parser/Chunking fix, siehe
docs/EMBEDDING.md), Retrieval-Strategie × 2-LLM (Parser/Chunking/
Embedding fix, siehe docs/RETRIEVAL.md), den "Best-of-Breed"-Vergleich
(Kombination der vier Einzelsieger gegen die Produktiv-Baseline, siehe
docs/BEST_OF_BREED.md) sowie den Antwort-LLM-Vergleich (Parser/Chunking/
Embedding/Retrieval fix, siehe docs/LLM.md)."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings

from . import _ragas_compat  # noqa: F401  (muss vor jedem ragas-Import laufen)
from .dataset import EVAL_QUESTIONS
from .metrics import build_judge_metrics, score_sample
from ..config import settings
from ..ingestion.chunking_backends import CHUNKING_BACKENDS
from ..ingestion.embedding_models import EMBEDDING_BACKENDS
from ..rag.chain import build_rag_chain
from ..rag.retrieval_backends import RETRIEVAL_BACKENDS
from ..rag.vectorstore import load_vectorstore

PARSER_COLLECTIONS = {
    "pymupdf4llm": "parser_demo_pymupdf4llm",
    "pdfplumber": "parser_demo_pdfplumber",
    "docling": "parser_demo_docling",
    "unstructured": "parser_demo_unstructured",
}
CHUNKING_COLLECTIONS = {name: f"chunking_demo_{name}" for name in CHUNKING_BACKENDS}
EMBEDDING_COLLECTIONS = {name: f"embedding_demo_{name}" for name in EMBEDDING_BACKENDS}
# Retrieval-Vergleich braucht keinen eigenen Demo-Korpus (siehe
# retrieval_backends.py) - er läuft auf der bereits vorhandenen
# Produktiv-äquivalenten Chunking-Demo-Collection (PyMuPDF4LLM +
# Header+Recursive 800 + text-embedding-3-small).
RETRIEVAL_BASE_COLLECTION = "chunking_demo_header_recursive_800"
LLM_PROVIDERS = ["openai", "mistral"]
# Antwort-LLM-Vergleich (siehe docs/LLM.md) braucht wie der Retrieval-Vergleich
# keinen eigenen Demo-Korpus - läuft ebenfalls auf RETRIEVAL_BASE_COLLECTION,
# mit MMR (Produktiv-Default) und variiert nur, welches LLM antwortet. Eigene
# Liste statt LLM_PROVIDERS zu erweitern, damit die bestehenden vier
# Vergleiche (die LLM_PROVIDERS als zweite Achse nutzen) unverändert bei 2
# LLMs bleiben, statt bei jedem Lauf zusätzlich Qwen mitzutesten.
ANSWER_LLM_PROVIDERS = ["openai", "mistral", "qwen"]


def _generate_answers(
    persist_directory: Path,
    collection_name: str,
    llm_provider: str,
    extra_fields: dict,
    embeddings: Embeddings | None = None,
    retrieval_strategy: str = "mmr",
) -> list[dict]:
    """Lässt die RAG-Kette einer Wissensbasis/LLM-Kombination alle Eval-Fragen beantworten.

    `extra_fields` (z. B. {"parser": ...}, {"chunking": ...},
    {"embedding_model": ...} oder {"retrieval_strategy": ...}) wird jedem
    Record vorangestellt, um die verglichene Dimension im Ergebnis-DataFrame
    zu labeln. `embeddings` muss beim Embedding-Vergleich zu dem Modell
    passen, mit dem die Collection aufgebaut wurde (siehe
    `rag/vectorstore.py`)."""
    vectorstore = load_vectorstore(persist_directory, collection_name, embeddings=embeddings)
    chain = build_rag_chain(vectorstore, llm_provider=llm_provider, retrieval_strategy=retrieval_strategy)

    records = []
    for eq in EVAL_QUESTIONS:
        result = chain.invoke({"question": eq.question, "chat_history": []})
        records.append(
            {
                **extra_fields,
                "llm_provider": llm_provider,
                "question": eq.question,
                "reference": eq.reference,
                "answer": result["answer"],
                "contexts": [d.page_content for d in result["source_documents"]],
            }
        )
    return records


async def _score_all(records: list[dict], metrics, concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)

    async def _score_one(record: dict) -> None:
        async with semaphore:
            scores = await score_sample(
                metrics, record["question"], record["answer"], record["contexts"], record["reference"]
            )
            record.update(scores)

    await asyncio.gather(*[_score_one(r) for r in records])


def run_full_evaluation(
    parsers: list[str] | None = None,
    llm_providers: list[str] | None = None,
    concurrency: int = 4,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Führt die komplette Parser×LLM-Evaluation aus und gibt eine Liste von
    Records zurück (ein Dict pro Frage×Parser×LLM mit Antwort, Kontexten und
    allen 5 Metrik-Scores)."""
    parsers = parsers or list(PARSER_COLLECTIONS)
    llm_providers = llm_providers or LLM_PROVIDERS
    log = on_progress or (lambda msg: None)

    metrics = build_judge_metrics()

    all_records: list[dict] = []
    for parser in parsers:
        for llm_provider in llm_providers:
            t0 = time.time()
            log(f"Generiere Antworten: parser={parser}, llm={llm_provider} ...")
            records = _generate_answers(
                settings.parser_demo_vectorstore_dir,
                PARSER_COLLECTIONS[parser],
                llm_provider,
                extra_fields={"parser": parser},
            )
            log(f"  -> {len(records)} Antworten in {time.time() - t0:.1f}s")
            all_records.extend(records)

    log(f"Berechne RAGAS-Metriken für {len(all_records)} Instanzen (Concurrency={concurrency}) ...")
    t0 = time.time()
    asyncio.run(_score_all(all_records, metrics, concurrency=concurrency))
    log(f"  -> Metriken berechnet in {time.time() - t0:.1f}s")

    return all_records


def run_embedding_evaluation(
    models: list[str] | None = None,
    llm_providers: list[str] | None = None,
    concurrency: int = 4,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Führt die komplette Embedding-Modell×LLM-Evaluation aus (Parser/Chunking
    fix: PyMuPDF4LLM/Header+Recursive 800, siehe docs/EMBEDDING.md) und gibt
    eine Liste von Records zurück (ein Dict pro Frage×Embedding-Modell×LLM mit
    Antwort, Kontexten und allen 5 Metrik-Scores)."""
    models = models or list(EMBEDDING_COLLECTIONS)
    llm_providers = llm_providers or LLM_PROVIDERS
    log = on_progress or (lambda msg: None)

    metrics = build_judge_metrics()

    all_records: list[dict] = []
    for model_name in models:
        for llm_provider in llm_providers:
            t0 = time.time()
            log(f"Generiere Antworten: embedding_model={model_name}, llm={llm_provider} ...")
            records = _generate_answers(
                settings.embedding_demo_vectorstore_dir,
                EMBEDDING_COLLECTIONS[model_name],
                llm_provider,
                extra_fields={"embedding_model": model_name},
                embeddings=EMBEDDING_BACKENDS[model_name](),
            )
            log(f"  -> {len(records)} Antworten in {time.time() - t0:.1f}s")
            all_records.extend(records)

    log(f"Berechne RAGAS-Metriken für {len(all_records)} Instanzen (Concurrency={concurrency}) ...")
    t0 = time.time()
    asyncio.run(_score_all(all_records, metrics, concurrency=concurrency))
    log(f"  -> Metriken berechnet in {time.time() - t0:.1f}s")

    return all_records


def run_retrieval_evaluation(
    strategies: list[str] | None = None,
    llm_providers: list[str] | None = None,
    concurrency: int = 4,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Führt die komplette Retrieval-Strategie×LLM-Evaluation aus (Parser/
    Chunking/Embedding fix: PyMuPDF4LLM/Header+Recursive 800/
    text-embedding-3-small, siehe docs/RETRIEVAL.md) und gibt eine Liste von
    Records zurück (ein Dict pro Frage×Retrieval-Strategie×LLM mit Antwort,
    Kontexten und allen 5 Metrik-Scores)."""
    strategies = strategies or list(RETRIEVAL_BACKENDS)
    llm_providers = llm_providers or LLM_PROVIDERS
    log = on_progress or (lambda msg: None)

    metrics = build_judge_metrics()

    all_records: list[dict] = []
    for strategy in strategies:
        for llm_provider in llm_providers:
            t0 = time.time()
            log(f"Generiere Antworten: retrieval_strategy={strategy}, llm={llm_provider} ...")
            records = _generate_answers(
                settings.chunking_demo_vectorstore_dir,
                RETRIEVAL_BASE_COLLECTION,
                llm_provider,
                extra_fields={"retrieval_strategy": strategy},
                retrieval_strategy=strategy,
            )
            log(f"  -> {len(records)} Antworten in {time.time() - t0:.1f}s")
            all_records.extend(records)

    log(f"Berechne RAGAS-Metriken für {len(all_records)} Instanzen (Concurrency={concurrency}) ...")
    t0 = time.time()
    asyncio.run(_score_all(all_records, metrics, concurrency=concurrency))
    log(f"  -> Metriken berechnet in {time.time() - t0:.1f}s")

    return all_records


BEST_OF_BREED_COLLECTION = "best_of_breed"


def run_best_of_breed_evaluation(
    llm_providers: list[str] | None = None,
    concurrency: int = 4,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Vergleicht die "Best-of-Breed"-Pipeline (Unstructured + Semantic +
    text-embedding-3-large + Rerank - je der empirisch beste Kandidat aus den
    vier Einzelvergleichen, siehe docs/BEST_OF_BREED.md) gegen die frisch neu
    generierte Produktiv-äquivalente Baseline (PyMuPDF4LLM + Header+Recursive
    800 + text-embedding-3-small + MMR). Records tragen `"pipeline"` ∈
    {"best_of_breed", "baseline"}. Beide Pipelines laufen im selben Lauf mit
    demselben Richter, damit der Vergleich sauber ist. Default-LLMs sind alle
    drei Antwort-LLMs (siehe docs/LLM.md) - Qwen gewann dort das
    Gesamtranking, ist also auch hier relevant."""
    llm_providers = llm_providers or ANSWER_LLM_PROVIDERS
    log = on_progress or (lambda msg: None)

    metrics = build_judge_metrics()

    all_records: list[dict] = []
    for llm_provider in llm_providers:
        t0 = time.time()
        log(f"Generiere Antworten: pipeline=best_of_breed, llm={llm_provider} ...")
        records = _generate_answers(
            settings.embedding_demo_vectorstore_dir,
            BEST_OF_BREED_COLLECTION,
            llm_provider,
            extra_fields={"pipeline": "best_of_breed"},
            embeddings=EMBEDDING_BACKENDS["text-embedding-3-large"](),
            retrieval_strategy="rerank",
        )
        log(f"  -> {len(records)} Antworten in {time.time() - t0:.1f}s")
        all_records.extend(records)

        t0 = time.time()
        log(f"Generiere Antworten: pipeline=baseline, llm={llm_provider} ...")
        baseline_records = _generate_answers(
            settings.chunking_demo_vectorstore_dir,
            RETRIEVAL_BASE_COLLECTION,
            llm_provider,
            extra_fields={"pipeline": "baseline"},
            retrieval_strategy="mmr",
        )
        log(f"  -> {len(baseline_records)} Antworten in {time.time() - t0:.1f}s")
        all_records.extend(baseline_records)

    log(f"Berechne RAGAS-Metriken für {len(all_records)} Instanzen (Concurrency={concurrency}) ...")
    t0 = time.time()
    asyncio.run(_score_all(all_records, metrics, concurrency=concurrency))
    log(f"  -> Metriken berechnet in {time.time() - t0:.1f}s")

    return all_records


def run_llm_evaluation(
    llm_providers: list[str] | None = None,
    concurrency: int = 4,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Führt die komplette Antwort-LLM-Evaluation aus (Parser/Chunking/
    Embedding/Retrieval fix: PyMuPDF4LLM/Header+Recursive 800/
    text-embedding-3-small/MMR, siehe docs/LLM.md) und gibt eine Liste von
    Records zurück (ein Dict pro Frage×LLM mit Antwort, Kontexten und allen
    5 Metrik-Scores)."""
    llm_providers = llm_providers or ANSWER_LLM_PROVIDERS
    log = on_progress or (lambda msg: None)

    metrics = build_judge_metrics()

    all_records: list[dict] = []
    for llm_provider in llm_providers:
        t0 = time.time()
        log(f"Generiere Antworten: llm={llm_provider} ...")
        records = _generate_answers(
            settings.chunking_demo_vectorstore_dir,
            RETRIEVAL_BASE_COLLECTION,
            llm_provider,
            extra_fields={},
        )
        log(f"  -> {len(records)} Antworten in {time.time() - t0:.1f}s")
        all_records.extend(records)

    log(f"Berechne RAGAS-Metriken für {len(all_records)} Instanzen (Concurrency={concurrency}) ...")
    t0 = time.time()
    asyncio.run(_score_all(all_records, metrics, concurrency=concurrency))
    log(f"  -> Metriken berechnet in {time.time() - t0:.1f}s")

    return all_records


def run_chunking_evaluation(
    chunkers: list[str] | None = None,
    llm_providers: list[str] | None = None,
    concurrency: int = 4,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict]:
    """Führt die komplette Chunking×LLM-Evaluation aus (Parser fix: PyMuPDF4LLM,
    siehe docs/CHUNKING.md) und gibt eine Liste von Records zurück (ein Dict
    pro Frage×Chunking-Strategie×LLM mit Antwort, Kontexten und allen 5
    Metrik-Scores)."""
    chunkers = chunkers or list(CHUNKING_COLLECTIONS)
    llm_providers = llm_providers or LLM_PROVIDERS
    log = on_progress or (lambda msg: None)

    metrics = build_judge_metrics()

    all_records: list[dict] = []
    for chunker_name in chunkers:
        for llm_provider in llm_providers:
            t0 = time.time()
            log(f"Generiere Antworten: chunking={chunker_name}, llm={llm_provider} ...")
            records = _generate_answers(
                settings.chunking_demo_vectorstore_dir,
                CHUNKING_COLLECTIONS[chunker_name],
                llm_provider,
                extra_fields={"chunking": chunker_name},
            )
            log(f"  -> {len(records)} Antworten in {time.time() - t0:.1f}s")
            all_records.extend(records)

    log(f"Berechne RAGAS-Metriken für {len(all_records)} Instanzen (Concurrency={concurrency}) ...")
    t0 = time.time()
    asyncio.run(_score_all(all_records, metrics, concurrency=concurrency))
    log(f"  -> Metriken berechnet in {time.time() - t0:.1f}s")

    return all_records
