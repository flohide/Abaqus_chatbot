"""Registry der vergleichbaren Embedding-Modelle (siehe docs/EMBEDDING.md).

Jeder Eintrag ist eine Factory `() -> Embeddings`, nicht die Instanz selbst -
so bleibt das Modul auch ohne die optionalen HuggingFace-Abhängigkeiten
(siehe requirements-embedding-comparison.txt) importierbar, solange nur die
OpenAI-Einträge tatsächlich aufgerufen werden.

Zwei grundverschiedene Provider:
- OpenAI (Produktiv-API/-Gateway, siehe `config.py`) - drei Modellgrößen.
- HuggingFace `sentence-transformers`, lokal ausgeführt (kein API-Key,
  Modellgewichte werden beim ersten Aufruf heruntergeladen und lokal
  gecacht) - vier multilingual/englische Modelle.
"""

from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..config import settings


def _resolve_openai_model(name: str) -> str:
    """Wendet auf `name` denselben Gateway-Prefix an, den auch
    `settings.embedding_model` trägt (z. B. `openai/` beim FH-SWF-Hub, siehe
    `DOKUMENTATION.md`, Abschnitt 7 "Hochschul-Gateway"). Läuft die App ohne
    Gateway (kein Prefix in `settings.embedding_model`), bleibt `name`
    unverändert."""
    current = settings.embedding_model
    if "/" in current:
        prefix = current.rsplit("/", 1)[0]
        return f"{prefix}/{name}"
    return name


def _openai_factory(name: str) -> Callable[[], Embeddings]:
    def factory() -> Embeddings:
        return OpenAIEmbeddings(
            model=_resolve_openai_model(name),
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    return factory


class _PrefixedHuggingFaceEmbeddings(Embeddings):
    """Fügt vor dem Embedding Query-/Passage-Prefixe hinzu.

    Nötig für E5-Modelle (z. B. `intfloat/multilingual-e5-large`,
    https://huggingface.co/intfloat/multilingual-e5-large): Ohne die
    "query: "/"passage: "-Prefixe, mit denen das Modell trainiert wurde,
    fällt die Retrieval-Qualität spürbar ab - ein leicht übersehener
    Stolperstein bei E5-Modellen. Für Modelle ohne Prefix-Konvention (BGE-M3,
    Sentence-Transformers) bleiben beide Prefixe leer (No-Op).

    `inner` ist injizierbar, damit Tests offline mit einer Fake-Embeddings-
    Implementierung laufen können, ohne das eigentliche HF-Modell zu laden.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        inner: Embeddings | None = None,
    ):
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._inner = inner or self._load(model_name)

    @staticmethod
    def _load(model_name: str) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self._query_prefix + text)


def _huggingface_factory(
    model_name: str, query_prefix: str = "", passage_prefix: str = ""
) -> Callable[[], Embeddings]:
    def factory() -> Embeddings:
        return _PrefixedHuggingFaceEmbeddings(model_name, query_prefix, passage_prefix)

    return factory


EMBEDDING_BACKENDS: dict[str, Callable[[], Embeddings]] = {
    "text-embedding-3-small": _openai_factory("text-embedding-3-small"),
    "text-embedding-3-large": _openai_factory("text-embedding-3-large"),
    "text-embedding-ada-002": _openai_factory("text-embedding-ada-002"),
    "multilingual-e5-large": _huggingface_factory(
        "intfloat/multilingual-e5-large", query_prefix="query: ", passage_prefix="passage: "
    ),
    "bge-m3": _huggingface_factory("BAAI/bge-m3"),
    "paraphrase-multilingual-mpnet": _huggingface_factory(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ),
    "minilm-l6-en": _huggingface_factory("sentence-transformers/all-MiniLM-L6-v2"),
}
