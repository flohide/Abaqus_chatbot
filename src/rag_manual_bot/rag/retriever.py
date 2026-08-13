"""Retriever-Konfiguration: MMR reduziert redundante Treffer aus derselben Seite/Sektion."""

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever

from ..config import settings


def build_retriever(vectorstore: Chroma) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retrieval_k,
            "fetch_k": settings.retrieval_fetch_k,
        },
    )
