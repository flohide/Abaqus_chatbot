"""LLM-Provider-Auswahl für die Pipelines (app.py) und den Antwort-LLM-Vergleich
(eval/run.py, siehe docs/LLM.md)."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from ..config import settings

SUPPORTED_PROVIDERS = ("openai", "mistral", "gpt4o")


def mistral_available() -> bool:
    return bool(settings.mistral_api_key)


def get_llm(provider: str = "openai") -> BaseChatModel:
    if provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            request_timeout=settings.api_request_timeout,
            max_retries=settings.api_max_retries,
        )
    if provider == "mistral":
        if not settings.mistral_api_key:
            raise ValueError(
                "MISTRAL_API_KEY ist nicht gesetzt. Bitte in .env eintragen, "
                "um das Mistral-Modell zu verwenden."
            )
        return ChatMistralAI(
            model=settings.mistral_model,
            api_key=settings.mistral_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.api_request_timeout,
            max_retries=settings.api_max_retries,
        )
    if provider == "gpt4o":
        # Läuft über denselben Hub/Key wie "openai" - volles gpt-4o statt
        # gpt-4o-mini, gleiche Modellgeneration wie der Produktiv-Default
        # (siehe docs/LLM.md).
        return ChatOpenAI(
            model=settings.gpt4o_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            request_timeout=settings.api_request_timeout,
            max_retries=settings.api_max_retries,
        )
    raise ValueError(f"Unbekannter LLM-Provider: {provider!r} (erlaubt: {SUPPORTED_PROVIDERS})")
