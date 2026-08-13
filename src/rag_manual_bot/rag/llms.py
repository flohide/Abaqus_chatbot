"""LLM-Provider-Auswahl für den Modellvergleich im Frontend."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from ..config import settings

SUPPORTED_PROVIDERS = ("openai", "mistral", "qwen")


def mistral_available() -> bool:
    return bool(settings.mistral_api_key)


def get_llm(provider: str = "openai") -> BaseChatModel:
    if provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
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
        )
    if provider == "qwen":
        # Läuft über denselben Hub/Key wie "openai" (siehe docs/LLM.md) - Qwen
        # ist ein offenes Gewichts-Modell, das der FH-SWF-Hub kostenlos mit
        # anbietet, kein separater API-Key nötig.
        return ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"Unbekannter LLM-Provider: {provider!r} (erlaubt: {SUPPORTED_PROVIDERS})")
