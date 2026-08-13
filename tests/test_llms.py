"""Tests für die LLM-Provider-Auswahl (siehe docs/LLM.md).

Laufen vollständig offline: Konstruieren eines Chat-Clients prüft nur die
übergebenen Parameter, macht keinen API-Call.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_manual_bot.config import settings
from rag_manual_bot.rag.llms import SUPPORTED_PROVIDERS, get_llm


def test_supported_providers_includes_qwen():
    assert SUPPORTED_PROVIDERS == ("openai", "mistral", "qwen")


def test_qwen_uses_same_hub_and_key_as_openai():
    llm = get_llm("qwen")
    assert llm.model_name == settings.qwen_model
    assert llm.openai_api_base == settings.openai_base_url


def test_openai_uses_configured_model():
    llm = get_llm("openai")
    assert llm.model_name == settings.llm_model


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unbekannter LLM-Provider"):
        get_llm("does-not-exist")
