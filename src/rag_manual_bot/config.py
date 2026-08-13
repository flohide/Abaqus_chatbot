"""Zentrale, typsichere Konfiguration für Ingestion und RAG-Engine."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI (oder OpenAI-kompatibles Gateway, z. B. Hochschul-Hub)
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(None, alias="OPENAI_BASE_URL")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")

    # Mistral (für den LLM-Vergleich im Frontend; optional, nur nötig wenn
    # im Frontend "Mistral" als Modell ausgewählt wird)
    mistral_api_key: str | None = Field(None, alias="MISTRAL_API_KEY")
    mistral_model: str = Field("mistral-small-latest", alias="MISTRAL_MODEL")

    # Qwen (offenes Gewichts-Modell, läuft über denselben Hub/Key wie OpenAI,
    # siehe docs/LLM.md) - kein eigener API-Key nötig
    qwen_model: str = Field("Qwen2.5-32B-Instruct-AWQ", alias="QWEN_MODEL")

    # Pfade
    raw_pdf_dir: Path = PROJECT_ROOT / "data" / "raw_pdfs"
    vectorstore_dir: Path = PROJECT_ROOT / "vectorstore"
    collection_name: str = "abaqus_manuals"

    # Parser-Vergleichs-Demo-Korpus (siehe docs/PARSER.md)
    parser_demo_pdf_dir: Path = PROJECT_ROOT / "data" / "parser_demo_pdfs"
    parser_demo_vectorstore_dir: Path = PROJECT_ROOT / "vectorstore_parser_demo"

    # Chunking-Vergleichs-Demo-Korpus (siehe docs/CHUNKING.md)
    chunking_demo_vectorstore_dir: Path = PROJECT_ROOT / "vectorstore_chunking_demo"

    # Embedding-Modell-Vergleichs-Demo-Korpus (siehe docs/EMBEDDING.md)
    embedding_demo_vectorstore_dir: Path = PROJECT_ROOT / "vectorstore_embedding_demo"

    # Frei kombinierte Parser×Chunking×Embedding-Collections auf dem vollen
    # Produktiv-Korpus, on-demand gebaut über die app.py-Sidebar (siehe
    # ingestion/custom_demo.py)
    full_custom_vectorstore_dir: Path = PROJECT_ROOT / "vectorstore_full_custom"

    # Chunking
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(150, alias="CHUNK_OVERLAP")

    # Retrieval
    retrieval_k: int = Field(5, alias="RETRIEVAL_K")
    retrieval_fetch_k: int = Field(20, alias="RETRIEVAL_FETCH_K")


settings = Settings()
