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

    # Mistral (für den Antwort-LLM-Vergleich in eval/run.py, siehe docs/LLM.md;
    # optional, nur nötig wenn Mistral dort als Antwort-LLM getestet wird -
    # im Streamlit-Frontend app.py ist das LLM pro Pipeline fest vorgegeben
    # und nicht frei wählbar)
    mistral_api_key: str | None = Field(None, alias="MISTRAL_API_KEY")
    mistral_model: str = Field("mistral-small-latest", alias="MISTRAL_MODEL")

    # GPT-4o (volles Modell, läuft über denselben Hub/Key wie "openai" -
    # zusätzliche Antwort-LLM-Variante im LLM-Vergleich, siehe docs/LLM.md:
    # gleiche Modellgeneration wie der Produktiv-Default gpt-4o-mini, nur
    # größer - isoliert die Modellgröße als einzige Variable)
    gpt4o_model: str = Field("gpt-4o", alias="GPT4O_MODEL")

    # Anthropic Claude - RAGAS-Richter fuer den LLM-Vergleich und
    # Best-of-Breed (siehe eval/metrics.py). Gehört bewusst zu keiner der
    # bewerteten Antwort-LLM-Familien (OpenAI/Mistral) - vermeidet den
    # Self-Preference-Bias eines gleichzeitig als Kandidat und Richter
    # eingesetzten Modells.
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    judge_model: str = Field("claude-sonnet-5", alias="JUDGE_MODEL")

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

    # Netzwerk-Robustheit fuer alle OpenAI-/Mistral-API-Clients (LLMs,
    # Embeddings) - zentral statt als Magic Numbers an mehreren Stellen
    # dupliziert.
    api_request_timeout: int = Field(60, alias="API_REQUEST_TIMEOUT")
    api_max_retries: int = Field(2, alias="API_MAX_RETRIES")


settings = Settings()
