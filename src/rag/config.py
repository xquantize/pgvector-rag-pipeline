"""Central configuration, loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_provider: Literal["ollama", "huggingface"] = "ollama"
    llm_provider: Literal["ollama"] = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    # Prefer a stronger instruct model when available, e.g. qwen2.5:7b or llama3.1
    ollama_chat_model: str = "llama3.2"

    hf_embed_model: str = "BAAI/bge-base-en-v1.5"
    hf_token: str | None = None

    postgres_user: str = "rag"
    postgres_password: str = "rag"
    postgres_db: str = "ragdb"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    embedding_dim: int = 768
    # vector = ANN only; hybrid = ANN + Postgres FTS fused with RRF in SQL
    retrieval_mode: Literal["vector", "hybrid"] = "hybrid"

    # Generation (grounded / low-temp is the usual production default for FAQ RAG)
    llm_temperature: float = 0.1
    llm_num_predict: int = 256
    context_chunk_chars: int = 1200
    judge_temperature: float = 0.0

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
