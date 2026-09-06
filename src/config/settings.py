"""
Configuration management using Pydantic Settings.
Supports loading configurations from environment variables and .env files.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and API configurations."""

    # API Keys
    nvidia_api_key: str = Field(
        default="",
        validation_alias="NVIDIA_API_KEY",
        description="General NVIDIA NIM API key for LLM and Embedding services (fallback)"
    )
    nvidia_chat_api_key: str = Field(
        default="",
        validation_alias="NVIDIA_CHAT_API_KEY",
        description="Distinct NVIDIA API key for Chat LLM (optional override)"
    )
    nvidia_embedding_api_key: str = Field(
        default="",
        validation_alias="NVIDIA_EMBEDDING_API_KEY",
        description="Distinct NVIDIA API key for Embedding models (optional override)"
    )
    voyage_api_key: str = Field(
        default="",
        validation_alias="VOYAGE_API_KEY",
        description="Voyage AI API key for embeddings"
    )
    tavily_api_key: str = Field(
        default="",
        validation_alias="TAVILY_API_KEY",
        description="Tavily API key for web search fallback"
    )
    qdrant_api_key: str = Field(
        default="",
        validation_alias="QDRANT_API_KEY",
        description="Optional API key for authenticated Qdrant instances"
    )

    @property
    def is_voyage_embedding(self) -> bool:
        """Check if active embedding model or provider is Voyage AI."""
        if self.embedding_provider.lower() == "voyage":
            return True
        if self.embedding_provider.lower() == "nvidia":
            return False
        # Auto-detect from model name
        model_lower = self.nvidia_embedding_model.lower()
        return "voyage" in model_lower

    @property
    def effective_chat_api_key(self) -> str:
        """Return the specific chat API key if set, otherwise fallback to general nvidia_api_key."""
        return self.nvidia_chat_api_key.strip() if self.nvidia_chat_api_key else self.nvidia_api_key.strip()

    @property
    def effective_embedding_api_key(self) -> str:
        """Return the appropriate embedding API key based on selected embedding provider/model."""
        if self.is_voyage_embedding:
            if self.voyage_api_key.strip():
                return self.voyage_api_key.strip()
            # fallback if user passed key in general/embedding fields
            return self.nvidia_embedding_api_key.strip() if self.nvidia_embedding_api_key else self.nvidia_api_key.strip()
        return self.nvidia_embedding_api_key.strip() if self.nvidia_embedding_api_key else self.nvidia_api_key.strip()

    # Vector Store Config
    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias="QDRANT_URL",
        description="Qdrant server endpoint URL"
    )
    qdrant_collection_name: str = Field(
        default="rag-qdrant",
        validation_alias="QDRANT_COLLECTION_NAME",
        description="Default Qdrant collection name"
    )

    # Model Defaults
    nvidia_chat_model: str = Field(
        default="meta/llama-3.1-70b-instruct",
        validation_alias="NVIDIA_CHAT_MODEL",
        description="Default NVIDIA chat model ID or custom model ID"
    )
    nvidia_embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        validation_alias="NVIDIA_EMBEDDING_MODEL",
        description="Default embedding model ID (NVIDIA NIM or Voyage AI)"
    )
    voyage_embedding_model: str = Field(
        default="voyage-3",
        validation_alias="VOYAGE_EMBEDDING_MODEL",
        description="Default Voyage AI embedding model ID"
    )
    embedding_provider: str = Field(
        default="auto",
        validation_alias="EMBEDDING_PROVIDER",
        description="Embedding provider: 'auto', 'nvidia', or 'voyage'"
    )

    # Ingestion & Splitting Config
    chunk_size: int = Field(default=500, description="Token/character chunk size for splitting")
    chunk_overlap: int = Field(default=100, description="Chunk overlap size")
    default_doc_url: str = Field(
        default="https://arxiv.org/pdf/2307.09288.pdf",
        description="Default document URL for initial testing"
    )

    # LLM Parameters
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Max tokens for LLM generation")

    # Search Tool Parameters
    max_search_results: int = Field(default=3, gt=0, description="Max Tavily search results")

    # Model Catalogs
    available_chat_models: List[str] = [
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "deepseek-ai/deepseek-r1",
        "qwen/qwen2.5-72b-instruct",
    ]
    available_embedding_models: List[str] = [
        "nvidia/nv-embedqa-e5-v5",
        "baai/bge-m3",
        "snowflake/arctic-embed-l",
        "nvidia/nv-embedqa-mistral-7b-v2",
        "voyage-3",
        "voyage-3-lite",
        "voyage-code-3",
        "voyage-finance-2",
        "voyage-law-2",
        "voyage-multilingual-2",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
