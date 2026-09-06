"""
Factory service for NVIDIA and Voyage AI LLM and Embedding models.
"""

from typing import Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger("ModelClientFactory")


class NVIDIAClientFactory:
    """Factory for instantiating Chat LLMs and Embedding clients (NVIDIA & Voyage AI)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def get_chat_model(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatNVIDIA:
        """
        Instantiate and return a ChatNVIDIA instance.

        Args:
            model_name: Name of the NVIDIA chat model (e.g. meta/llama-3.1-70b-instruct).
            api_key: NVIDIA API Key.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Max completion tokens.

        Returns:
            Configured ChatNVIDIA instance.
        """
        model = model_name or self.settings.nvidia_chat_model
        key = api_key or self.settings.effective_chat_api_key
        temp = temperature if temperature is not None else self.settings.temperature
        tokens = max_tokens if max_tokens is not None else self.settings.max_tokens

        logger.info(f"Initializing ChatNVIDIA model: {model} (temp={temp})")
        return ChatNVIDIA(
            model=model,
            api_key=key,
            temperature=temp,
            max_tokens=tokens,
        )

    def get_voyage_embeddings(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> VoyageAIEmbeddings:
        """
        Instantiate and return a VoyageAIEmbeddings instance.

        Args:
            model_name: Name of the Voyage model (e.g. voyage-3, voyage-3-lite, voyage-code-3).
            api_key: Voyage AI API Key.

        Returns:
            Configured VoyageAIEmbeddings instance.
        """
        model = model_name or self.settings.voyage_embedding_model or "voyage-3"
        key = api_key or self.settings.voyage_api_key or self.settings.effective_embedding_api_key
        logger.info(f"Initializing VoyageAIEmbeddings model: {model}")
        return VoyageAIEmbeddings(
            model=model,
            voyage_api_key=key,
        )

    def get_nvidia_embeddings(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        truncate: str = "END",
    ) -> NVIDIAEmbeddings:
        """
        Instantiate and return an NVIDIAEmbeddings instance.

        Args:
            model_name: Name of the NVIDIA embedding model (e.g. nvidia/nv-embedqa-e5-v5).
            api_key: NVIDIA API Key.
            truncate: Truncation strategy ('END', 'START', 'NONE').

        Returns:
            Configured NVIDIAEmbeddings instance.
        """
        model = model_name or self.settings.nvidia_embedding_model
        key = api_key or self.settings.effective_embedding_api_key
        logger.info(f"Initializing NVIDIAEmbeddings model: {model} (truncate={truncate})")
        return NVIDIAEmbeddings(
            model=model,
            api_key=key,
            truncate=truncate,
        )

    def get_embedding_model(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        truncate: str = "END",
    ) -> Embeddings:
        """
        Instantiate and return an Embeddings instance (auto-routes to Voyage AI or NVIDIA).

        Args:
            model_name: Name of the embedding model (e.g. voyage-3 or nvidia/nv-embedqa-e5-v5).
            api_key: API Key for the respective provider.
            truncate: Truncation strategy for NVIDIA models.

        Returns:
            Configured Embeddings instance.
        """
        if model_name:
            is_voyage = "voyage" in model_name.lower() or self.settings.embedding_provider.lower() == "voyage"
            if is_voyage:
                return self.get_voyage_embeddings(model_name=model_name, api_key=api_key)
            return self.get_nvidia_embeddings(model_name=model_name, api_key=api_key, truncate=truncate)

        # When model_name is not provided, evaluate settings
        if self.settings.embedding_provider.lower() == "voyage":
            if "voyage" in self.settings.nvidia_embedding_model.lower():
                target = self.settings.nvidia_embedding_model
            else:
                target = self.settings.voyage_embedding_model or "voyage-3"
            return self.get_voyage_embeddings(model_name=target, api_key=api_key)

        if "voyage" in self.settings.nvidia_embedding_model.lower():
            return self.get_voyage_embeddings(model_name=self.settings.nvidia_embedding_model, api_key=api_key)

        return self.get_nvidia_embeddings(model_name=self.settings.nvidia_embedding_model, api_key=api_key, truncate=truncate)


# Alias for multi-provider clarity
ModelClientFactory = NVIDIAClientFactory
