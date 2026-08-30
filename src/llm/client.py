"""
Factory service for NVIDIA LLM and Embedding models.
"""

from typing import Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger("NVIDIAClientFactory")


class NVIDIAClientFactory:
    """Factory for instantiating NVIDIA chat models and embedding clients."""

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
        key = api_key or self.settings.nvidia_api_key
        temp = temperature if temperature is not None else self.settings.temperature
        tokens = max_tokens if max_tokens is not None else self.settings.max_tokens

        logger.info(f"Initializing ChatNVIDIA model: {model} (temp={temp})")
        return ChatNVIDIA(
            model=model,
            api_key=key,
            temperature=temp,
            max_tokens=tokens,
        )

    def get_embedding_model(
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
        key = api_key or self.settings.nvidia_api_key

        logger.info(f"Initializing NVIDIAEmbeddings model: {model} (truncate={truncate})")
        return NVIDIAEmbeddings(
            model=model,
            api_key=key,
            truncate=truncate,
        )
