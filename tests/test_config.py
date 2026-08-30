"""
Unit tests for configuration settings.
"""

from src.config.settings import Settings, get_settings


def test_default_settings():
    settings = Settings()
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection_name == "rag-qdrant"
    assert "meta/llama-3.1-70b-instruct" in settings.available_chat_models
    assert "nvidia/nv-embedqa-e5-v5" in settings.available_embedding_models


def test_settings_override():
    settings = Settings(
        nvidia_api_key="custom-nv-key",
        qdrant_url="http://custom-qdrant:6333",
        chunk_size=1000,
    )
    assert settings.nvidia_api_key == "custom-nv-key"
    assert settings.qdrant_url == "http://custom-qdrant:6333"
    assert settings.chunk_size == 1000


def test_get_settings_caching():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
