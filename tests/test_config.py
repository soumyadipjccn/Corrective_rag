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
    # Assert deprecated model is removed
    assert "nvidia/llama-3.2-nv-embedqa-1b-v2" not in settings.available_embedding_models


def test_settings_override():
    settings = Settings(
        _env_file=None,
        nvidia_api_key="custom-nv-key",
        qdrant_url="http://custom-qdrant:6333",
        chunk_size=1000,
        nvidia_chat_model="custom/my-special-llm-model",
        nvidia_embedding_model="custom/my-special-embedding-model",
    )
    assert settings.nvidia_api_key == "custom-nv-key"
    assert settings.effective_chat_api_key == "custom-nv-key"
    assert settings.effective_embedding_api_key == "custom-nv-key"
    assert settings.qdrant_url == "http://custom-qdrant:6333"
    assert settings.chunk_size == 1000
    assert settings.nvidia_chat_model == "custom/my-special-llm-model"
    assert settings.nvidia_embedding_model == "custom/my-special-embedding-model"


def test_distinct_api_keys():
    settings = Settings(
        _env_file=None,
        nvidia_api_key="shared-key",
        nvidia_chat_api_key="distinct-chat-key",
        nvidia_embedding_api_key="distinct-embed-key",
    )
    assert settings.effective_chat_api_key == "distinct-chat-key"
    assert settings.effective_embedding_api_key == "distinct-embed-key"


def test_voyage_settings_detection():
    # Test auto-detection via model name
    s1 = Settings(_env_file=None, nvidia_embedding_model="voyage-3", voyage_api_key="pa-test-key")
    assert s1.is_voyage_embedding is True
    assert s1.effective_embedding_api_key == "pa-test-key"

    # Test explicit provider selection
    s2 = Settings(_env_file=None, embedding_provider="voyage", voyage_api_key="pa-test-key-2")
    assert s2.is_voyage_embedding is True
    assert s2.effective_embedding_api_key == "pa-test-key-2"

    # Test fallback to nvidia_embedding_api_key if voyage_api_key not provided
    s3 = Settings(_env_file=None, nvidia_embedding_model="voyage-3", nvidia_embedding_api_key="pa-fallback-key")
    assert s3.is_voyage_embedding is True
    assert s3.effective_embedding_api_key == "pa-fallback-key"


def test_voyage_factory_instantiation():
    from src.llm.client import NVIDIAClientFactory
    from langchain_voyageai import VoyageAIEmbeddings

    settings = Settings(_env_file=None, nvidia_embedding_model="voyage-3-lite", voyage_api_key="pa-mock-key")
    factory = NVIDIAClientFactory(settings)
    emb = factory.get_embedding_model()
    assert isinstance(emb, VoyageAIEmbeddings)
    assert emb.model == "voyage-3-lite"


def test_fastembed_settings_and_factory():
    from src.llm.client import NVIDIAClientFactory
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    # Auto-detection via model name
    s1 = Settings(_env_file=None, nvidia_embedding_model="BAAI/bge-small-en-v1.5")
    assert s1.is_fastembed_embedding is True
    assert s1.effective_embedding_api_key == "local-fastembed"

    # Explicit provider selection
    s2 = Settings(_env_file=None, embedding_provider="fastembed")
    assert s2.is_fastembed_embedding is True
    assert s2.effective_embedding_api_key == "local-fastembed"

    # Factory instantiation
    factory = NVIDIAClientFactory(s1)
    emb = factory.get_embedding_model()
    assert isinstance(emb, FastEmbedEmbeddings)
    assert emb.model_name == "BAAI/bge-small-en-v1.5"


def test_get_settings_caching():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2



