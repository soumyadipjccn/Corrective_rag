"""
Diagnostic script to verify embedding connectivity (Voyage AI or NVIDIA).
"""
from src.config.settings import get_settings
from src.llm.client import NVIDIAClientFactory


def test_embeddings():
    settings = get_settings()
    if settings.is_fastembed_embedding:
        provider = "FastEmbed (Local Offline)"
        model = settings.fastembed_model if "bge" in settings.nvidia_embedding_model.lower() else settings.nvidia_embedding_model
    elif settings.is_voyage_embedding:
        provider = "Voyage AI"
        model = settings.voyage_embedding_model if (settings.embedding_provider.lower() == "voyage" and "voyage" not in settings.nvidia_embedding_model.lower()) else settings.nvidia_embedding_model
    else:
        provider = "NVIDIA NIM"
        model = settings.nvidia_embedding_model

    print(f"Testing Embedding Provider: {provider}")
    print(f"Active Model ID: {model}")
    print(f"Effective API Key configured: {'Yes (Local - no key needed)' if settings.is_fastembed_embedding else ('Yes' if settings.effective_embedding_api_key else 'No')}")

    try:
        emb = NVIDIAClientFactory(settings).get_embedding_model()
        vec = emb.embed_query("Test embedding connection")
        print(f"✅ Success! Generated embedding with dimension: {len(vec)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        if settings.is_voyage_embedding and not settings.voyage_api_key:
            print("💡 Tip: Set VOYAGE_API_KEY=pa-your-key in .env or provide it in the Web UI / CLI.")
        elif not settings.is_fastembed_embedding and not settings.is_voyage_embedding:
            print("💡 Tip: Set NVIDIA_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b in .env, or use EMBEDDING_PROVIDER=fastembed for local embeddings.")


if __name__ == "__main__":
    test_embeddings()