"""
Diagnostic script to verify embedding connectivity (Voyage AI or NVIDIA).
"""
from src.config.settings import get_settings
from src.llm.client import NVIDIAClientFactory


def test_embeddings():
    settings = get_settings()
    provider = "Voyage AI" if settings.is_voyage_embedding else "NVIDIA NIM"
    model = settings.voyage_embedding_model if (settings.embedding_provider.lower() == "voyage" and "voyage" not in settings.nvidia_embedding_model.lower()) else settings.nvidia_embedding_model

    print(f"Testing Embedding Provider: {provider}")
    print(f"Active Model ID: {model}")
    print(f"Effective API Key configured: {'Yes' if settings.effective_embedding_api_key else 'No'}")

    try:
        emb = NVIDIAClientFactory(settings).get_embedding_model()
        vec = emb.embed_query("Test embedding connection")
        print(f"✅ Success! Generated embedding with dimension: {len(vec)}")
    except Exception as e:
        print(f"❌ Error: {e}")
        if settings.is_voyage_embedding and not settings.voyage_api_key:
            print("💡 Tip: Set VOYAGE_API_KEY=pa-your-key in .env or provide it in the Web UI / CLI.")
        elif not settings.is_voyage_embedding:
            print("💡 Tip: To use Voyage AI, set NVIDIA_EMBEDDING_MODEL=voyage-3 and VOYAGE_API_KEY=pa-... in .env")


if __name__ == "__main__":
    test_embeddings()