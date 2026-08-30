"""
Shared pytest fixtures and mock objects.
"""

import pytest
from langchain_core.documents import Document
from src.config.settings import Settings


@pytest.fixture
def mock_settings():
    return Settings(
        nvidia_api_key="nvapi-mock-test-key",
        tavily_api_key="tvly-mock-test-key",
        qdrant_url="http://localhost:6333",
        qdrant_collection_name="test-collection",
        nvidia_chat_model="meta/llama-3.1-70b-instruct",
        nvidia_embedding_model="nvidia/nv-embedqa-e5-v5",
        chunk_size=500,
        chunk_overlap=100,
    )


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7B to 70B parameters.",
            metadata={"source": "https://arxiv.org/pdf/2307.09288.pdf", "page": 1}
        ),
        Document(
            page_content="Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases and outperform open-source chat models.",
            metadata={"source": "https://arxiv.org/pdf/2307.09288.pdf", "page": 2}
        )
    ]
