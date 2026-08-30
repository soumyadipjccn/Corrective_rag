"""
Unit tests for FastAPI Web & REST Server.
"""

import pytest
from fastapi.testclient import TestClient
from ui.server import app, serialize_document, serialize_graph_state
from langchain_core.documents import Document


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_get_and_update(client):
    # GET config
    get_res = client.get("/api/config")
    assert get_res.status_code == 200
    data = get_res.json()
    assert "available_chat_models" in data
    assert "chunk_size" in data

    # POST config update
    post_res = client.post("/api/config", json={
        "chunk_size": 600,
        "chunk_overlap": 150,
        "nvidia_chat_model": "meta/llama-3.1-8b-instruct"
    })
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "success"

    # Verify updated
    get_res_2 = client.get("/api/config")
    data_2 = get_res_2.json()
    assert data_2["chunk_size"] == 600
    assert data_2["chunk_overlap"] == 150
    assert data_2["nvidia_chat_model"] == "meta/llama-3.1-8b-instruct"


def test_serialize_helpers(sample_documents):
    # Document serialization
    doc = sample_documents[0]
    serialized_doc = serialize_document(doc)
    assert serialized_doc["source"] == "https://arxiv.org/pdf/2307.09288.pdf"
    assert "Llama 2" in serialized_doc["snippet"]

    # Graph state serialization
    state = {
        "question": "What is Llama 2?",
        "documents": sample_documents,
        "run_web_search": "No"
    }
    serialized_state = serialize_graph_state(state)
    assert serialized_state["question"] == "What is Llama 2?"
    assert len(serialized_state["documents"]) == 2
    assert serialized_state["documents"][0]["source"] == "https://arxiv.org/pdf/2307.09288.pdf"


def test_index_page_serving(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Corrective RAG Agent" in response.text
