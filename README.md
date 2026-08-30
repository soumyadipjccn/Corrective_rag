# 🔄 Corrective RAG (CRAG) Agent - Industry-Grade Architecture

An enterprise-grade, modular **Corrective Retrieval-Augmented Generation (CRAG)** agent built with **LangGraph**, **NVIDIA NIM Models**, **NVIDIA Embeddings**, and **Qdrant Vector Database**.

---

## 🏛️ System Architecture

Corrective RAG enhances standard RAG workflows by evaluating the relevance of retrieved documents before generating responses. If the retrieved context is insufficient or off-topic, the system triggers a query transformation and falls back to live web search via Tavily.

```mermaid
graph TD
    Start([User Question]) --> Retrieve[Retrieve Documents from Qdrant]
    Retrieve --> Grade[Grade Documents Relevance]
    Grade --> Decision{Relevant Context Found?}
    
    Decision -- Yes --> Generate[Generate Answer with NVIDIA LLM]
    Decision -- No / Incomplete --> Transform[Transform & Optimize Query]
    
    Transform --> WebSearch[Execute Web Search via Tavily]
    WebSearch --> Generate
    Generate --> End([Final Answer with Grounded Context])
```

---

## 📂 Project Structure

```text
Corrective_rag/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Centralized Pydantic Settings & environment variables
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── state.py             # TypedDict GraphState, DocumentGrade & Ingestion models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # Standardized application logging
│   │   └── exceptions.py        # Domain-specific exception hierarchy
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py            # Factory for ChatNVIDIA & NVIDIAEmbeddings
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── crag_prompts.py      # Grading, Query Optimization, and Generation prompts
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Document loaders (PDF URL, Web HTML, Local Files)
│   │   └── splitters.py         # Recursive token-aware text splitters
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── qdrant_store.py      # Qdrant client, dynamic vector dimensions & indexing
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search.py            # Tavily search tool with retry backoff & formatting
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── nodes.py             # Retrieve, Grade, TransformQuery, WebSearch, Generate nodes
│   │   ├── edges.py             # Conditional routing decisions
│   │   └── workflow.py          # StateGraph compilation & execution graph
│   └── service.py               # Unified CRAGService facade
├── ui/
│   ├── __init__.py
│   ├── components.py            # Streamlit UI widgets, sidebar & state formatters
│   └── app.py                   # Streamlit web interface application
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures & sample documents
│   ├── test_config.py           # Configuration unit tests
│   ├── test_schemas.py          # Pydantic schema validation tests
│   ├── test_graph.py            # StateGraph routing & node execution tests
│   └── test_ingestion.py        # Chunking & ingestion tests
├── app.py                       # Main Streamlit web application launcher
├── main.py                      # CLI tool for terminal & automated queries
├── corrective_rag.py            # Backward-compatible entrypoint
├── .env.example                 # Configuration environment template
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## ⚡ Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example configuration and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
NVIDIA_API_KEY=nvapi-your-key-here
TAVILY_API_KEY=tvly-your-key-here
QDRANT_URL=http://localhost:6333
NVIDIA_CHAT_MODEL=meta/llama-3.1-70b-instruct
NVIDIA_EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
```

---

## 🚀 Running the Application

### Option A: Streamlit Interactive Web UI

```bash
streamlit run app.py
```

### Option B: CLI Terminal Interface

```bash
# Ingest a document and query directly
python main.py --url "https://arxiv.org/pdf/2307.09288.pdf" --query "What are the fine-tuning methods used in Llama 2?"
```

---

## 🧪 Running Unit Tests

Run the test suite with pytest:

```bash
pytest tests/ -v
```
