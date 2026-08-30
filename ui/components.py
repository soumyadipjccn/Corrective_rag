"""
Reusable Streamlit UI components for the Corrective RAG application.
"""

import pprint
from typing import Any, Dict
import streamlit as st
from langchain_core.documents import Document
from src.config.settings import Settings, get_settings


def format_document(doc: Document) -> str:
    """Format single Document object for clean UI display."""
    source = doc.metadata.get("source", "Unknown")
    title = doc.metadata.get("title", "No title")
    preview = doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else "")
    return f"📄 [Source: {source} | Title: {title}]\n{preview}"


def format_state_for_display(state_keys: Dict[str, Any]) -> Dict[str, Any]:
    """Format graph execution state for readable expander inspection."""
    formatted = {}
    for k, v in state_keys.items():
        if k == "documents" and isinstance(v, list):
            formatted[k] = [format_document(d) if isinstance(d, Document) else str(d) for d in v]
        else:
            formatted[k] = v
    return formatted


def render_sidebar(settings: Settings) -> Settings:
    """
    Render Streamlit sidebar for API keys and configuration parameters.

    Returns:
        Updated Settings instance.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔑 API Credentials")
        nvidia_key = st.text_input(
            "NVIDIA API Key",
            value=settings.nvidia_api_key,
            type="password",
            help="NVIDIA NIM API key (nvapi-...)"
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            value=settings.tavily_api_key,
            type="password",
            help="Tavily API key for web search fallback"
        )

        st.subheader("🗄️ Vector Database")
        qdrant_url = st.text_input(
            "Qdrant URL",
            value=settings.qdrant_url,
            help="Endpoint for Qdrant vector database"
        )
        qdrant_key = st.text_input(
            "Qdrant API Key",
            value=settings.qdrant_api_key,
            type="password",
            help="Optional API key for authenticated Qdrant cluster"
        )

        st.subheader("🧠 NVIDIA Models")
        chat_idx = (
            settings.available_chat_models.index(settings.nvidia_chat_model)
            if settings.nvidia_chat_model in settings.available_chat_models
            else 0
        )
        selected_chat = st.selectbox(
            "Chat LLM",
            options=settings.available_chat_models,
            index=chat_idx,
        )

        embed_idx = (
            settings.available_embedding_models.index(settings.nvidia_embedding_model)
            if settings.nvidia_embedding_model in settings.available_embedding_models
            else 0
        )
        selected_embed = st.selectbox(
            "Embedding Model",
            options=settings.available_embedding_models,
            index=embed_idx,
        )

        st.subheader("✂️ Chunking")
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=settings.chunk_size, step=50)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=settings.chunk_overlap, step=25)

        updated_settings = Settings(
            nvidia_api_key=nvidia_key,
            tavily_api_key=tavily_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            nvidia_chat_model=selected_chat,
            nvidia_embedding_model=selected_embed,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not updated_settings.nvidia_api_key:
            st.warning("⚠️ Please provide an NVIDIA API Key to continue.")

        return updated_settings
