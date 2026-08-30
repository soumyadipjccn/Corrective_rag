"""
Streamlit Application UI for Corrective RAG (CRAG) Agent.
"""

import pprint
import streamlit as st
import nest_asyncio

from src.config.settings import get_settings
from src.service import CRAGService
from ui.components import render_sidebar, format_state_for_display

nest_asyncio.apply()


def initialize_app_state():
    """Initialize session state variables."""
    if "service" not in st.session_state:
        st.session_state.service = None
    if "ingested_source" not in st.session_state:
        st.session_state.ingested_source = None
    if "doc_url" not in st.session_state:
        st.session_state.doc_url = "https://arxiv.org/pdf/2307.09288.pdf"


def run_ui():
    """Main Streamlit execution loop."""
    st.set_page_config(
        page_title="Corrective RAG Agent",
        page_icon="🔄",
        layout="wide",
    )

    initialize_app_state()
    base_settings = get_settings()
    settings = render_sidebar(base_settings)

    # Instantiate or update service
    if st.session_state.service is None:
        st.session_state.service = CRAGService(settings=settings)
    else:
        st.session_state.service.settings = settings

    service: CRAGService = st.session_state.service

    # Main Interface
    st.title("🔄 Corrective RAG Agent")
    st.caption("Powered by LangGraph, NVIDIA NIM Models, and Qdrant Vector Search")

    if not settings.nvidia_api_key:
        st.info("👈 Enter your **NVIDIA API Key** in the sidebar to get started.")
        st.stop()

    # Document Ingestion Section
    st.subheader("📥 1. Document Ingestion")
    col1, col2 = st.columns([3, 1])

    with col1:
        input_method = st.radio(
            "Select Ingestion Source:",
            ["Document URL", "File Upload"],
            horizontal=True
        )

        source_key = None
        if input_method == "Document URL":
            url_input = st.text_input("Enter Document URL (PDF or Web Page):", value=st.session_state.doc_url)
            source_key = url_input
        else:
            uploaded_file = st.file_uploader("Upload a file (.pdf, .txt, .md):", type=["pdf", "txt", "md"])
            if uploaded_file:
                source_key = uploaded_file.name

    with col2:
        st.write("")
        st.write("")
        ingest_clicked = st.button("🚀 Ingest Document", use_container_width=True)

    # Ingest document if source changed or button clicked
    should_ingest = ingest_clicked or (source_key and st.session_state.ingested_source != source_key)

    if should_ingest and source_key:
        with st.spinner(f"Ingesting and indexing '{source_key}' into Qdrant..."):
            if input_method == "Document URL":
                result = service.ingest_url(source_key)
            else:
                result = service.ingest_bytes(uploaded_file.getvalue(), uploaded_file.name)

            if result.status == "success":
                st.session_state.ingested_source = source_key
                st.success(
                    f"✅ Successfully indexed **{result.doc_count} pages** into **{result.chunk_count} chunks** "
                    f"(Vector Dim: {result.vector_dim}) in collection `{result.collection_name}`"
                )
            else:
                st.error(f"❌ Ingestion failed: {result.error}")

    # Query Section
    st.divider()
    st.subheader("💬 2. Ask Questions")
    st.info("💡 **Example query:** *What are the experiment results and ablation studies in this research paper?*")

    user_question = st.text_input("Enter your question:", placeholder="Ask anything about the ingested document...")

    if user_question:
        st.subheader("⚙️ LangGraph Execution Flow")
        progress_container = st.container()

        final_generation = None
        last_state = {}

        with progress_container:
            for node_name, state_keys in service.stream_query(user_question):
                last_state = state_keys
                with st.expander(f"📍 Step executed: `{node_name}`", expanded=(node_name == "generate")):
                    formatted_dict = format_state_for_display(state_keys)
                    st.text(pprint.pformat(formatted_dict, indent=2, width=90))

                if "generation" in state_keys:
                    final_generation = state_keys["generation"]

        st.divider()
        st.subheader("🎯 Final Answer")
        if final_generation:
            st.markdown(final_generation)
        else:
            fallback = last_state.get("generation", "No generation produced.")
            st.write(fallback)


if __name__ == "__main__":
    run_ui()
