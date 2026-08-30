"""
Unified high-level Service Facade for Corrective RAG Agent.
Handles document ingestion, vectorstore lifecycle, model management, and workflow streaming.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config.settings import Settings, get_settings
from src.schemas.state import IngestionResult
from src.llm.client import NVIDIAClientFactory
from src.ingestion.loaders import DocumentLoaderService
from src.ingestion.splitters import DocumentSplitter
from src.vectorstore.qdrant_store import QdrantVectorStoreManager
from src.tools.search import TavilySearchService
from src.graph.workflow import CRAGWorkflowBuilder
from src.utils.logger import get_logger
from src.utils.exceptions import CRAGException

logger = get_logger("CRAGService")


class CRAGService:
    """Enterprise-grade service facade coordinating all CRAG modules."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        chat_model_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.settings = settings or get_settings()

        # Overrides if provided
        if nvidia_api_key:
            self.settings.nvidia_api_key = nvidia_api_key
        if tavily_api_key:
            self.settings.tavily_api_key = tavily_api_key
        if qdrant_url:
            self.settings.qdrant_url = qdrant_url
        if qdrant_api_key:
            self.settings.qdrant_api_key = qdrant_api_key
        if chat_model_name:
            self.settings.nvidia_chat_model = chat_model_name
        if embedding_model_name:
            self.settings.nvidia_embedding_model = embedding_model_name

        self.client_factory = NVIDIAClientFactory(self.settings)
        self.loader_service = DocumentLoaderService()
        self.splitter = DocumentSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.search_service = TavilySearchService(
            api_key=self.settings.tavily_api_key,
            settings=self.settings
        )

        # Lazy initialized components
        self._embeddings = None
        self._llm = None
        self._vector_manager: Optional[QdrantVectorStoreManager] = None
        self._retriever: Optional[VectorStoreRetriever] = None
        self._app = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self.client_factory.get_embedding_model()
        return self._embeddings

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self.client_factory.get_chat_model()
        return self._llm

    @property
    def vector_manager(self) -> QdrantVectorStoreManager:
        if self._vector_manager is None:
            self._vector_manager = QdrantVectorStoreManager(
                embeddings=self.embeddings,
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                collection_name=self.settings.qdrant_collection_name,
                settings=self.settings,
            )
        return self._vector_manager

    def set_retriever(self, retriever: VectorStoreRetriever):
        """Set or update the active retriever and rebuild workflow."""
        self._retriever = retriever
        self._rebuild_app()

    def _rebuild_app(self):
        """Recompile the LangGraph application with updated components."""
        builder = CRAGWorkflowBuilder(
            retriever=self._retriever,
            llm=self.llm,
            search_service=self.search_service,
        )
        self._app = builder.build()

    @property
    def app(self):
        if self._app is None:
            self._rebuild_app()
        return self._app

    def ingest_url(self, url: str) -> IngestionResult:
        """Load and index documents from a remote URL."""
        logger.info(f"Initiating URL ingestion for: {url}")
        try:
            docs = self.loader_service.load_from_url(url)
            return self._process_and_index(docs, source=url)
        except Exception as e:
            return IngestionResult(
                source=url,
                collection_name=self.settings.qdrant_collection_name,
                vector_dim=0,
                status="failed",
                error=str(e)
            )

    def ingest_file(self, file_path: str) -> IngestionResult:
        """Load and index documents from a local filesystem path."""
        logger.info(f"Initiating file ingestion for: {file_path}")
        try:
            docs = self.loader_service.load_from_file_path(file_path)
            return self._process_and_index(docs, source=file_path)
        except Exception as e:
            return IngestionResult(
                source=file_path,
                collection_name=self.settings.qdrant_collection_name,
                vector_dim=0,
                status="failed",
                error=str(e)
            )

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> IngestionResult:
        """Load and index documents from binary memory content."""
        logger.info(f"Initiating byte upload ingestion for: {filename}")
        try:
            docs = self.loader_service.load_from_bytes(file_bytes, filename)
            return self._process_and_index(docs, source=filename)
        except Exception as e:
            return IngestionResult(
                source=filename,
                collection_name=self.settings.qdrant_collection_name,
                vector_dim=0,
                status="failed",
                error=str(e)
            )

    def _process_and_index(self, docs: List[Document], source: str) -> IngestionResult:
        """Split and index documents into Qdrant."""
        splits = self.splitter.split_documents(docs)
        dim = self.vector_manager.get_embedding_dimension()
        retriever = self.vector_manager.index_documents(splits, recreate_collection=True)
        self.set_retriever(retriever)

        return IngestionResult(
            source=source,
            doc_count=len(docs),
            chunk_count=len(splits),
            collection_name=self.settings.qdrant_collection_name,
            vector_dim=dim,
            status="success"
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Execute full CRAG pipeline synchronously."""
        inputs = {"keys": {"question": question}}
        result = self.app.invoke(inputs)
        return result.get("keys", {})

    def stream_query(self, question: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream CRAG execution step by step.

        Yields:
            (node_name, state_dict) tuples.
        """
        inputs = {"keys": {"question": question}}
        for output in self.app.stream(inputs):
            for node_name, state_payload in output.items():
                yield node_name, state_payload.get("keys", {})
