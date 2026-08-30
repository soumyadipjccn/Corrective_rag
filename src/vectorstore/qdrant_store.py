"""
Qdrant Vector Store Manager with dynamic dimension detection and collection lifecycle management.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger
from src.utils.exceptions import VectorStoreError

logger = get_logger("QdrantVectorStoreManager")


class QdrantVectorStoreManager:
    """Manager for Qdrant client, collection creation, indexing, and retriever retrieval."""

    def __init__(
        self,
        embeddings: Embeddings,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.embeddings = embeddings
        self.url = url or self.settings.qdrant_url
        self.api_key = api_key or self.settings.qdrant_api_key
        self.collection_name = collection_name or self.settings.qdrant_collection_name
        self._client: Optional[QdrantClient] = None
        self._vectorstore: Optional[Qdrant] = None

    @property
    def client(self) -> QdrantClient:
        """Lazily initialize and return QdrantClient."""
        if self._client is None:
            logger.info(f"Connecting to Qdrant at {self.url}")
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key if self.api_key else None,
            )
        return self._client

    def get_embedding_dimension(self) -> int:
        """
        Dynamically calculate embedding vector dimension by running a test embedding.

        Returns:
            Vector dimension integer.
        """
        try:
            sample_vec = self.embeddings.embed_query("dimension probe query")
            dim = len(sample_vec)
            logger.info(f"Detected embedding vector dimension: {dim}")
            return dim
        except Exception as e:
            msg = f"Failed to compute embedding dimension: {str(e)}"
            logger.error(msg)
            raise VectorStoreError(msg) from e

    def create_or_recreate_collection(self, vector_dim: Optional[int] = None) -> int:
        """
        Recreate collection with proper vector dimension and cosine metric.

        Args:
            vector_dim: Explicit vector size. If None, auto-detected from embedding model.

        Returns:
            Vector dimension used.
        """
        dim = vector_dim or self.get_embedding_dimension()
        logger.info(f"Recreating Qdrant collection '{self.collection_name}' (size={dim}, distance=Cosine)")

        try:
            # Delete existing collection if present
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted old collection '{self.collection_name}'")
            except Exception:
                pass

            # Create fresh collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created fresh collection '{self.collection_name}'")
            return dim

        except Exception as e:
            msg = f"Failed to create Qdrant collection '{self.collection_name}': {str(e)}"
            logger.error(msg)
            raise VectorStoreError(msg) from e

    def index_documents(
        self,
        splits: List[Document],
        recreate_collection: bool = True
    ) -> VectorStoreRetriever:
        """
        Index document splits into Qdrant collection and return a retriever.

        Args:
            splits: List of document chunks.
            recreate_collection: Whether to recreate collection before inserting.

        Returns:
            VectorStoreRetriever instance.
        """
        if not splits:
            raise VectorStoreError("Cannot index empty document list")

        try:
            if recreate_collection:
                self.create_or_recreate_collection()

            logger.info(f"Indexing {len(splits)} chunks into '{self.collection_name}'")
            self._vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )
            self._vectorstore.add_documents(splits)
            logger.info(f"Successfully indexed {len(splits)} documents into Qdrant")

            return self.as_retriever()

        except Exception as e:
            msg = f"Failed to index documents into Qdrant: {str(e)}"
            logger.error(msg)
            raise VectorStoreError(msg) from e

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> VectorStoreRetriever:
        """
        Get retriever for the current collection.

        Args:
            search_kwargs: Optional search parameters (e.g. {'k': 4}).

        Returns:
            VectorStoreRetriever.
        """
        if self._vectorstore is None:
            self._vectorstore = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )

        kwargs = search_kwargs or {"k": 4}
        return self._vectorstore.as_retriever(search_kwargs=kwargs)
