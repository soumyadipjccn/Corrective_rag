"""
Domain-specific exception hierarchy for Corrective RAG.
"""


class CRAGException(Exception):
    """Base exception for all CRAG operations."""
    pass


class IngestionError(CRAGException):
    """Raised when document loading or parsing fails."""
    pass


class VectorStoreError(CRAGException):
    """Raised when Qdrant vector store operations fail."""
    pass


class SearchError(CRAGException):
    """Raised when web search execution fails."""
    pass


class LLMError(CRAGException):
    """Raised when LLM invocation or parsing fails."""
    pass


class GraphExecutionError(CRAGException):
    """Raised when LangGraph workflow execution encounters a critical error."""
    pass
