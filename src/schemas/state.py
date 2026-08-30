"""
State schemas and data models for Corrective RAG workflow.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Represents the state of the Corrective RAG graph.

    Attributes:
        keys: Dictionary storing all graph state keys (question, documents, generation, etc.)
    """
    keys: Dict[str, Any]


class DocumentGrade(BaseModel):
    """Grading decision for document relevance."""
    score: Literal["yes", "no"] = Field(
        description="Relevance binary score indicating whether the document is relevant to the question"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Optional justification for the grading score"
    )


class FormattedDocument(BaseModel):
    """Structured representation of a document for UI / logging display."""
    source: str = Field(default="Unknown", description="Source URL or filename")
    title: str = Field(default="No title", description="Title or header")
    snippet: str = Field(default="", description="Snippet of page content")


class IngestionResult(BaseModel):
    """Result summary of a document ingestion job."""
    source: str = Field(description="Source identifier (file path or URL)")
    doc_count: int = Field(default=0, description="Total loaded raw documents")
    chunk_count: int = Field(default=0, description="Total chunks produced")
    collection_name: str = Field(description="Target vector store collection name")
    vector_dim: int = Field(description="Dimension of the indexed vectors")
    status: str = Field(default="success", description="Ingestion status ('success' or 'failed')")
    error: Optional[str] = Field(default=None, description="Error message if failed")
