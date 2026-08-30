"""
Unit tests for document splitting and ingestion logic.
"""

from langchain_core.documents import Document
from src.ingestion.splitters import DocumentSplitter


def test_document_splitter(sample_documents):
    splitter = DocumentSplitter(chunk_size=50, chunk_overlap=10)
    splits = splitter.split_documents(sample_documents)

    assert len(splits) >= len(sample_documents)
    for s in splits:
        assert isinstance(s, Document)
        assert len(s.page_content) > 0
