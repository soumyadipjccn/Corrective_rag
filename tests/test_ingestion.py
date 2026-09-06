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


from unittest.mock import patch

def test_loader_service_arxiv_url_detection():
    from src.ingestion.loaders import DocumentLoaderService
    
    with patch("src.ingestion.loaders.PyPDFLoader") as mock_pypdf:
        mock_pypdf.return_value.load.return_value = [Document(page_content="ArXiv content", metadata={})]

        docs = DocumentLoaderService.load_from_url("https://arxiv.org/pdf/2401.15884")
        mock_pypdf.assert_called_once_with("https://arxiv.org/pdf/2401.15884")
        assert len(docs) == 1
        assert docs[0].page_content == "ArXiv content"


