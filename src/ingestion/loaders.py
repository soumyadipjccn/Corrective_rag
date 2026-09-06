"""
Document loading services supporting PDF URLs, Web pages, and local files.
"""

import os
import tempfile
from typing import List
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from src.utils.logger import get_logger
from src.utils.exceptions import IngestionError

logger = get_logger("DocumentLoaderService")


class DocumentLoaderService:
    """Service to load documents from various web sources and local files."""

    @staticmethod
    def load_from_url(url: str) -> List[Document]:
        """
        Load document from a web URL (auto-detects PDF vs HTML).

        Args:
            url: URL to web document or remote PDF.

        Returns:
            List of langchain Document objects.
        """
        logger.info(f"Loading document from URL: {url}")
        try:
            parsed_path = urlparse(url).path.lower()
            if parsed_path.endswith(".pdf") or "/pdf/" in parsed_path or "arxiv.org/pdf/" in url.lower():
                logger.info("URL identified as remote PDF. Using PyPDFLoader.")
                loader = PyPDFLoader(url)
            else:
                logger.info("URL identified as web page. Using WebBaseLoader.")
                loader = WebBaseLoader(url)
                loader.requests_per_second = 1

            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} document pages from {url}")
            return docs

        except Exception as e:
            msg = f"Failed to load document from URL '{url}': {str(e)}"
            logger.error(msg)
            raise IngestionError(msg) from e

    @staticmethod
    def load_from_file_path(file_path: str) -> List[Document]:
        """
        Load document from a local file path.

        Args:
            file_path: Local filesystem path.

        Returns:
            List of langchain Document objects.
        """
        logger.info(f"Loading document from file: {file_path}")
        if not os.path.exists(file_path):
            raise IngestionError(f"File does not exist: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".txt", ".md"]:
                loader = TextLoader(file_path)
            else:
                raise IngestionError(f"Unsupported file format: {ext}. Expected .pdf, .txt, or .md")

            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} document pages from {file_path}")
            return docs

        except Exception as e:
            msg = f"Failed to load file '{file_path}': {str(e)}"
            logger.error(msg)
            raise IngestionError(msg) from e

    @classmethod
    def load_from_bytes(cls, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load document from in-memory bytes (e.g. Streamlit file uploader).

        Args:
            file_bytes: Raw binary content of the file.
            filename: Original filename to extract extension.

        Returns:
            List of langchain Document objects.
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".txt", ".md"]:
            raise IngestionError(f"Unsupported upload type '{ext}'. Allowed: .pdf, .txt, .md")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(file_bytes)

        try:
            return cls.load_from_file_path(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
