"""
Document splitting services with tiktoken and character-based fallback.
"""

from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger

logger = get_logger("DocumentSplitter")


class DocumentSplitter:
    """Service to split documents into semantically coherent chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks.

        Args:
            documents: List of input documents.

        Returns:
            List of split document chunks.
        """
        logger.info(
            f"Splitting {len(documents)} documents (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
        )
        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splits = splitter.split_documents(documents)
        except Exception as e:
            logger.warning(f"Tiktoken encoder failed ({str(e)}). Falling back to character splitter.")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            splits = splitter.split_documents(documents)

        logger.info(f"Split {len(documents)} source docs into {len(splits)} chunks")
        return splits
