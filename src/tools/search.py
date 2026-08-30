"""
Tavily Search Tool integration with retry logic and document formatting.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.tools import TavilySearchResults
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger("TavilySearchService")


class TavilySearchService:
    """Service to perform web searches using Tavily API and return structured Documents."""

    def __init__(self, api_key: Optional[str] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.tavily_api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _execute_search(self, tool: TavilySearchResults, query: str):
        """Execute Tavily query with exponential retry backoff."""
        return tool.invoke({"query": query})

    def search(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """
        Execute web search and convert results to LangChain Document format.

        Args:
            query: Search query string.
            max_results: Max results to return.

        Returns:
            List of Document objects created from web results.
        """
        if not self.api_key:
            logger.warning("Tavily API key is not configured. Skipping web search.")
            return []

        limit = max_results or self.settings.max_search_results
        logger.info(f"Executing Tavily web search for query: '{query}' (limit={limit})")

        try:
            tool = TavilySearchResults(
                api_key=self.api_key,
                max_results=limit,
                search_depth="advanced"
            )

            raw_results = self._execute_search(tool, query)
            if not raw_results:
                logger.info("No search results returned by Tavily.")
                return []

            documents = []
            formatted_chunks = []
            for item in raw_results:
                title = item.get("title", "No Title")
                content = item.get("content", "")
                url = item.get("url", "")
                chunk = f"Title: {title}\nURL: {url}\nContent: {content}\n"
                formatted_chunks.append(chunk)

            if formatted_chunks:
                doc = Document(
                    page_content="\n\n".join(formatted_chunks),
                    metadata={
                        "source": "tavily_search",
                        "query": query,
                        "result_count": len(formatted_chunks)
                    }
                )
                documents.append(doc)

            logger.info(f"Successfully generated {len(documents)} document container with search results")
            return documents

        except Exception as e:
            logger.error(f"Error during Tavily search execution: {str(e)}")
            return []
