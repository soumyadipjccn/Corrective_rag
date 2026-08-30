"""
LangGraph execution nodes for the Corrective RAG pipeline.
"""

import json
import re
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from src.schemas.state import GraphState
from src.prompts.crag_prompts import (
    GRADE_DOCUMENTS_PROMPT,
    TRANSFORM_QUERY_PROMPT,
    GENERATE_PROMPT,
)
from src.tools.search import TavilySearchService
from src.utils.logger import get_logger

logger = get_logger("CRAGNodes")


class CRAGNodes:
    """Encapsulates all node actions for the Corrective RAG workflow."""

    def __init__(
        self,
        retriever: Optional[VectorStoreRetriever] = None,
        llm: Optional[BaseChatModel] = None,
        search_service: Optional[TavilySearchService] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.search_service = search_service or TavilySearchService()

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector store based on user question.

        Args:
            state: Current GraphState.

        Returns:
            Updated state dict with retrieved documents.
        """
        logger.info("--- NODE: RETRIEVE ---")
        state_dict = state["keys"]
        question = state_dict["question"]

        if self.retriever is None:
            logger.warning("No retriever initialized. Returning empty document list.")
            return {"keys": {"documents": [], "question": question}}

        try:
            documents = self.retriever.get_relevant_documents(question)
            logger.info(f"Retrieved {len(documents)} documents from vector store")
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            documents = []

        return {"keys": {"documents": documents, "question": question}}

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Grade retrieved documents for relevance to the question.
        If any document is irrelevant or no relevant docs exist, flag for web search.

        Args:
            state: Current GraphState.

        Returns:
            Updated state dict with filtered documents and 'run_web_search' flag.
        """
        logger.info("--- NODE: GRADE DOCUMENTS ---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict.get("documents", [])

        if not self.llm:
            logger.warning("No LLM configured for grading. Keeping all documents.")
            return {"keys": {"documents": documents, "question": question, "run_web_search": "No"}}

        chain = GRADE_DOCUMENTS_PROMPT | self.llm | StrOutputParser()
        filtered_docs: List[Document] = []
        search = "No"

        if not documents:
            logger.info("No documents retrieved, setting web search to Yes")
            search = "Yes"

        for doc in documents:
            try:
                raw_response = chain.invoke({"question": question, "context": doc.page_content})
                
                # Extract JSON from potential markdown formatting
                json_match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
                clean_json = json_match.group(0) if json_match else raw_response.strip()
                
                score_data = json.loads(clean_json)
                grade = score_data.get("score", "yes").lower().strip()

                if grade == "yes":
                    logger.info("Grade: Document is RELEVANT")
                    filtered_docs.append(doc)
                else:
                    logger.info("Grade: Document is NOT RELEVANT -> Web search required")
                    search = "Yes"

            except Exception as e:
                logger.warning(f"Grading parser error ({str(e)}). Preserving document as fallback.")
                filtered_docs.append(doc)

        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "run_web_search": search,
            }
        }

    def transform_query(self, state: GraphState) -> Dict[str, Any]:
        """
        Rewrite question into a search-engine optimized query.

        Args:
            state: Current GraphState.

        Returns:
            Updated state dict with transformed query.
        """
        logger.info("--- NODE: TRANSFORM QUERY ---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict.get("documents", [])

        if not self.llm:
            logger.warning("No LLM configured for query transform. Using original question.")
            return {"keys": {"documents": documents, "question": question}}

        try:
            chain = TRANSFORM_QUERY_PROMPT | self.llm | StrOutputParser()
            better_question = chain.invoke({"question": question}).strip()
            logger.info(f"Transformed query: '{question}' -> '{better_question}'")
        except Exception as e:
            logger.error(f"Error transforming query: {str(e)}. Using original question.")
            better_question = question

        return {"keys": {"documents": documents, "question": better_question, "original_question": question}}

    def web_search(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute web search for the transformed question and append results.

        Args:
            state: Current GraphState.

        Returns:
            Updated state dict with web search documents appended.
        """
        logger.info("--- NODE: WEB SEARCH ---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = list(state_dict.get("documents", []))
        original_question = state_dict.get("original_question", question)

        try:
            search_docs = self.search_service.search(query=question)
            documents.extend(search_docs)
            logger.info(f"Appended {len(search_docs)} search result documents")
        except Exception as e:
            logger.error(f"Web search execution error: {str(e)}")

        # Restore original question for the final generation step
        return {"keys": {"documents": documents, "question": original_question}}

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate final grounded answer using LLM over retrieved + web documents.

        Args:
            state: Current GraphState.

        Returns:
            Updated state dict with final generation.
        """
        logger.info("--- NODE: GENERATE ---")
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict.get("documents", [])

        if not self.llm:
            return {
                "keys": {
                    "documents": documents,
                    "question": question,
                    "generation": "LLM model is not configured."
                }
            }

        try:
            context = "\n\n".join(doc.page_content for doc in documents) if documents else "No context available."
            chain = (
                {"context": lambda x: context, "question": lambda x: question}
                | GENERATE_PROMPT
                | self.llm
                | StrOutputParser()
            )

            generation = chain.invoke({})
            logger.info("Successfully generated response")

            return {
                "keys": {
                    "documents": documents,
                    "question": question,
                    "generation": generation,
                }
            }

        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            logger.error(error_msg)
            return {
                "keys": {
                    "documents": documents,
                    "question": question,
                    "generation": f"Error generating answer: {str(e)}"
                }
            }
