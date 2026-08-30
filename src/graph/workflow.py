"""
LangGraph state machine builder for the Corrective RAG pipeline.
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.schemas.state import GraphState
from src.graph.nodes import CRAGNodes
from src.graph.edges import decide_to_generate
from src.tools.search import TavilySearchService
from src.utils.logger import get_logger

logger = get_logger("CRAGWorkflowBuilder")


class CRAGWorkflowBuilder:
    """Builder class for constructing and compiling the Corrective RAG LangGraph workflow."""

    def __init__(
        self,
        retriever: Optional[VectorStoreRetriever] = None,
        llm: Optional[BaseChatModel] = None,
        search_service: Optional[TavilySearchService] = None,
    ):
        self.nodes = CRAGNodes(
            retriever=retriever,
            llm=llm,
            search_service=search_service,
        )

    def build(self) -> CompiledStateGraph:
        """
        Construct and compile the state graph.

        Graph topology:
            START -> retrieve -> grade_documents -> [conditional]
                -> if Yes (search needed) -> transform_query -> web_search -> generate -> END
                -> if No (docs sufficient) -> generate -> END

        Returns:
            CompiledStateGraph instance ready for execution.
        """
        logger.info("Building Corrective RAG LangGraph state machine")
        workflow = StateGraph(GraphState)

        # Register nodes
        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("transform_query", self.nodes.transform_query)
        workflow.add_node("web_search", self.nodes.web_search)
        workflow.add_node("generate", self.nodes.generate)

        # Build workflow edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional branch after document grading
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        # Corrective fallback path
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        compiled_app = workflow.compile()
        logger.info("Compiled CRAG LangGraph application successfully")
        return compiled_app
