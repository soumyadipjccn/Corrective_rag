"""
Unit tests for LangGraph state machine, nodes, and routing edges.
"""

from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.schemas.state import GraphState
from src.graph.edges import decide_to_generate
from src.graph.nodes import CRAGNodes
from src.graph.workflow import CRAGWorkflowBuilder


def test_decide_to_generate_routing():
    # When search is required
    state_search: GraphState = {"keys": {"run_web_search": "Yes"}}
    assert decide_to_generate(state_search) == "transform_query"

    # When search is not required
    state_generate: GraphState = {"keys": {"run_web_search": "No"}}
    assert decide_to_generate(state_generate) == "generate"

    # Default fallback
    state_empty: GraphState = {"keys": {}}
    assert decide_to_generate(state_empty) == "generate"


def test_nodes_retrieve_with_mock():
    mock_retriever = MagicMock()
    mock_doc = Document(page_content="Mock content", metadata={"source": "test"})
    mock_retriever.get_relevant_documents.return_value = [mock_doc]

    nodes = CRAGNodes(retriever=mock_retriever)
    input_state: GraphState = {"keys": {"question": "What is AI?"}}
    output = nodes.retrieve(input_state)

    assert "documents" in output["keys"]
    assert len(output["keys"]["documents"]) == 1
    assert output["keys"]["documents"][0].page_content == "Mock content"
    mock_retriever.get_relevant_documents.assert_called_once_with("What is AI?")


def test_workflow_compilation():
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    mock_search = MagicMock()

    builder = CRAGWorkflowBuilder(
        retriever=mock_retriever,
        llm=mock_llm,
        search_service=mock_search
    )
    app = builder.build()
    assert app is not None
