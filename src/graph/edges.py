"""
Graph routing edges and conditional branching logic.
"""

from typing import Literal
from src.schemas.state import GraphState
from src.utils.logger import get_logger

logger = get_logger("GraphEdges")


def decide_to_generate(state: GraphState) -> Literal["transform_query", "generate"]:
    """
    Evaluate grading outcome and route to either query transformation + web search or generation.

    Args:
        state: Current graph state containing 'keys'.

    Returns:
        Next node identifier ('transform_query' or 'generate').
    """
    state_dict = state.get("keys", {})
    search_required = state_dict.get("run_web_search", "No")

    logger.info(f"Evaluating routing condition: run_web_search='{search_required}'")

    if search_required == "Yes":
        logger.info("Decision -> Route to 'transform_query' for web search fallback")
        return "transform_query"
    else:
        logger.info("Decision -> Route directly to 'generate'")
        return "generate"
