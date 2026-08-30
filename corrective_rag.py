"""
Backward-compatible entrypoint for Corrective RAG Agent.
Delegates to the modular application architecture in `src/` and `ui/`.

Usage:
    streamlit run corrective_rag.py
"""

from ui.app import run_ui

if __name__ == "__main__":
    run_ui()