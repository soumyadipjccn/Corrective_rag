"""
Backward-compatible entrypoint for Corrective RAG Agent.
Delegates to the modern web server in `ui.server`.

Usage:
    python corrective_rag.py
"""

import uvicorn
from ui.server import app

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Corrective RAG Web Server starting...")
    print("🌐 Access the UI at: http://localhost:8000")
    print("=" * 60)
    uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=True)