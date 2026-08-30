"""
FastAPI backend server for Corrective RAG (CRAG) Agent.
Serves static HTML/CSS/JS frontend and provides REST & SSE streaming APIs.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings
from src.service import CRAGService
from src.utils.logger import get_logger

logger = get_logger("CRAG-Server")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Corrective RAG Agent API",
    description="REST & SSE API backend for Corrective RAG workflow",
    version="1.0.0",
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Instance
_service_instance: Optional[CRAGService] = None
_current_ingested_source: Optional[str] = None


def get_service() -> CRAGService:
    """Retrieve or initialize singleton CRAGService."""
    global _service_instance
    if _service_instance is None:
        settings = get_settings()
        _service_instance = CRAGService(settings=settings)
    return _service_instance


# ---------------------------------------------------------------------------
# Pydantic Request/Response Models
# ---------------------------------------------------------------------------

class ConfigUpdateRequest(BaseModel):
    nvidia_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    nvidia_chat_model: Optional[str] = None
    nvidia_embedding_model: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=500)


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="Document URL to ingest (PDF or Web Page)")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the CRAG pipeline")


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------

def serialize_document(doc: Any) -> Dict[str, Any]:
    """Serialize LangChain Document into JSON-compatible dict."""
    if isinstance(doc, Document):
        preview = doc.page_content[:300].strip() + ("..." if len(doc.page_content) > 300 else "")
        return {
            "source": doc.metadata.get("source", "Unknown"),
            "title": doc.metadata.get("title", doc.metadata.get("source", "No title")),
            "snippet": preview,
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
    return {"source": "Unknown", "snippet": str(doc), "page_content": str(doc), "metadata": {}}


def serialize_graph_state(state_keys: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize LangGraph state keys for API transport."""
    output = {}
    for k, v in state_keys.items():
        if k == "documents" and isinstance(v, list):
            output[k] = [serialize_document(d) for d in v]
        elif isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            output[k] = v
        else:
            output[k] = str(v)
    return output


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "Corrective RAG Agent"}


@app.get("/api/config")
def get_config():
    """Retrieve current system configuration and available model lists."""
    service = get_service()
    settings = service.settings
    return {
        "nvidia_api_key": settings.nvidia_api_key,
        "has_nvidia_key": bool(settings.nvidia_api_key),
        "tavily_api_key": settings.tavily_api_key,
        "has_tavily_key": bool(settings.tavily_api_key),
        "qdrant_url": settings.qdrant_url,
        "qdrant_api_key": settings.qdrant_api_key,
        "qdrant_collection_name": settings.qdrant_collection_name,
        "nvidia_chat_model": settings.nvidia_chat_model,
        "nvidia_embedding_model": settings.nvidia_embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "default_doc_url": settings.default_doc_url,
        "available_chat_models": settings.available_chat_models,
        "available_embedding_models": settings.available_embedding_models,
        "ingested_source": _current_ingested_source,
    }


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    """Update active settings and refresh service."""
    global _service_instance
    service = get_service()
    s = service.settings

    if req.nvidia_api_key is not None:
        s.nvidia_api_key = req.nvidia_api_key.strip()
    if req.tavily_api_key is not None:
        s.tavily_api_key = req.tavily_api_key.strip()
    if req.qdrant_url is not None:
        s.qdrant_url = req.qdrant_url.strip()
    if req.qdrant_api_key is not None:
        s.qdrant_api_key = req.qdrant_api_key.strip()
    if req.nvidia_chat_model is not None:
        s.nvidia_chat_model = req.nvidia_chat_model.strip()
    if req.nvidia_embedding_model is not None:
        s.nvidia_embedding_model = req.nvidia_embedding_model.strip()
    if req.chunk_size is not None:
        s.chunk_size = req.chunk_size
    if req.chunk_overlap is not None:
        s.chunk_overlap = req.chunk_overlap

    # Re-instantiate service with updated settings
    _service_instance = CRAGService(settings=s)
    logger.info("Configuration updated successfully.")

    return {
        "status": "success",
        "message": "Configuration updated successfully",
        "has_nvidia_key": bool(s.nvidia_api_key),
    }


@app.post("/api/ingest/url")
def ingest_url(req: IngestUrlRequest):
    """Ingest and index a remote document URL into Qdrant."""
    global _current_ingested_source
    service = get_service()
    if not service.settings.nvidia_api_key:
        raise HTTPException(status_code=400, detail="NVIDIA API Key is required before ingestion.")

    res = service.ingest_url(req.url.strip())
    if res.status == "success":
        _current_ingested_source = req.url.strip()
    return res.model_dump()


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest and index an uploaded document file (.pdf, .txt, .md) into Qdrant."""
    global _current_ingested_source
    service = get_service()
    if not service.settings.nvidia_api_key:
        raise HTTPException(status_code=400, detail="NVIDIA API Key is required before ingestion.")

    content = await file.read()
    filename = file.filename or "uploaded_document"

    res = service.ingest_bytes(content, filename)
    if res.status == "success":
        _current_ingested_source = filename
    return res.model_dump()


@app.post("/api/query")
def execute_query(req: QueryRequest):
    """Execute synchronous CRAG query."""
    service = get_service()
    if not service.settings.nvidia_api_key:
        raise HTTPException(status_code=400, detail="NVIDIA API Key is required to execute queries.")

    steps = []
    final_generation = ""

    try:
        for node_name, state_keys in service.stream_query(req.question.strip()):
            serialized = serialize_graph_state(state_keys)
            steps.append({
                "node": node_name,
                "state": serialized,
            })
            if "generation" in state_keys:
                final_generation = state_keys["generation"]

        return {
            "status": "success",
            "question": req.question,
            "steps": steps,
            "generation": final_generation,
        }
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/query/stream")
def stream_query(question: str = Query(..., min_length=1)):
    """
    Stream CRAG execution step by step via Server-Sent Events (SSE).
    Sends 'step', 'answer', 'error', and 'done' events.
    """
    service = get_service()
    if not service.settings.nvidia_api_key:
        def error_gen():
            payload = json.dumps({"error": "NVIDIA API Key is required to execute queries."})
            yield f"event: error\ndata: {payload}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    def event_stream():
        final_answer = ""
        try:
            for node_name, state_keys in service.stream_query(question.strip()):
                serialized = serialize_graph_state(state_keys)
                payload = {
                    "node": node_name,
                    "state": serialized,
                }
                yield f"event: step\ndata: {json.dumps(payload)}\n\n"

                if "generation" in state_keys:
                    final_answer = state_keys["generation"]

            # Final answer event
            answer_payload = json.dumps({
                "generation": final_answer,
                "status": "completed",
            })
            yield f"event: answer\ndata: {answer_payload}\n\n"

            # Completion marker
            yield f"event: done\ndata: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            err_payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err_payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ---------------------------------------------------------------------------
# Static File Mounting & Single-Page Application (SPA) Serving
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve main HTML index page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h2>Corrective RAG API Server is running. Static files not found.</h2>")


# Mount static assets directory after index route so /static/ works for css/js/etc
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
