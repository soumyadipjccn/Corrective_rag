"""
Command Line Interface (CLI) entrypoint for Corrective RAG Agent.
Supports document ingestion and querying from the terminal.

Usage:
    python main.py --url "https://arxiv.org/pdf/2307.09288.pdf" --query "What is Llama 2?"
"""

import argparse
import sys
import pprint
from src.config.settings import get_settings
from src.service import CRAGService
from src.utils.logger import get_logger

logger = get_logger("CRAG-CLI")


def parse_args():
    parser = argparse.ArgumentParser(description="Corrective RAG Agent CLI")
    parser.add_argument("--url", type=str, help="Document URL to ingest (PDF or Web Page)")
    parser.add_argument("--file", type=str, help="Local file path to ingest (.pdf, .txt, .md)")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the CRAG agent")
    parser.add_argument("--chat-model", type=str, default=None, help="Custom or preset NVIDIA chat model ID (e.g. meta/llama-3.1-70b-instruct, deepseek-ai/deepseek-r1)")
    parser.add_argument("--embed-model", type=str, default=None, help="Custom or preset embedding model ID (e.g. voyage-3, nvidia/nv-embedqa-e5-v5, baai/bge-m3)")
    parser.add_argument("--nvidia-api-key", type=str, default=None, help="General NVIDIA API key override")
    parser.add_argument("--chat-api-key", type=str, default=None, help="Distinct NVIDIA API key override for Chat LLM")
    parser.add_argument("--embed-api-key", type=str, default=None, help="Distinct API key override for Embedding model")
    parser.add_argument("--voyage-api-key", type=str, default=None, help="Voyage AI API key override for embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()

    if args.nvidia_api_key:
        settings.nvidia_api_key = args.nvidia_api_key
    if args.chat_api_key:
        settings.nvidia_chat_api_key = args.chat_api_key
    if args.embed_api_key:
        settings.nvidia_embedding_api_key = args.embed_api_key
        if args.embed_model and "voyage" in args.embed_model.lower():
            settings.voyage_api_key = args.embed_api_key
    if args.voyage_api_key:
        settings.voyage_api_key = args.voyage_api_key

    if not settings.effective_chat_api_key:
        logger.error("NVIDIA API Key for Chat LLM is not set in environment or CLI args.")
        sys.exit(1)

    if (args.url or args.file) and not settings.effective_embedding_api_key:
        provider = "Voyage AI" if settings.is_voyage_embedding else "NVIDIA"
        logger.error(f"{provider} Embedding API Key is not set in environment or CLI args.")
        sys.exit(1)

    service = CRAGService(
        settings=settings,
        chat_model_name=args.chat_model,
        embedding_model_name=args.embed_model,
    )

    # Ingest document if provided
    if args.url:
        logger.info(f"Ingesting URL: {args.url}")
        res = service.ingest_url(args.url)
        if res.status != "success":
            logger.error(f"Ingestion failed: {res.error}")
            sys.exit(1)
        logger.info(f"Ingested {res.doc_count} pages into {res.chunk_count} chunks.")
    elif args.file:
        logger.info(f"Ingesting file: {args.file}")
        res = service.ingest_file(args.file)
        if res.status != "success":
            logger.error(f"Ingestion failed: {res.error}")
            sys.exit(1)
        logger.info(f"Ingested {res.doc_count} pages into {res.chunk_count} chunks.")

    logger.info(f"Executing question: '{args.query}'")
    print("\n" + "=" * 60)
    print(f"QUERY: {args.query}")
    print("=" * 60)

    for node_name, state in service.stream_query(args.query):
        print(f"\n[NODE COMPLETED]: {node_name}")
        if node_name == "grade_documents":
            print(f" -> Web search needed: {state.get('run_web_search')}")
            print(f" -> Relevant docs count: {len(state.get('documents', []))}")
        elif node_name == "transform_query":
            print(f" -> Rewritten Query: '{state.get('question')}'")

    final_generation = state.get("generation", "No generation produced.")
    print("\n" + "=" * 60)
    print("FINAL GENERATION:")
    print("=" * 60)
    print(final_generation)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
