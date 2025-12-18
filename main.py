"""
Main entry point for the RAG Chatbot system

This script handles:
1. Document ingestion and processing
2. Vector store creation
3. Interactive Q&A
4. API server startup
"""

import argparse
import sys
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.config import settings
from src.api import start_api


def ingest_document(source_path: str, source_type: str = "auto"):
    """
    Ingest a document and create vector store

    Args:
        source_path: Path or URL to document
        source_type: Type of source (pdf, url, text, auto)
    """
    print("\n" + "="*60)
    print("DOCUMENT INGESTION PIPELINE")
    print("="*60 + "\n")

    # Step 1: Process document
    print("Step 1: Processing document...")
    processor = DocumentProcessor()
    chunks = processor.process_document(source_path, source_type)

    # Save chunks
    processor.save_chunks(chunks)

    # Step 2: Generate embeddings
    print("\nStep 2: Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_generator.encode_batch(texts)

    # Step 3: Create vector store
    print("\nStep 3: Creating vector store...")
    vector_store = VectorStore()
    vector_store.create_index(embedding_generator.get_embedding_dimension())
    vector_store.add_vectors(embeddings, chunks)

    # Step 4: Save vector store
    print("\nStep 4: Saving vector store...")
    vector_store.save()

    print("\n" + "="*60)
    print("INGESTION COMPLETE!")
    print("="*60)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Embedding dimension: {embedding_generator.get_embedding_dimension()}")
    print(f"Vector store saved to: {settings.vector_store_path}")
    print("\nYou can now:")
    print("  1. Start the API: python main.py --api")
    print("  2. Run interactive Q&A: python main.py --interactive")


def interactive_qa():
    """
    Interactive Q&A session
    """
    print("\n" + "="*60)
    print("INTERACTIVE Q&A SESSION")
    print("="*60 + "\n")

    # Initialize RAG engine
    print("Initializing RAG engine...")
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()

    try:
        vector_store.load()
    except FileNotFoundError:
        print("Error: Vector store not found. Please run ingestion first:")
        print("  python main.py --ingest <path_to_document>")
        sys.exit(1)

    rag_engine = RAGEngine(
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )

    print("\nRAG engine ready!")
    print("\nType your questions (or 'quit' to exit):")
    print("-" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Get answer
            result = rag_engine.answer_question(question)

            # Display answer
            print(f"\nBot: {result['answer']}")
            print(f"\nConfidence: {result['confidence']:.2%}")
            print(f"Sufficient context: {result['has_sufficient_context']}")

            # Display sources
            if result['sources']:
                print("\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. Score: {source['score']:.3f} - {source['preview']}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG Chatbot System")

    # Commands
    parser.add_argument("--ingest", type=str, help="Ingest a document (path or URL)")
    parser.add_argument("--type", type=str, default="auto",
                       choices=["auto", "pdf", "url", "text"],
                       help="Document type")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive Q&A session")
    parser.add_argument("--api", action="store_true",
                       help="Start API server")
    parser.add_argument("--host", type=str, help="API host")
    parser.add_argument("--port", type=int, help="API port")

    args = parser.parse_args()

    # Show help if no command
    if not any([args.ingest, args.interactive, args.api]):
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START GUIDE")
        print("="*60)
        print("\n1. Ingest a document:")
        print("   python main.py --ingest data/raw/document.pdf")
        print("\n2. Start interactive Q&A:")
        print("   python main.py --interactive")
        print("\n3. Start API server:")
        print("   python main.py --api")
        return

    # Execute commands
    if args.ingest:
        ingest_document(args.ingest, args.type)

    elif args.interactive:
        interactive_qa()

    elif args.api:
        start_api(args.host, args.port)


if __name__ == "__main__":
    main()
