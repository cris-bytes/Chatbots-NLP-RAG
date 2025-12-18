"""
Example usage of the RAG Chatbot system

This script demonstrates the complete workflow:
1. Document ingestion
2. Interactive Q&A
"""

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine


def example_ingestion():
    """
    Example: Ingest a document and create vector store
    """
    print("\n" + "="*60)
    print("EXAMPLE: Document Ingestion")
    print("="*60 + "\n")

    # 1. Process document
    processor = DocumentProcessor()
    chunks = processor.process_document(
        "data/raw/fastapi_guide.txt",
        source_type="text"
    )

    print(f"\nProcessed {len(chunks)} chunks")

    # 2. Generate embeddings
    embedding_generator = EmbeddingGenerator()
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_generator.encode_batch(texts)

    print(f"Generated embeddings with dimension {embeddings.shape[1]}")

    # 3. Create and populate vector store
    vector_store = VectorStore()
    vector_store.create_index(embedding_generator.get_embedding_dimension())
    vector_store.add_vectors(embeddings, chunks)

    # 4. Save
    vector_store.save()
    processor.save_chunks(chunks)

    print("\nVector store saved successfully!")


def example_question_answering():
    """
    Example: Answer questions using RAG
    """
    print("\n" + "="*60)
    print("EXAMPLE: Question Answering")
    print("="*60 + "\n")

    # Initialize RAG engine
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()

    try:
        vector_store.load()
    except FileNotFoundError:
        print("Error: Vector store not found.")
        print("Please run example_ingestion() first.")
        return

    rag_engine = RAGEngine(
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )

    # Example questions
    questions = [
        "What is FastAPI?",
        "How do I install FastAPI?",
        "What are FastAPI's key features?",
        "How does FastAPI handle validation?",
        "What security features does FastAPI offer?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")

        result = rag_engine.answer_question(question)

        print(f"\nA: {result['answer']}\n")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Sufficient context: {result['has_sufficient_context']}")

        if result['sources']:
            print("\nTop source:")
            source = result['sources'][0]
            print(f"  Score: {source['score']:.3f}")
            print(f"  Preview: {source['preview']}")

        print()


def main():
    """Main example runner"""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_usage.py ingest    - Ingest document")
        print("  python example_usage.py qa        - Run Q&A examples")
        print("  python example_usage.py all       - Run both")
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        example_ingestion()
    elif command == "qa":
        example_question_answering()
    elif command == "all":
        example_ingestion()
        example_question_answering()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
