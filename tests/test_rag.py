"""
Tests for RAG system components
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor, DocumentChunk
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore


class TestDocumentProcessor:
    """Test document processing"""

    def test_clean_text(self):
        """Test text cleaning"""
        processor = DocumentProcessor()

        dirty_text = "This   has    multiple   spaces\n\n\nand\n\nlines"
        clean = processor.clean_text(dirty_text)

        assert "  " not in clean  # No double spaces
        assert clean == "This has multiple spaces and lines"

    def test_create_chunks(self):
        """Test chunking"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "Sentence one. " * 50  # Long text
        chunks = processor.create_chunks(text, source="test")

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(len(chunk.text) <= 150 for chunk in chunks)  # Rough size check

    def test_chunk_metadata(self):
        """Test chunk metadata"""
        processor = DocumentProcessor()

        text = "Test sentence."
        chunks = processor.create_chunks(text, source="test.txt")

        assert chunks[0].source == "test.txt"
        assert chunks[0].chunk_id == 0
        assert "chunk_size" in chunks[0].metadata


class TestEmbeddingGenerator:
    """Test embedding generation"""

    @pytest.fixture
    def generator(self):
        """Create generator instance"""
        return EmbeddingGenerator()

    def test_single_embedding(self, generator):
        """Test single text embedding"""
        text = "This is a test sentence."
        embedding = generator.encode(text, show_progress=False)

        assert embedding.shape[0] == 1
        assert embedding.shape[1] == generator.embedding_dim

    def test_batch_embedding(self, generator):
        """Test batch embedding"""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = generator.encode(texts, show_progress=False)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == generator.embedding_dim

    def test_embedding_dimension(self, generator):
        """Test embedding dimension"""
        dim = generator.get_embedding_dimension()
        assert dim > 0
        assert isinstance(dim, int)


class TestVectorStore:
    """Test vector store operations"""

    @pytest.fixture
    def vector_store(self):
        """Create vector store instance"""
        return VectorStore(embedding_dim=384)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks"""
        return [
            DocumentChunk(
                chunk_id=i,
                text=f"Sample text {i}",
                metadata={"pos": i},
                source="test.txt"
            )
            for i in range(5)
        ]

    def test_create_index(self, vector_store):
        """Test index creation"""
        vector_store.create_index(384)

        assert vector_store.index is not None
        assert vector_store.is_trained

    def test_add_vectors(self, vector_store, sample_chunks):
        """Test adding vectors"""
        import numpy as np

        vector_store.create_index(384)

        # Create random embeddings
        embeddings = np.random.rand(5, 384).astype('float32')

        vector_store.add_vectors(embeddings, sample_chunks)

        assert vector_store.index.ntotal == 5

    def test_search(self, vector_store, sample_chunks):
        """Test vector search"""
        import numpy as np

        vector_store.create_index(384)

        # Add vectors
        embeddings = np.random.rand(5, 384).astype('float32')
        vector_store.add_vectors(embeddings, sample_chunks)

        # Search
        query = np.random.rand(384).astype('float32')
        results = vector_store.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(chunk, DocumentChunk) for chunk, score in results)
        assert all(isinstance(score, float) for chunk, score in results)

    def test_stats(self, vector_store):
        """Test statistics"""
        vector_store.create_index(384)
        stats = vector_store.get_stats()

        assert "num_vectors" in stats
        assert "embedding_dim" in stats
        assert stats["embedding_dim"] == 384


class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self):
        """Test full ingestion pipeline"""
        # Create sample document
        text = "FastAPI is a modern web framework. " * 10

        # Process
        processor = DocumentProcessor(chunk_size=100)
        chunks = processor.create_chunks(text, source="test")

        # Generate embeddings
        generator = EmbeddingGenerator()
        texts = [chunk.text for chunk in chunks]
        embeddings = generator.encode(texts, show_progress=False)

        # Create vector store
        store = VectorStore()
        store.create_index(generator.get_embedding_dimension())
        store.add_vectors(embeddings, chunks)

        # Search
        query = "What is FastAPI?"
        query_embedding = generator.encode(query, show_progress=False)
        results = store.search(query_embedding, top_k=2)

        assert len(results) > 0
        assert results[0][1] > 0  # Has similarity score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
