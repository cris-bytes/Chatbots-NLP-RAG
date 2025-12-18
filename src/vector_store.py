"""
Vector Store module using FAISS

Handles vector indexing, storage, and similarity search
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .config import settings
from .document_processor import DocumentChunk


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search
    """

    def __init__(self, embedding_dim: int = None):
        """
        Initialize vector store

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.is_trained = False

    def create_index(self, embedding_dim: int = None):
        """
        Create a new FAISS index

        Args:
            embedding_dim: Dimension of embeddings
        """
        if embedding_dim:
            self.embedding_dim = embedding_dim

        if not self.embedding_dim:
            raise ValueError("Embedding dimension must be specified")

        # Using IndexFlatL2 for exact search with L2 distance
        # For cosine similarity, we normalize embeddings
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine with normalized vectors)
        self.is_trained = True
        print(f"Created FAISS index with dimension {self.embedding_dim}")

    def add_vectors(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """
        Add vectors to the index

        Args:
            embeddings: Numpy array of embeddings
            chunks: Corresponding document chunks
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")

        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        # Ensure embeddings are normalized for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)

        print(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Add vectors first.")

        # Ensure query is normalized
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(distance)))

        return results

    def save(self, path: Path = None):
        """
        Save index and chunks to disk

        Args:
            path: Directory path to save to
        """
        if path is None:
            path = settings.vector_store_path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))

        # Save chunks
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        # Save metadata
        metadata_path = path / "metadata.pkl"
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_vectors': self.index.ntotal if self.index else 0
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Saved vector store to {path}")

    def load(self, path: Path = None):
        """
        Load index and chunks from disk

        Args:
            path: Directory path to load from
        """
        if path is None:
            path = settings.vector_store_path

        path = Path(path)

        # Load FAISS index
        index_path = path / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load chunks
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.embedding_dim = metadata['embedding_dim']

        self.is_trained = True
        print(f"Loaded vector store from {path}")
        print(f"Index contains {self.index.ntotal} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store

        Returns:
            Dictionary of statistics
        """
        return {
            'num_vectors': self.index.ntotal if self.index else 0,
            'num_chunks': len(self.chunks),
            'embedding_dim': self.embedding_dim,
            'is_trained': self.is_trained
        }
