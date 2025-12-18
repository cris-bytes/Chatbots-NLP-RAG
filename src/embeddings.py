"""
Embeddings generation module

Handles creating embeddings using sentence-transformers
Supports multiple embedding models
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import settings


class EmbeddingGenerator:
    """
    Generate embeddings for text using sentence-transformers
    """

    def __init__(self, model_name: str = None):
        """
        Initialize embedding generator

        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name or settings.embedding_model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return embeddings

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings in batches for efficiency

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch, show_progress=False)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings

        Returns:
            Embedding dimension
        """
        return self.embedding_dim
