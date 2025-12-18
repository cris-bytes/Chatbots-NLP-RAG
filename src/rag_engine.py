"""
PARTE 2: RAG (Retrieval Augmented Generation) Engine

Handles:
- Question embedding
- Context retrieval
- Prompt construction
- LLM invocation
- Hallucination control
- Fallback mechanisms
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .document_processor import DocumentChunk
from .config import settings


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    context_used: List[str]
    sources: List[Dict[str, Any]]
    confidence: float
    has_sufficient_context: bool


class RAGEngine:
    """
    RAG Engine for intelligent question answering
    """

    def __init__(self,
                 embedding_generator: EmbeddingGenerator = None,
                 vector_store: VectorStore = None,
                 use_openai: bool = True):
        """
        Initialize RAG engine

        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
            use_openai: Whether to use OpenAI API (requires API key)
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()
        self.use_openai = use_openai and settings.openai_api_key is not None

        if self.use_openai:
            import openai
            openai.api_key = settings.openai_api_key
            self.openai = openai
            print("RAG Engine initialized with OpenAI")
        else:
            print("RAG Engine initialized (local mode - using fallback)")

    def retrieve_context(self, query: str, top_k: int = None) -> List[tuple]:
        """
        Retrieve relevant context for a query

        Args:
            query: User question
            top_k: Number of chunks to retrieve

        Returns:
            List of (chunk, score) tuples
        """
        if top_k is None:
            top_k = settings.top_k_chunks

        # Generate query embedding
        query_embedding = self.embedding_generator.encode(query, show_progress=False)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

    def build_prompt(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """
        Build prompt with context for LLM

        Args:
            query: User question
            context_chunks: Retrieved context chunks

        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([
            f"[Context {i+1}]:\n{chunk.text}"
            for i, chunk in enumerate(context_chunks)
        ])

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information from the context below
2. If the context doesn't contain enough information to answer, say "I don't have sufficient information to answer this question based on the provided document."
3. Be concise and precise
4. If you're not confident, express uncertainty
5. Cite which context section you're using when relevant

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

        return prompt

    def call_llm(self, prompt: str) -> str:
        """
        Call LLM to generate answer

        Args:
            prompt: Formatted prompt

        Returns:
            Generated answer
        """
        if self.use_openai:
            try:
                response = self.openai.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Never make up information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling OpenAI: {e}")
                return self._fallback_answer(prompt)
        else:
            return self._fallback_answer(prompt)

    def _fallback_answer(self, prompt: str) -> str:
        """
        Simple fallback when LLM is not available

        Args:
            prompt: The prompt

        Returns:
            Fallback message
        """
        return "I apologize, but I cannot generate an answer at this moment. The language model is not available. Please configure OpenAI API key or use a local model."

    def assess_context_quality(self, query: str, chunks: List[DocumentChunk], scores: List[float]) -> tuple:
        """
        Assess if retrieved context is sufficient

        Args:
            query: User query
            chunks: Retrieved chunks
            scores: Similarity scores

        Returns:
            (has_sufficient_context, confidence_score)
        """
        # Simple heuristic: check if top score is above threshold
        RELEVANCE_THRESHOLD = 0.3

        if not scores or len(scores) == 0:
            return False, 0.0

        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        has_sufficient_context = max_score > RELEVANCE_THRESHOLD
        confidence = min(max_score, 1.0)

        return has_sufficient_context, confidence

    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Main RAG pipeline: retrieve context and generate answer

        Args:
            query: User question

        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing query: {query}")
        print(f"{'='*60}\n")

        # Step 1: Retrieve context
        print("Step 1: Retrieving relevant context...")
        results = self.retrieve_context(query)

        if not results:
            return {
                "answer": "I don't have any information to answer this question.",
                "context_used": [],
                "sources": [],
                "confidence": 0.0,
                "has_sufficient_context": False
            }

        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]

        print(f"Retrieved {len(chunks)} context chunks")
        for i, (chunk, score) in enumerate(results):
            print(f"  Chunk {i+1}: Score={score:.3f}, Length={len(chunk.text)} chars")

        # Step 2: Assess context quality
        print("\nStep 2: Assessing context quality...")
        has_sufficient_context, confidence = self.assess_context_quality(query, chunks, scores)
        print(f"Has sufficient context: {has_sufficient_context}")
        print(f"Confidence: {confidence:.3f}")

        # Step 3: Build prompt
        print("\nStep 3: Building prompt...")
        prompt = self.build_prompt(query, chunks)

        # Step 4: Generate answer
        print("\nStep 4: Generating answer...")
        if has_sufficient_context:
            answer = self.call_llm(prompt)
        else:
            answer = "I don't have sufficient information to answer this question based on the provided document. The retrieved context doesn't seem relevant enough."

        # Prepare response
        context_used = [chunk.text for chunk in chunks]
        sources = [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "score": score,
                "preview": chunk.text[:100] + "..."
            }
            for chunk, score in results
        ]

        response = {
            "answer": answer,
            "context_used": context_used,
            "sources": sources,
            "confidence": confidence,
            "has_sufficient_context": has_sufficient_context
        }

        print("\n" + "="*60)
        print("Answer generated successfully")
        print("="*60 + "\n")

        return response

    def load_vector_store(self, path: str = None):
        """
        Load pre-built vector store

        Args:
            path: Path to vector store
        """
        self.vector_store.load(path)
        print("Vector store loaded successfully")
