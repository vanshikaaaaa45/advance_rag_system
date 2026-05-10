"""
retrieval/dense.py — Dense vector retrieval.

This is a thin wrapper around EmbeddingStore.similarity_search().
Why a separate file? Clean separation of concerns — retrieval logic
lives in retrieval/, not in ingestion/.

DENSE RETRIEVAL = semantic search using embedding vectors.
Good at: finding conceptually similar content even with different words.
Bad at: exact keyword matching, proper nouns, codes, acronyms.

Example:
Query: "how does attention work in transformers?"
Dense retrieval finds: chunks about "self-attention mechanism" even if
they never use the word "attention" — because the vectors are close.
"""

from langchain.schema import Document
from loguru import logger

from backend.ingestion.embedder import EmbeddingStore
from backend.config import TOP_K_DENSE


class DenseRetriever:
    def __init__(self):
        # Reuses the singleton EmbeddingStore — no re-initialization
        self.store = EmbeddingStore()

    def retrieve(self, query: str, k: int = TOP_K_DENSE) -> list[Document]:
        """
        Retrieve top-K semantically similar chunks.
        Returns Documents with similarity_score in metadata.
        """
        logger.info(f"Dense retrieval: '{query[:50]}...' top-{k}")
        results = self.store.similarity_search(query, k=k)
        return results
