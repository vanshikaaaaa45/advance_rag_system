"""
retrieval/reranker.py — Cross-encoder reranking.

WHY DO WE NEED RERANKING?
Embedding similarity (dense retrieval) has a fundamental limitation:
it encodes query and document SEPARATELY, then compares the vectors.

This means it can't model INTERACTIONS between query and document.
Example: query = "What are the limitations of RAG?"
A chunk about "RAG advantages" might have high embedding similarity
(it's about RAG!) but is actually the WRONG answer.

CROSS-ENCODER:
A cross-encoder takes BOTH query and document as input simultaneously:
  Input: "[CLS] query [SEP] document [SEP]"
  Output: relevance score 0.0 to 1.0

Because it sees both together, it can model:
- Whether the document actually ANSWERS the question
- Semantic entailment (does doc support query's assumption?)
- Negation (doc says opposite of what query expects)

WHY NOT USE CROSS-ENCODER FOR EVERYTHING?
It's SLOW. For 1000 docs: embed once, compare vectors = fast.
Cross-encode 1000 (query, doc) pairs = ~10 seconds on CPU.

The trick: use fast retrieval (dense + BM25) to get top-20,
then use slow-but-accurate cross-encoder on just those 20.
This is called the "retrieve-then-rerank" pattern.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking dataset (510k queries)
- Very fast (6 layers) but still much better than bi-encoder alone
- ~80MB, runs on CPU, free
"""

from sentence_transformers import CrossEncoder
from langchain.schema import Document
from loguru import logger

from backend.config import RERANKER_MODEL, TOP_K_RERANK


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    Singleton pattern — model loaded once, reused across requests.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        # CrossEncoder from sentence-transformers
        # Downloads ~80MB model on first run
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512  # Max tokens for (query + doc) combined
        )
        self._initialized = True
        logger.info("Cross-encoder loaded")

    def rerank(self, query: str, documents: list[Document], top_k: int = TOP_K_RERANK) -> list[Document]:
        """
        Rerank documents by cross-encoder relevance score.

        Args:
            query: The user's question
            documents: Candidate documents from fusion step
            top_k: How many to return after reranking

        Returns:
            Documents sorted by cross-encoder score, highest first
        """
        if not documents:
            return []

        if len(documents) == 1:
            return documents

        logger.info(f"Reranking {len(documents)} documents...")

        # Create (query, document_text) pairs for the cross-encoder
        # The model scores each pair: how relevant is this doc to this query?
        pairs = [(query, doc.page_content) for doc in documents]

        # Get relevance scores — higher = more relevant
        # predict() runs the cross-encoder on all pairs (batched internally)
        scores = self.model.predict(pairs)

        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)

        # Sort by rerank score (highest = most relevant)
        reranked = sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)

        # Update final rank
        for i, doc in enumerate(reranked):
            doc.metadata["final_rank"] = i + 1

        top = reranked[:top_k]

        logger.info(
            f"Reranking complete. Top score: {top[0].metadata['rerank_score']:.3f}, "
            f"Bottom: {top[-1].metadata['rerank_score']:.3f}"
        )
        return top
