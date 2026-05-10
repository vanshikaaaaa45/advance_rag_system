"""
retrieval/sparse.py — BM25 sparse keyword retrieval.

WHAT IS BM25?
BM25 (Best Match 25) is the classic information retrieval algorithm.
It's what search engines used before neural embeddings existed.
Google's early search was essentially BM25.

HOW IT WORKS:
BM25 scores a document for a query based on:
1. Term Frequency (TF): how often query words appear in the chunk
2. Inverse Document Frequency (IDF): rare words score higher than common words
3. Document length normalization: penalizes very long documents

Score formula (simplified):
score = Σ IDF(word) × TF(word, doc) / (TF + k1 × (1 - b + b × doc_len/avg_len))
where k1=1.5, b=0.75 are tuning constants

WHY DO WE NEED BM25 ALONGSIDE DENSE?
Dense retrieval is great for semantic similarity but fails for:
- Exact product codes: "GPT-4o" vs "GPT-4" — dense might treat as same
- Rare proper nouns: "Anthropic" might have weak embeddings
- Abbreviations: "RAG" might not semantically match "Retrieval-Augmented Generation"
- Numbers: "Section 4.2.1" needs exact match

BM25 catches all of these. Together they cover both semantic AND lexical matching.

LAZY LOADING:
We build the BM25 index on first use, not at startup.
Why? The BM25 index is built over ALL documents in the collection.
If we built at startup and someone adds documents later, the index would be stale.
Rebuilding on each query ensures freshness.
For large collections (100k+ docs), you'd cache and invalidate smarter.
"""

from rank_bm25 import BM25Okapi
from langchain.schema import Document
from loguru import logger

from backend.ingestion.embedder import EmbeddingStore
from backend.config import TOP_K_SPARSE


class SparseRetriever:
    def __init__(self):
        self.store = EmbeddingStore()

    def retrieve(self, query: str, k: int = TOP_K_SPARSE) -> list[Document]:
        """
        BM25 keyword search over all stored documents.

        Steps:
        1. Fetch all documents from ChromaDB
        2. Tokenize all docs + query (simple whitespace split)
        3. Build BM25 index
        4. Score all docs for the query
        5. Return top-K by score
        """
        # Get all documents from the vector store
        all_docs = self.store.get_all_documents()

        if not all_docs:
            logger.warning("BM25: no documents in store")
            return []

        logger.info(f"BM25: building index over {len(all_docs)} documents")

        # Tokenize: split text into words, lowercase
        # Simple but effective for English text.
        # For production: use NLTK or spaCy for better tokenization
        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in all_docs
        ]

        # Build BM25 index
        # BM25Okapi is the most commonly used variant of BM25
        bm25 = BM25Okapi(tokenized_corpus)

        # Tokenize query the same way
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        # Returns array of scores, one per document
        scores = bm25.get_scores(tokenized_query)

        # Pair each doc with its score, sort descending
        scored_docs = sorted(
            zip(all_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top-K and add metadata
        results = []
        for rank, (doc, score) in enumerate(scored_docs[:k]):
            # Skip docs with zero score (no query terms matched at all)
            if score == 0:
                break
            doc.metadata["bm25_score"] = round(float(score), 4)
            doc.metadata["bm25_rank"] = rank + 1
            results.append(doc)

        logger.info(f"BM25: returned {len(results)} results for '{query[:40]}'")
        return results
