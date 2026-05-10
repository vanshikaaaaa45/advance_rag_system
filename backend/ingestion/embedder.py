"""
embedder.py — Embedding and vector store management.

WHAT ARE EMBEDDINGS?
An embedding model converts text → a list of numbers (a vector).
"The cat sat on the mat" → [0.23, -0.41, 0.87, ..., 0.12]  (384 numbers)

Similar meaning = similar vectors (close in vector space).
"The dog rested on the rug" → nearly same vector as above.
"Quantum physics equations" → very different vector.

This lets us do SEMANTIC search — find chunks that MEAN the same thing
as the query, even if they use different words.

WHY sentence-transformers?
- Runs 100% locally — no API calls, no cost, no internet needed
- all-MiniLM-L6-v2 is small (80MB) but very fast and good quality
- Same model used for both indexing AND querying (critical!)

WHY CHROMADB?
- Stores text + embedding vector + metadata together
- Persists to disk — survives server restarts
- Fast approximate nearest neighbour search via HNSW index
- Free, local, no external service needed

SINGLETON PATTERN:
Loading the embedding model takes ~2 seconds.
We use a singleton so it's loaded once and reused for every request.
"""

import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from typing import Optional

from backend.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_DENSE,
)


class EmbeddingStore:
    """
    Manages embedding generation and ChromaDB storage/retrieval.
    Singleton: only one instance exists across the entire app lifecycle.
    """

    _instance: Optional["EmbeddingStore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("Initializing embedding model and ChromaDB...")

        # ── Load embedding model ──────────────────────────────────────────
        # First call downloads model (~80MB) to ~/.cache/huggingface/
        # Subsequent calls load from disk cache instantly.
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,  # Normalize → cosine similarity works correctly
                "batch_size": 32,
            }
        )
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")

        # ── Set up ChromaDB ───────────────────────────────────────────────
        # PersistentClient saves to disk → data survives restarts
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )

        # get_or_create: safe to call multiple times
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self._initialized = True
        logger.info(
            f"ChromaDB ready: '{CHROMA_COLLECTION_NAME}' "
            f"with {self.collection.count()} existing chunks"
        )

    def add_documents(self, chunks: list[Document]) -> int:
        """
        Embed chunks and upsert into ChromaDB.

        UPSERT = insert if new ID, update if ID already exists.
        This makes re-ingestion safe — no duplicates.
        """
        if not chunks:
            return 0

        logger.info(f"Embedding {len(chunks)} chunks...")

        texts = [c.page_content for c in chunks]
        ids = [c.metadata["chunk_id"] for c in chunks]

        # ChromaDB only accepts str/int/float/bool in metadata
        metadatas = []
        for chunk in chunks:
            meta = {}
            for k, v in chunk.metadata.items():
                meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
            metadatas.append(meta)

        # This is where the actual embedding computation happens
        # sentence-transformers processes in batches of 32
        embeddings = self.embedding_model.embed_documents(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            f"Stored {len(chunks)} chunks. "
            f"Collection total: {self.collection.count()}"
        )
        return len(chunks)

    def similarity_search(self, query: str, k: int = TOP_K_DENSE) -> list[Document]:
        """
        Semantic search: find K chunks most similar to the query.

        HOW:
        1. Embed query with same model used for indexing
        2. ChromaDB computes cosine similarity vs all stored vectors
        3. Returns top K sorted by similarity (highest first)

        distance → score: score = 1 - distance
        (ChromaDB returns distance where 0 = identical, 2 = opposite)
        """
        if self.collection.count() == 0:
            logger.warning("Collection empty — ingest documents first")
            return []

        query_embedding = self.embedding_model.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        for i, (text, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            doc = Document(
                page_content=text,
                metadata={**meta, "similarity_score": round(1 - dist, 4), "rank": i + 1}
            )
            documents.append(doc)

        logger.info(f"Dense search returned {len(documents)} results")
        return documents

    def get_all_documents(self) -> list[Document]:
        """Retrieve all chunks — used by BM25 to build its index."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=t, metadata=m)
            for t, m in zip(results["documents"], results["metadatas"])
        ]

    def delete_collection(self):
        """Wipe all stored documents and start fresh."""
        self.chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("Collection wiped and recreated")

    def count(self) -> int:
        return self.collection.count()
