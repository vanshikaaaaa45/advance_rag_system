"""
retrieval/fusion.py — Reciprocal Rank Fusion (RRF).

THE PROBLEM:
Dense retrieval returns chunks ranked by cosine similarity (0.0 to 1.0).
BM25 returns chunks ranked by BM25 score (0.0 to ~15.0).

These scores are on completely different scales!
We can't just add them: 0.87 (dense) + 12.3 (BM25) would always
let BM25 dominate, making dense retrieval useless.

THE SOLUTION — Reciprocal Rank Fusion:
Instead of combining raw scores, we combine RANKS.

RRF formula:
  RRF_score(doc) = Σ 1 / (k + rank_in_list)

Where:
  k = 60 (a constant — empirically found to work well, rarely changed)
  rank_in_list = position of doc in each retrieval list (1-indexed)

EXAMPLE:
Doc A: rank 1 in dense, rank 3 in BM25
  RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

Doc B: rank 2 in dense, rank 1 in BM25
  RRF = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252

Doc C: rank 1 in dense only (not in BM25)
  RRF = 1/(60+1) + 0 = 0.01639

WHY RRF WORKS:
- Scale-invariant: works regardless of score magnitudes
- Rank-based: top results from either list get boosted
- Missing results: if a doc only appears in one list, it still gets a score
- Documents appearing in BOTH lists get double credit → consensus boosts them

RRF consistently outperforms simple score combination in research papers.
It's the standard approach for hybrid retrieval.
"""

from langchain.schema import Document
from loguru import logger
from collections import defaultdict

from backend.config import RRF_K, TOP_K_RERANK


class RRFFusion:
    """
    Combines multiple ranked lists into one using Reciprocal Rank Fusion.
    """

    def __init__(self, k: int = RRF_K):
        # k=60 is the standard. Higher k = less weight difference between
        # top and bottom ranks. Lower k = top ranks matter much more.
        self.k = k

    def fuse(
        self,
        dense_results: list[Document],
        sparse_results: list[Document],
        top_k: int = TOP_K_RERANK
    ) -> list[Document]:
        """
        Fuse dense and sparse results using RRF.

        Args:
            dense_results: Ranked list from vector similarity search
            sparse_results: Ranked list from BM25 keyword search
            top_k: How many results to return after fusion

        Returns:
            Fused and re-ranked list of Documents with rrf_score in metadata
        """
        if not dense_results and not sparse_results:
            logger.warning("RRF: both result lists are empty")
            return []

        # rrf_scores: maps chunk_id → running RRF score
        rrf_scores: dict[str, float] = defaultdict(float)

        # doc_map: maps chunk_id → Document object (to reconstruct results)
        doc_map: dict[str, Document] = {}

        # Process dense results (1-indexed rank)
        for rank, doc in enumerate(dense_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            doc_map[chunk_id] = doc

        # Process sparse results
        for rank, doc in enumerate(sparse_results, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_scores[chunk_id] += 1.0 / (self.k + rank)
            # Only add to doc_map if not already there (dense takes priority
            # for the document object since it has similarity_score)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build output list
        fused = []
        for final_rank, chunk_id in enumerate(sorted_ids[:top_k], start=1):
            doc = doc_map[chunk_id]
            doc.metadata["rrf_score"] = round(rrf_scores[chunk_id], 6)
            doc.metadata["final_rank"] = final_rank

            # Track which retrieval methods found this doc
            in_dense = any(
                d.metadata.get("chunk_id") == chunk_id for d in dense_results
            )
            in_sparse = any(
                d.metadata.get("chunk_id") == chunk_id for d in sparse_results
            )
            doc.metadata["found_by"] = (
                "both" if in_dense and in_sparse
                else "dense" if in_dense
                else "sparse"
            )
            fused.append(doc)

        logger.info(
            f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} fused results"
        )
        return fused
