"""
chunker.py — Document chunking strategies.

WHY DO WE CHUNK AT ALL?
LLMs have a context window limit (e.g. 8192 tokens for LLaMA 3).
A PDF might be 50,000 tokens. We can't feed the whole thing to the LLM.

So we:
1. Split the document into small chunks (e.g. 512 characters each)
2. Embed each chunk as a vector
3. At query time, only retrieve the TOP 5 most relevant chunks
4. Feed only those 5 chunks to the LLM

This way we stay within the context window AND only send relevant info.

WHY NOT JUST SPLIT AT FIXED SIZES?
"Hello, my name is John. I work at" | "Anthropic as an engineer."
Fixed splitting can cut sentences in half, losing meaning.

BETTER APPROACH — RecursiveCharacterTextSplitter:
Tries to split at natural boundaries in this order:
1. Paragraph breaks (\n\n)
2. Line breaks (\n)
3. Sentences (. )
4. Words ( )
5. Characters (last resort)

OVERLAP — WHY IS IT IMPORTANT?
Chunk 1: "...the transformer uses attention mechanism to..."
Chunk 2: "...weight different parts of the input sequence..."

Without overlap: question about "attention mechanism" might only match
chunk 1, missing the continuation in chunk 2.

With overlap of 64 chars: chunk 2 starts with the end of chunk 1,
so both chunks contain "attention mechanism" context.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from loguru import logger
import hashlib

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:
    """
    Splits documents into overlapping chunks for embedding.

    We use RecursiveCharacterTextSplitter — LangChain's best general-purpose
    splitter. It tries paragraph → sentence → word boundaries before
    resorting to hard character cuts.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,          # Max chars per chunk (512)
            chunk_overlap=CHUNK_OVERLAP,    # Overlap between chunks (64)
            length_function=len,            # Use character count, not token count
            separators=[
                "\n\n",   # Try paragraph breaks first
                "\n",     # Then line breaks
                ". ",     # Then sentence endings
                "! ",
                "? ",
                ", ",     # Then clauses
                " ",      # Then words
                "",       # Last resort: hard cut
            ]
        )
        logger.info(
            f"Chunker initialized: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of documents into chunks.

        Each chunk keeps the original document's metadata PLUS
        we add a unique chunk_id (hash of content) for deduplication.

        WHY HASH-BASED IDs?
        If you ingest the same document twice, the chunks will have the
        same hash → ChromaDB won't store duplicates.
        Deterministic IDs also let us update specific chunks later.
        """
        if not documents:
            logger.warning("No documents to chunk")
            return []

        all_chunks = []

        for doc in documents:
            # Skip documents with very little content
            if len(doc.page_content.strip()) < 50:
                continue

            # Split this document into chunks
            chunks = self.splitter.split_documents([doc])

            # Add chunk-level metadata
            for i, chunk in enumerate(chunks):
                # Generate a unique ID from content hash
                # sha256 of content → take first 16 chars → unique enough
                chunk_id = hashlib.sha256(
                    chunk.page_content.encode()
                ).hexdigest()[:16]

                chunk.metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "char_count": len(chunk.page_content),
                })

            all_chunks.extend(chunks)

        # Remove duplicate chunks (same content from re-ingestion)
        all_chunks = self._deduplicate(all_chunks)

        logger.info(
            f"Chunked {len(documents)} documents → {len(all_chunks)} chunks"
        )
        return all_chunks

    def _deduplicate(self, chunks: list[Document]) -> list[Document]:
        """
        Remove chunks with identical content.

        Uses chunk_id (content hash) for deduplication.
        This means re-ingesting the same document is safe — no duplicates
        will be stored in the vector DB.
        """
        seen_ids = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", "")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)

        removed = len(chunks) - len(unique_chunks)
        if removed > 0:
            logger.info(f"Deduplication removed {removed} duplicate chunks")

        return unique_chunks
