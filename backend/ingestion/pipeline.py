"""
pipeline.py — Ingestion pipeline orchestrator.

This is the glue that connects:
  Loader → Chunker → Embedder

SINGLE RESPONSIBILITY:
Each component does one thing:
- Loader: raw source → Documents
- Chunker: Documents → smaller chunk Documents
- Embedder: chunk Documents → vectors stored in ChromaDB

This pipeline just calls them in order and handles errors.

WHY A SEPARATE PIPELINE CLASS?
So routes.py stays clean. Instead of:
  docs = loader.load(...)
  chunks = chunker.chunk(docs)
  count = embedder.add_documents(chunks)
  ...error handling...

We just call:
  result = pipeline.ingest_url(url)
"""

from loguru import logger
from typing import Union

from backend.ingestion.loaders import MultiSourceLoader
from backend.ingestion.chunker import DocumentChunker
from backend.ingestion.embedder import EmbeddingStore


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
    source → load → chunk → embed → store
    """

    def __init__(self):
        self.loader = MultiSourceLoader()
        self.chunker = DocumentChunker()
        self.store = EmbeddingStore()  # Singleton — reuses existing instance

    def ingest_url(self, url: str) -> dict:
        """Ingest a URL or YouTube video."""
        logger.info(f"Pipeline: ingesting URL {url}")
        try:
            docs = self.loader.load(url, source_type="auto")
            chunks = self.chunker.chunk(docs)
            count = self.store.add_documents(chunks)
            return {
                "success": True,
                "source": url,
                "docs_loaded": len(docs),
                "chunks_created": count,
            }
        except Exception as e:
            logger.error(f"Ingestion failed for {url}: {e}")
            return {"success": False, "source": url, "error": str(e)}

    def ingest_file(self, file_bytes: bytes, filename: str) -> dict:
        """Ingest an uploaded PDF or CSV file."""
        logger.info(f"Pipeline: ingesting file {filename}")
        try:
            ext = filename.rsplit(".", 1)[-1].lower()
            source_type = "pdf" if ext == "pdf" else "csv"
            docs = self.loader.load(file_bytes, source_type=source_type, filename=filename)
            chunks = self.chunker.chunk(docs)
            count = self.store.add_documents(chunks)
            return {
                "success": True,
                "source": filename,
                "docs_loaded": len(docs),
                "chunks_created": count,
            }
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}")
            return {"success": False, "source": filename, "error": str(e)}

    def reset(self):
        """Wipe the vector store."""
        self.store.delete_collection()
        logger.warning("Vector store reset")
