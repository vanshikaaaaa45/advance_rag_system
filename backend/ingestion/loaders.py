"""
loaders.py — Multi-source document loaders.

WHAT IS A DOCUMENT LOADER?
A loader takes raw input (a PDF file, a URL, a YouTube video) and converts
it into a list of LangChain Document objects. Each Document has:
  - page_content: str  → the actual text
  - metadata: dict     → source, page number, title, etc.

We need metadata because later when we show citations to the user,
we want to say "This answer came from page 3 of your PDF" not just
"some chunk of text".

WHY MULTIPLE SOURCES?
Real enterprise RAG systems don't just eat PDFs. Your knowledge base
might be: internal docs (PDF) + company wiki (URLs) + meeting recordings
(YouTube) + data exports (CSV). We handle all of them here.
"""

import tempfile
import os
import re
from pathlib import Path
from typing import Union
from loguru import logger

from langchain_community.document_loaders import (
    PyPDFLoader,           # Extracts text from PDFs page by page
    WebBaseLoader,         # Fetches and parses HTML from a URL
    CSVLoader,             # Reads CSV rows as documents
    YoutubeLoader,         # Fetches YouTube video transcripts
)
from langchain.schema import Document


class MultiSourceLoader:
    """
    Unified loader that detects input type and routes to the right loader.

    DESIGN PATTERN: Instead of calling different loaders manually everywhere,
    we have one class with one method: load(source). It figures out what
    the source is and handles it. This is the "facade" pattern.
    """

    def load(self, source: Union[str, bytes], source_type: str = "auto",
             filename: str = "") -> list[Document]:
        """
        Load documents from any supported source.

        Args:
            source: URL string, file path, or raw bytes (for uploads)
            source_type: "pdf", "url", "csv", "youtube", or "auto"
            filename: original filename (used when source is bytes)

        Returns:
            List of Document objects ready for chunking
        """
        # Auto-detect type from the source string
        if source_type == "auto" and isinstance(source, str):
            source_type = self._detect_type(source)

        logger.info(f"Loading source type: {source_type}")

        if source_type == "youtube":
            return self._load_youtube(source)
        elif source_type == "url":
            return self._load_url(source)
        elif source_type == "pdf":
            return self._load_pdf(source, filename)
        elif source_type == "csv":
            return self._load_csv(source, filename)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _detect_type(self, source: str) -> str:
        """
        Automatically detect what kind of source this is.

        HOW: We check the URL/path for known patterns.
        YouTube URLs contain 'youtube.com' or 'youtu.be'.
        PDF files end in .pdf.
        CSV files end in .csv.
        Everything else is treated as a generic URL.
        """
        source_lower = source.lower()

        if "youtube.com" in source_lower or "youtu.be" in source_lower:
            return "youtube"
        elif source_lower.endswith(".pdf"):
            return "pdf"
        elif source_lower.endswith(".csv"):
            return "csv"
        else:
            return "url"

    def _load_youtube(self, url: str) -> list[Document]:
        """
        Load transcript from a YouTube video.

        HOW IT WORKS:
        YouTube auto-generates transcripts for most videos.
        YoutubeLoader calls YouTube's transcript API to fetch the text.
        No audio processing needed — just pure text retrieval.

        WHY THIS IS POWERFUL:
        Any conference talk, tutorial, or lecture on YouTube becomes
        queryable. Imagine ingesting 50 ML conference talks and asking
        "What did Andrej Karpathy say about tokenization?"
        """
        try:
            logger.info(f"Loading YouTube transcript: {url}")
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=True,    # Adds title, author to metadata
                language=["en"],        # Try English first
            )
            docs = loader.load()
            logger.info(f"YouTube: loaded {len(docs)} transcript chunks")
            return docs
        except Exception as e:
            logger.error(f"YouTube load failed: {e}")
            raise

    def _load_url(self, url: str) -> list[Document]:
        """
        Fetch and parse a web page.

        HOW IT WORKS:
        WebBaseLoader uses requests to fetch the HTML, then BeautifulSoup
        to strip tags and extract clean text. The result is the visible
        text content of the page.

        LIMITATION: JavaScript-rendered pages won't work well here
        (SPAs, React apps). For those you'd need Playwright/Selenium.
        For most documentation sites and blogs, this works perfectly.
        """
        try:
            logger.info(f"Loading URL: {url}")
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={
                    # Only extract text from these HTML tags
                    # Ignores nav bars, footers, ads etc.
                    "parse_only": None  # None = parse everything
                }
            )
            docs = loader.load()

            # Clean up the text — web pages have lots of whitespace
            for doc in docs:
                doc.page_content = self._clean_text(doc.page_content)
                doc.metadata["source"] = url
                doc.metadata["source_type"] = "url"

            logger.info(f"URL: loaded {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"URL load failed: {e}")
            raise

    def _load_pdf(self, source: Union[str, bytes], filename: str = "") -> list[Document]:
        """
        Load and extract text from a PDF.

        HOW IT WORKS:
        PyPDFLoader reads the PDF page by page using pypdf.
        Each page becomes a separate Document with page number in metadata.

        WHY PAGE BY PAGE?
        - Preserves page number for citations ("See page 12")
        - Avoids loading huge PDFs into memory all at once
        - Allows chunking to respect page boundaries

        HANDLING BYTES vs PATH:
        When a user uploads a file via the API, FastAPI gives us bytes.
        We write to a temp file, load it, then delete the temp file.
        """
        try:
            if isinstance(source, bytes):
                # Write bytes to a temporary file
                # tempfile.NamedTemporaryFile creates a file that auto-deletes
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False  # We'll delete manually after loading
                ) as tmp:
                    tmp.write(source)
                    tmp_path = tmp.name

                logger.info(f"PDF from upload: {filename}, temp path: {tmp_path}")
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                os.unlink(tmp_path)  # Delete temp file
            else:
                logger.info(f"PDF from path: {source}")
                loader = PyPDFLoader(str(source))
                docs = loader.load()

            # Enrich metadata
            for doc in docs:
                doc.metadata["source"] = filename or str(source)
                doc.metadata["source_type"] = "pdf"
                doc.page_content = self._clean_text(doc.page_content)

            # Filter out empty pages (scanned PDFs sometimes have blank pages)
            docs = [d for d in docs if len(d.page_content.strip()) > 50]

            logger.info(f"PDF: loaded {len(docs)} pages")
            return docs

        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            raise

    def _load_csv(self, source: Union[str, bytes], filename: str = "") -> list[Document]:
        """
        Load a CSV file, treating each row as a document.

        HOW IT WORKS:
        CSVLoader reads each row and combines all columns into a single
        text string. Good for FAQ datasets, product catalogs, etc.

        EXAMPLE CSV use case:
        A company FAQ with columns: question, answer, category
        Each row becomes: "question: How do I reset? answer: Click settings..."
        Then users can ask questions and get matched to the right FAQ row.
        """
        try:
            if isinstance(source, bytes):
                with tempfile.NamedTemporaryFile(
                    suffix=".csv",
                    delete=False,
                    mode='wb'
                ) as tmp:
                    tmp.write(source)
                    tmp_path = tmp.name

                loader = CSVLoader(tmp_path)
                docs = loader.load()
                os.unlink(tmp_path)
            else:
                loader = CSVLoader(str(source))
                docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename or str(source)
                doc.metadata["source_type"] = "csv"

            logger.info(f"CSV: loaded {len(docs)} rows as documents")
            return docs

        except Exception as e:
            logger.error(f"CSV load failed: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean up messy text from web pages and PDFs.

        Common problems we fix:
        - Multiple consecutive newlines (\\n\\n\\n → \\n\\n)
        - Multiple spaces (   → single space)
        - Weird unicode whitespace characters
        """
        # Replace multiple whitespace chars with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace 3+ newlines with 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
