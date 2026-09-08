"""
Policy RAG - Production Ingestion Service

Purpose:
    - Read PDF / TXT / MD / HTML policy documents
    - Clean and chunk text
    - Generate embeddings with Sentence Transformers
    - Store vectors in ChromaDB
    - Avoid duplicate ingestion
    - Detect changed documents and re-index them
    - Run continuously / safely for long periods
    - Recover from transient errors
    - Provide useful logging

Run once:
    python ingest.py

Run continuously:
    python ingest.py --watch

Force complete rebuild:
    python ingest.py --reset

Continuous rebuild every N seconds:
    python ingest.py --watch --interval 3600
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POLICIES_DIR = BASE_DIR / "policies"

# IMPORTANT:
# This is the database used by the application.
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "policy_docs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".htm",
}

# Number of documents processed in one batch
EMBED_BATCH_SIZE = 32

# Chroma insertion batch
CHROMA_BATCH_SIZE = 100

# For continuous mode
DEFAULT_WATCH_INTERVAL = 300  # 5 minutes

# Maximum retry delay
MAX_RETRY_DELAY = 300


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("policy-rag-ingest")


# ============================================================
# GLOBAL STATE
# ============================================================

STOP_REQUESTED = False


# ============================================================
# SIGNAL HANDLING
# ============================================================

def handle_shutdown(signum, frame):
    global STOP_REQUESTED

    logger.info(
        "Shutdown signal received (%s). Finishing current operation...",
        signum,
    )

    STOP_REQUESTED = True


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(path: Path) -> str:
    """
    Generate SHA256 hash for a file.

    This allows us to detect whether a document changed.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving readable text.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines).strip()


# ============================================================
# FILE EXTRACTION
# ============================================================

def read_pdf(path: Path) -> str:
    """
    Extract text from PDF.
    """

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""

            if text.strip():
                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        except Exception as exc:

            logger.warning(
                "Could not read page %s from %s: %s",
                page_number,
                path.name,
                exc,
            )

    return "\n\n".join(pages)


def read_text_file(path: Path) -> str:
    """
    Read TXT / MD files using several common encodings.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ]

    for encoding in encodings:

        try:

            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path}",
    )


def read_html(path: Path) -> str:
    """
    Extract readable text from HTML.
    """

    raw_html = read_text_file(path)

    soup = BeautifulSoup(
        raw_html,
        "html.parser",
    )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):
        element.decompose()

    return soup.get_text(
        separator="\n"
    )


def extract_text(path: Path) -> str:
    """
    Extract text based on file extension.
    """

    extension = path.suffix.lower()

    if extension == ".pdf":
        return read_pdf(path)

    if extension in {".txt", ".md"}:
        return read_text_file(path)

    if extension in {".html", ".htm"}:
        return read_html(path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# CHUNKING
# ============================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ============================================================
# DOCUMENT DISCOVERY
# ============================================================

def discover_documents() -> List[Path]:
    """
    Find all supported documents inside policies/.
    """

    if not POLICIES_DIR.exists():

        POLICIES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.warning(
            "Policies directory did not exist. "
            "Created: %s",
            POLICIES_DIR,
        )

        return []

    documents = []

    for path in POLICIES_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(path)

    documents.sort()

    return documents


# ============================================================
# CHROMADB
# ============================================================

def create_chroma_client():

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Opening ChromaDB: %s",
        CHROMA_DIR,
    )

    # IMPORTANT:
    # Do NOT use Settings(...) here.
    #
    # This avoids the ChromaDB configuration error:
    #
    # KeyError: '_type'
    #

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client


def get_or_create_collection(client):

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        logger.info(
            "Loaded existing collection: %s",
            COLLECTION_NAME,
        )

        return collection

    except Exception:

        logger.info(
            "Creating collection: %s",
            COLLECTION_NAME,
        )

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            },
        )

        return collection


# ============================================================
# MODEL
# ============================================================

def load_embedding_model():

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    logger.info(
        "Embedding model loaded successfully."
    )

    return model


# ============================================================
# EXISTING DOCUMENT INFORMATION
# ============================================================

def get_existing_sources(collection) -> Dict[str, Dict]:

    try:

        result = collection.get(
            include=["metadatas"]
        )

    except Exception as exc:

        logger.warning(
            "Could not read existing Chroma data: %s",
            exc,
        )

        return {}

    sources = {}

    metadatas = result.get(
        "metadatas",
        []
    )

    for metadata in metadatas:

        if not metadata:
            continue

        source = metadata.get("source")

        if not source:
            continue

        sources[source] = {
            "file_hash": metadata.get(
                "file_hash"
            )
        }

    return sources


# ============================================================
# DELETE SOURCE
# ============================================================

def delete_source(
    collection,
    source: str,
):
    """
    Delete all chunks belonging to a document.
    """

    try:

        collection.delete(
            where={
                "source": source
            }
        )

        logger.info(
            "Deleted previous chunks for: %s",
            source,
        )

    except Exception as exc:

        logger.warning(
            "Could not delete old chunks for %s: %s",
            source,
            exc,
        )


# ============================================================
# INGEST ONE DOCUMENT
# ============================================================

def ingest_document(
    path: Path,
    collection,
    embedding_model,
) -> Tuple[int, str]:

    relative_source = str(
        path.relative_to(BASE_DIR)
    )

    logger.info(
        "Processing: %s",
        relative_source,
    )

    file_hash = calculate_file_hash(path)

    raw_text = extract_text(path)

    text = clean_text(raw_text)

    if not text:

        logger.warning(
            "No text extracted from: %s",
            path.name,
        )

        return 0, file_hash

    chunks = split_text(text)

    if not chunks:

        logger.warning(
            "No chunks generated for: %s",
            path.name,
        )

        return 0, file_hash

    logger.info(
        "Generated %d chunks from %s",
        len(chunks),
        path.name,
    )

    # Remove old version first.
    delete_source(
        collection,
        relative_source,
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embeddings = embedding_model.encode(
        chunks,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = embeddings.tolist()

    # --------------------------------------------------------
    # CHROMA IDS
    # --------------------------------------------------------

    ids = []

    metadatas = []

    documents = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{hashlib.sha256(relative_source.encode()).hexdigest()[:16]}"
            f"_{index}"
        )

        ids.append(chunk_id)

        documents.append(chunk)

        metadatas.append(
            {
                "source": relative_source,
                "filename": path.name,
                "chunk_id": str(index),
                "file_hash": file_hash,
            }
        )

    # --------------------------------------------------------
    # INSERT IN BATCHES
    # --------------------------------------------------------

    for start in range(
        0,
        len(ids),
        CHROMA_BATCH_SIZE,
    ):

        end = min(
            start + CHROMA_BATCH_SIZE,
            len(ids),
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(
        "Successfully indexed %s (%d chunks)",
        relative_source,
        len(chunks),
    )

    return len(chunks), file_hash


# ============================================================
# REMOVE DELETED FILES
# ============================================================

def remove_deleted_documents(
    collection,
    discovered_documents: List[Path],
):
    """
    Remove vectors for files that no longer exist.
    """

    discovered_sources = {
        str(path.relative_to(BASE_DIR))
        for path in discovered_documents
    }

    existing_sources = get_existing_sources(
        collection
    )

    for source in existing_sources:

        if source not in discovered_sources:

            logger.info(
                "Document no longer exists. "
                "Removing from index: %s",
                source,
            )

            delete_source(
                collection,
                source,
            )


# ============================================================
# SINGLE INGESTION RUN
# ============================================================

def run_ingestion(
    embedding_model=None,
    reset=False,
):

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("POLICY RAG INGESTION STARTED")
    logger.info("=" * 70)

    documents = discover_documents()

    logger.info(
        "Found %d policy documents.",
        len(documents),
    )

    if not documents:

        logger.warning(
            "No policy documents found in %s",
            POLICIES_DIR,
        )

        return

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if reset:

        logger.warning(
            "RESET requested."
        )

        if CHROMA_DIR.exists():

            shutil.rmtree(
                CHROMA_DIR,
                ignore_errors=True,
            )

            logger.info(
                "Deleted ChromaDB: %s",
                CHROMA_DIR,
            )

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    client = create_chroma_client()

    collection = get_or_create_collection(
        client
    )

    logger.info(
        "Current vector count: %d",
        collection.count(),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if embedding_model is None:

        embedding_model = load_embedding_model()

    # --------------------------------------------------------
    # REMOVE DELETED DOCUMENTS
    # --------------------------------------------------------

    remove_deleted_documents(
        collection,
        documents,
    )

    # --------------------------------------------------------
    # CHECK EXISTING
    # --------------------------------------------------------

    existing_sources = get_existing_sources(
        collection
    )

    indexed_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks = 0

    # --------------------------------------------------------
    # PROCESS DOCUMENTS
    # --------------------------------------------------------

    for path in documents:

        if STOP_REQUESTED:

            logger.warning(
                "Stopping ingestion gracefully."
            )

            break

        relative_source = str(
            path.relative_to(BASE_DIR)
        )

        try:

            current_hash = calculate_file_hash(
                path
            )

            previous = existing_sources.get(
                relative_source
            )

            previous_hash = (
                previous.get("file_hash")
                if previous
                else None
            )

            # ------------------------------------------------
            # SKIP UNCHANGED DOCUMENT
            # ------------------------------------------------

            if (
                previous_hash
                and previous_hash == current_hash
            ):

                skipped_count += 1

                logger.info(
                    "SKIPPED unchanged: %s",
                    relative_source,
                )

                continue

            # ------------------------------------------------
            # INGEST
            # ------------------------------------------------

            chunks, _ = ingest_document(
                path=path,
                collection=collection,
                embedding_model=embedding_model,
            )

            indexed_count += 1
            total_chunks += chunks

        except Exception as exc:

            failed_count += 1

            logger.exception(
                "FAILED: %s",
                relative_source,
            )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    final_count = collection.count()

    logger.info("=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)

    logger.info(
        "Documents found: %d",
        len(documents),
    )

    logger.info(
        "Documents indexed: %d",
        indexed_count,
    )

    logger.info(
        "Documents skipped: %d",
        skipped_count,
    )

    logger.info(
        "Documents failed: %d",
        failed_count,
    )

    logger.info(
        "New chunks: %d",
        total_chunks,
    )

    logger.info(
        "Total vectors in ChromaDB: %d",
        final_count,
    )

    logger.info(
        "Execution time: %.2f seconds",
        elapsed,
    )

    logger.info(
        "Database: %s",
        CHROMA_DIR,
    )

    logger.info("=" * 70)


# ============================================================
# CONTINUOUS MODE
# ============================================================

def run_watch_mode(
    interval: int,
    reset: bool = False,
):

    logger.info("=" * 70)
    logger.info("24/7 POLICY INGESTION MODE")
    logger.info("=" * 70)

    logger.info(
        "Checking policies every %d seconds.",
        interval,
    )

    logger.info(
        "Press CTRL+C to stop."
    )

    embedding_model = None

    first_run = True

    retry_delay = 5

    while not STOP_REQUESTED:

        try:

            run_ingestion(
                embedding_model=embedding_model,
                reset=reset if first_run else False,
            )

            first_run = False

            retry_delay = 5

            # Keep model loaded between runs.
            if embedding_model is None:

                embedding_model = (
                    load_embedding_model()
                )

            logger.info(
                "Next check in %d seconds.",
                interval,
            )

            # Interruptible sleep
            for _ in range(interval):

                if STOP_REQUESTED:
                    break

                time.sleep(1)

        except Exception as exc:

            logger.exception(
                "Ingestion cycle failed: %s",
                exc,
            )

            logger.warning(
                "Retrying in %d seconds...",
                retry_delay,
            )

            for _ in range(retry_delay):

                if STOP_REQUESTED:
                    break

                time.sleep(1)

            retry_delay = min(
                retry_delay * 2,
                MAX_RETRY_DELAY,
            )

    logger.info(
        "24/7 ingestion service stopped."
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Policy RAG ingestion service"
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild ChromaDB.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_WATCH_INTERVAL,
        help=(
            "Seconds between checks "
            f"(default: {DEFAULT_WATCH_INTERVAL})"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.interval < 10:

        logger.warning(
            "Interval too small. "
            "Using 10 seconds."
        )

        args.interval = 10

    if args.watch:

        run_watch_mode(
            interval=args.interval,
            reset=args.reset,
        )

    else:

        run_ingestion(
            reset=args.reset
        )


if __name__ == "__main__":
    main()
