"""
Policy RAG App - Document Ingestion

This script:
1. Reads documents from ./policies
2. Supports PDF, TXT, MD and HTML
3. Cleans extracted text
4. Splits documents into overlapping chunks
5. Creates embeddings with Sentence Transformers
6. Stores chunks in ChromaDB
7. Uses the SAME database and collection expected by app.py
8. Automatically rebuilds the index when necessary

Run:
    python ingest.py

Force rebuild:
    python ingest.py --reset
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POLICIES_DIR = BASE_DIR / "policies"

# IMPORTANT:
# app.py MUST use this exact same directory.
CHROMA_DIR = BASE_DIR / "chroma_db"

# IMPORTANT:
# app.py MUST use this exact same collection.
COLLECTION_NAME = "policy_docs"


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ============================================================
# SUPPORTED FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".html",
    ".htm",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("policy-rag")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted document text.
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
# PDF
# ============================================================

def extract_pdf(path: Path) -> str:

    logger.info("Reading PDF: %s", path.name)

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:

            text = page.extract_text() or ""

            if text.strip():

                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        except Exception as exc:

            logger.warning(
                "Could not extract page %d from %s: %s",
                page_number,
                path.name,
                exc,
            )

    return "\n\n".join(pages)


# ============================================================
# TEXT / MARKDOWN
# ============================================================

def extract_text_file(path: Path) -> str:

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:

        try:

            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError:

            continue

    raise RuntimeError(
        f"Could not decode file: {path}"
    )


# ============================================================
# HTML
# ============================================================

def extract_html(path: Path) -> str:

    raw = extract_text_file(path)

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):

        tag.decompose()

    return soup.get_text(
        separator="\n"
    )


# ============================================================
# GENERIC EXTRACTION
# ============================================================

def extract_document(path: Path) -> str:

    extension = path.suffix.lower()

    if extension == ".pdf":

        return extract_pdf(path)

    if extension in {
        ".txt",
        ".md",
    }:

        return extract_text_file(path)

    if extension in {
        ".html",
        ".htm",
    }:

        return extract_html(path)

    raise ValueError(
        f"Unsupported extension: {extension}"
    )


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):

    if not text:

        return []

    if overlap >= chunk_size:

        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE"
        )

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= len(text):

            break

        start = end - overlap

    return chunks


# ============================================================
# FIND POLICIES
# ============================================================

def find_policy_files():

    if not POLICIES_DIR.exists():

        logger.warning(
            "Policies directory does not exist."
        )

        POLICIES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        return []

    files = []

    for file in POLICIES_DIR.rglob("*"):

        if not file.is_file():

            continue

        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:

            continue

        files.append(file)

    return sorted(files)


# ============================================================
# FILE HASH
# ============================================================

def calculate_hash(path: Path):

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:

                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# CHROMA CLIENT
# ============================================================

def create_chroma_client():

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "ChromaDB location:"
    )

    logger.info(
        "%s",
        CHROMA_DIR,
    )

    # IMPORTANT:
    # Do NOT use Settings(...)
    #
    # This avoids the previous:
    #
    # KeyError: '_type'
    #

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client


# ============================================================
# CREATE COLLECTION
# ============================================================

def create_collection(client):

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        logger.info(
            "Existing collection found: %s",
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
# DOCUMENT SOURCE
# ============================================================

def source_name(path: Path):

    return str(
        path.relative_to(BASE_DIR)
    ).replace("\\", "/")


# ============================================================
# INDEX ONE DOCUMENT
# ============================================================

def index_document(
    path: Path,
    collection,
    model,
):

    source = source_name(path)

    logger.info(
        "--------------------------------------------------"
    )

    logger.info(
        "Processing: %s",
        source,
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    raw_text = extract_document(path)

    text = clean_text(raw_text)

    if not text:

        logger.warning(
            "No text extracted from %s",
            source,
        )

        return 0

    # --------------------------------------------------------
    # CHUNK
    # --------------------------------------------------------

    chunks = chunk_text(text)

    logger.info(
        "Created %d chunks",
        len(chunks),
    )

    if not chunks:

        return 0

    # --------------------------------------------------------
    # DELETE OLD VERSION
    # --------------------------------------------------------

    try:

        collection.delete(
            where={
                "source": source
            }
        )

    except Exception as exc:

        logger.warning(
            "Could not delete previous chunks: %s",
            exc,
        )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    logger.info(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = embeddings.tolist()

    # --------------------------------------------------------
    # IDS
    # --------------------------------------------------------

    file_hash = calculate_hash(path)

    ids = []

    documents = []

    metadatas = []

    source_hash = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{source_hash}_{index}"
        )

        ids.append(chunk_id)

        documents.append(chunk)

        metadatas.append(
            {
                "source": source,
                "filename": path.name,
                "chunk_id": str(index),
                "file_hash": file_hash,
            }
        )

    # --------------------------------------------------------
    # ADD TO CHROMA
    # --------------------------------------------------------

    logger.info(
        "Writing %d chunks to ChromaDB...",
        len(ids),
    )

    batch_size = 100

    for start in range(
        0,
        len(ids),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(ids),
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(
        "SUCCESS: %s",
        source,
    )

    return len(chunks)


# ============================================================
# RESET DATABASE
# ============================================================

def reset_database():

    if CHROMA_DIR.exists():

        logger.warning(
            "Deleting old ChromaDB..."
        )

        shutil.rmtree(
            CHROMA_DIR
        )

        logger.info(
            "Old ChromaDB deleted."
        )


# ============================================================
# MAIN INGESTION
# ============================================================

def ingest():

    logger.info("")
    logger.info(
        "======================================================"
    )
    logger.info(
        "          POLICY RAG INGESTION"
    )
    logger.info(
        "======================================================"
    )

    logger.info(
        "Project directory: %s",
        BASE_DIR,
    )

    logger.info(
        "Policies directory: %s",
        POLICIES_DIR,
    )

    logger.info(
        "ChromaDB directory: %s",
        CHROMA_DIR,
    )

    logger.info(
        "Collection: %s",
        COLLECTION_NAME,
    )

    logger.info(
        "Embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )

    # --------------------------------------------------------
    # FIND DOCUMENTS
    # --------------------------------------------------------

    files = find_policy_files()

    if not files:

        logger.error(
            ""
        )

        logger.error(
            "NO POLICY FILES FOUND!"
        )

        logger.error(
            "Put your PDF/TXT/MD/HTML files inside:"
        )

        logger.error(
            "%s",
            POLICIES_DIR,
        )

        sys.exit(1)

    logger.info(
        "Found %d policy documents.",
        len(files),
    )

    for file in files:

        logger.info(
            "  - %s",
            source_name(file),
        )

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    client = create_chroma_client()

    collection = create_collection(
        client
    )

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    logger.info(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    logger.info(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    total_chunks = 0

    successful = 0

    failed = 0

    for file in files:

        try:

            chunks = index_document(
                path=file,
                collection=collection,
                model=model,
            )

            total_chunks += chunks

            successful += 1

        except Exception as exc:

            failed += 1

            logger.exception(
                "FAILED: %s",
                file,
            )

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    try:

        final_count = collection.count()

    except Exception as exc:

        logger.error(
            "Could not verify ChromaDB: %s",
            exc,
        )

        sys.exit(1)

    logger.info("")
    logger.info(
        "======================================================"
    )

    logger.info(
        "              INGESTION FINISHED"
    )

    logger.info(
        "======================================================"
    )

    logger.info(
        "Documents found: %d",
        len(files),
    )

    logger.info(
        "Documents processed: %d",
        successful,
    )

    logger.info(
        "Documents failed: %d",
        failed,
    )

    logger.info(
        "Chunks generated: %d",
        total_chunks,
    )

    logger.info(
        "TOTAL CHUNKS IN CHROMADB: %d",
        final_count,
    )

    logger.info(
        "Database: %s",
        CHROMA_DIR,
    )

    logger.info(
        "Collection: %s",
        COLLECTION_NAME,
    )

    logger.info(
        "======================================================"
    )

    # --------------------------------------------------------
    # CRITICAL VALIDATION
    # --------------------------------------------------------

    if final_count == 0:

        logger.error(
            "ERROR: ChromaDB contains ZERO chunks."
        )

        logger.error(
            "The Streamlit application will not be able to answer questions."
        )

        sys.exit(1)

    logger.info(
        "SUCCESS: Policy RAG database is ready."
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB before ingestion.",
    )

    args = parser.parse_args()

    if args.reset:

        reset_database()

    ingest()


if __name__ == "__main__":

    main()
