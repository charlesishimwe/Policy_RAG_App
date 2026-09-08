"""
POLICY RAG APP - ROBUST INGESTION ENGINE

Features
--------
- PDF / TXT / MD / HTML support
- Automatic text extraction
- Text cleaning
- Overlapping chunks
- Sentence Transformer embeddings
- Persistent ChromaDB
- Automatic recovery from broken ChromaDB
- Automatic recovery from empty collections
- Duplicate-safe indexing
- Changed-file detection
- Deleted-file cleanup
- Continuous watch mode
- Strong validation
- Same database path expected by app.py

Commands
--------
Build/rebuild:
    python ingest.py --reset

Normal:
    python ingest.py

Continuous 24/7:
    python ingest.py --watch

Continuous every 60 seconds:
    python ingest.py --watch --interval 60
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List

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
# app.py MUST use exactly this same directory.
CHROMA_DIR = BASE_DIR / "chroma_db"

# IMPORTANT:
# app.py MUST use exactly this same collection.
COLLECTION_NAME = "policy_docs"

# IMPORTANT:
# app.py MUST use exactly this same model.
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

BATCH_SIZE = 100

DEFAULT_INTERVAL = 300


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("policy-rag-ingestion")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\x00", " ")

    cleaned_lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(path: Path) -> str:

    logger.info("Reading PDF: %s", path.name)

    try:

        reader = PdfReader(str(path))

    except Exception as exc:

        raise RuntimeError(
            f"Unable to open PDF {path.name}: {exc}"
        ) from exc

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        try:

            text = page.extract_text() or ""

            text = text.strip()

            if text:

                pages.append(
                    f"[Page {page_number}]\n{text}"
                )

        except Exception as exc:

            logger.warning(
                "Page %d failed in %s: %s",
                page_number,
                path.name,
                exc,
            )

    return "\n\n".join(pages)


# ============================================================
# TEXT EXTRACTION
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
        f"Unable to decode {path.name}"
    )


# ============================================================
# HTML EXTRACTION
# ============================================================

def extract_html(path: Path) -> str:

    raw_html = extract_text_file(path)

    soup = BeautifulSoup(
        raw_html,
        "html.parser",
    )

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "svg",
        ]
    ):

        tag.decompose()

    return soup.get_text(
        separator="\n"
    )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_document(path: Path) -> str:

    extension = path.suffix.lower()

    if extension == ".pdf":

        return extract_pdf(path)

    if extension in {".txt", ".md"}:

        return extract_text_file(path)

    if extension in {".html", ".htm"}:

        return extract_html(path)

    raise RuntimeError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text: str,
) -> List[str]:

    if not text:

        return []

    if CHUNK_OVERLAP >= CHUNK_SIZE:

        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(
            start + CHUNK_SIZE,
            text_length,
        )

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# FIND POLICY FILES
# ============================================================

def find_policy_files() -> List[Path]:

    POLICIES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = []

    for path in POLICIES_DIR.rglob("*"):

        if not path.is_file():

            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:

            continue

        files.append(path)

    return sorted(files)


# ============================================================
# SOURCE NAME
# ============================================================

def get_source(path: Path) -> str:

    return str(
        path.relative_to(BASE_DIR)
    ).replace("\\", "/")


# ============================================================
# FILE HASH
# ============================================================

def calculate_hash(path: Path) -> str:

    sha256 = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            data = file.read(
                1024 * 1024
            )

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
        "ChromaDB path: %s",
        CHROMA_DIR,
    )

    # IMPORTANT:
    #
    # Do NOT use Settings(...)
    #
    # This avoids the previous:
    #
    # KeyError: '_type'
    #

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# ============================================================
# COLLECTION
# ============================================================

def get_or_create_collection(client):

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        logger.info(
            "Collection loaded: %s",
            COLLECTION_NAME,
        )

        return collection

    except Exception as exc:

        logger.warning(
            "Collection unavailable: %s",
            exc,
        )

        logger.info(
            "Creating collection: %s",
            COLLECTION_NAME,
        )

        try:

            return client.create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                },
            )

        except Exception as create_error:

            raise RuntimeError(
                "Could not create ChromaDB collection. "
                f"Original error: {exc}. "
                f"Create error: {create_error}"
            ) from create_error


# ============================================================
# RESET CHROMA
# ============================================================

def reset_chroma():

    if not CHROMA_DIR.exists():

        logger.info(
            "No existing ChromaDB to reset."
        )

        return

    logger.warning(
        "Removing existing ChromaDB..."
    )

    try:

        shutil.rmtree(
            CHROMA_DIR
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not delete {CHROMA_DIR}: {exc}"
        ) from exc

    logger.info(
        "Old ChromaDB removed."
    )


# ============================================================
# REBUILD BROKEN DATABASE
# ============================================================

def create_fresh_database():

    logger.warning(
        "Creating a fresh ChromaDB."
    )

    reset_chroma()

    client = create_chroma_client()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    return client, collection


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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
# DELETE SOURCE
# ============================================================

def delete_source(
    collection,
    source: str,
):

    try:

        collection.delete(
            where={
                "source": source
            }
        )

        logger.info(
            "Removed previous chunks: %s",
            source,
        )

    except Exception as exc:

        logger.warning(
            "Could not remove previous chunks for %s: %s",
            source,
            exc,
        )


# ============================================================
# GET EXISTING SOURCE HASHES
# ============================================================

def get_indexed_sources(collection):

    sources = {}

    try:

        result = collection.get(
            include=["metadatas"]
        )

    except Exception as exc:

        logger.warning(
            "Could not inspect existing index: %s",
            exc,
        )

        return sources

    metadatas = result.get(
        "metadatas",
        []
    )

    for metadata in metadatas:

        if not metadata:

            continue

        source = metadata.get(
            "source"
        )

        file_hash = metadata.get(
            "file_hash"
        )

        if source:

            sources[source] = file_hash

    return sources


# ============================================================
# INDEX DOCUMENT
# ============================================================

def index_document(
    path: Path,
    collection,
    model,
):

    source = get_source(path)

    logger.info(
        ""
    )

    logger.info(
        "Processing: %s",
        source,
    )

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    file_hash = calculate_hash(
        path
    )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    raw_text = extract_document(
        path
    )

    text = clean_text(
        raw_text
    )

    if not text:

        logger.warning(
            "No text extracted: %s",
            source,
        )

        return 0

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = create_chunks(
        text
    )

    if not chunks:

        logger.warning(
            "No chunks generated: %s",
            source,
        )

        return 0

    logger.info(
        "Chunks generated: %d",
        len(chunks),
    )

    # --------------------------------------------------------
    # REMOVE OLD VERSION
    # --------------------------------------------------------

    delete_source(
        collection,
        source,
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    logger.info(
        "Creating embeddings..."
    )

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = embeddings.tolist()

    # --------------------------------------------------------
    # IDS / METADATA
    # --------------------------------------------------------

    source_id = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:20]

    ids = []

    documents = []

    metadatas = []

    for index, chunk in enumerate(
        chunks
    ):

        ids.append(
            f"{source_id}_{index}"
        )

        documents.append(
            chunk
        )

        metadatas.append(
            {
                "source": source,
                "filename": path.name,
                "chunk_id": str(index),
                "file_hash": file_hash,
            }
        )

    # --------------------------------------------------------
    # WRITE TO CHROMA
    # --------------------------------------------------------

    for start in range(
        0,
        len(ids),
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            len(ids),
        )

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    logger.info(
        "INDEXED: %s | %d chunks",
        source,
        len(chunks),
    )

    return len(chunks)


# ============================================================
# REMOVE DELETED FILES
# ============================================================

def remove_deleted_files(
    collection,
    policy_files,
):

    current_sources = {
        get_source(path)
        for path in policy_files
    }

    indexed_sources = get_indexed_sources(
        collection
    )

    for source in indexed_sources:

        if source not in current_sources:

            logger.info(
                "Removing deleted policy: %s",
                source,
            )

            delete_source(
                collection,
                source,
            )


# ============================================================
# INGESTION
# ============================================================

def run_ingestion(
    model,
    force_reset=False,
):

    logger.info(
        ""
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "POLICY RAG INGESTION START"
    )

    logger.info(
        "============================================================"
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    policy_files = find_policy_files()

    logger.info(
        "Policy files found: %d",
        len(policy_files),
    )

    if not policy_files:

        logger.error(
            "NO POLICY DOCUMENTS FOUND."
        )

        logger.error(
            "Add PDF/TXT/MD/HTML files to:"
        )

        logger.error(
            "%s",
            POLICIES_DIR,
        )

        return False

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    if force_reset:

        reset_chroma()

        client = create_chroma_client()

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            },
        )

    else:

        try:

            client = create_chroma_client()

            collection = get_or_create_collection(
                client
            )

        except Exception as exc:

            logger.error(
                "Existing ChromaDB is unusable."
            )

            logger.error(
                "%s",
                exc,
            )

            logger.warning(
                "Automatically rebuilding ChromaDB..."
            )

            client, collection = (
                create_fresh_database()
            )

    # --------------------------------------------------------
    # REMOVE DELETED
    # --------------------------------------------------------

    remove_deleted_files(
        collection,
        policy_files,
    )

    # --------------------------------------------------------
    # EXISTING INDEX
    # --------------------------------------------------------

    existing_sources = get_indexed_sources(
        collection
    )

    total_chunks = 0

    processed = 0

    skipped = 0

    failed = 0

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for path in policy_files:

        source = get_source(
            path
        )

        try:

            current_hash = calculate_hash(
                path
            )

            existing_hash = (
                existing_sources.get(
                    source
                )
            )

            # ------------------------------------------------
            # SKIP UNCHANGED
            # ------------------------------------------------

            if (
                existing_hash
                and existing_hash == current_hash
            ):

                logger.info(
                    "SKIPPED unchanged: %s",
                    source,
                )

                skipped += 1

                continue

            # ------------------------------------------------
            # INDEX
            # ------------------------------------------------

            chunks = index_document(
                path,
                collection,
                model,
            )

            if chunks > 0:

                total_chunks += chunks

                processed += 1

            else:

                failed += 1

        except Exception as exc:

            failed += 1

            logger.exception(
                "FAILED: %s",
                source,
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    try:

        final_count = collection.count()

    except Exception as exc:

        logger.error(
            "Unable to read ChromaDB count: %s",
            exc,
        )

        return False

    logger.info(
        ""
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "INGESTION RESULT"
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "Policy files:       %d",
        len(policy_files),
    )

    logger.info(
        "Indexed:            %d",
        processed,
    )

    logger.info(
        "Skipped:            %d",
        skipped,
    )

    logger.info(
        "Failed:             %d",
        failed,
    )

    logger.info(
        "New chunks:         %d",
        total_chunks,
    )

    logger.info(
        "TOTAL CHROMA CHUNKS: %d",
        final_count,
    )

    logger.info(
        "Database:           %s",
        CHROMA_DIR,
    )

    logger.info(
        "Collection:         %s",
        COLLECTION_NAME,
    )

    logger.info(
        "============================================================"
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if final_count <= 0:

        logger.error(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        logger.error(
            "ZERO POLICY CHUNKS INDEXED."
        )

        logger.error(
            "Check that your PDFs contain selectable text."
        )

        logger.error(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        return False

    logger.info(
        "SUCCESS: POLICY DATABASE READY."
    )

    return True


# ============================================================
# CONTINUOUS MODE
# ============================================================

def watch_mode(
    interval: int,
    reset: bool = False,
):

    logger.info(
        ""
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "POLICY RAG 24/7 WATCH MODE"
    )

    logger.info(
        "============================================================"
    )

    logger.info(
        "Checking every %d seconds.",
        interval,
    )

    logger.info(
        "Press CTRL+C to stop."
    )

    model = None

    first_run = True

    while True:

        try:

            # Load model once.
            if model is None:

                model = load_model()

            success = run_ingestion(
                model=model,
                force_reset=(
                    reset
                    if first_run
                    else False
                ),
            )

            first_run = False

            if success:

                logger.info(
                    "Next check in %d seconds.",
                    interval,
                )

            else:

                logger.warning(
                    "Ingestion was not successful."
                )

                logger.info(
                    "Will retry in %d seconds.",
                    interval,
                )

            time.sleep(
                interval
            )

        except KeyboardInterrupt:

            logger.info(
                "24/7 ingestion stopped."
            )

            break

        except Exception as exc:

            logger.exception(
                "Unexpected ingestion error: %s",
                exc,
            )

            logger.info(
                "Retrying in %d seconds.",
                interval,
            )

            time.sleep(
                interval
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Policy RAG ingestion engine"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild ChromaDB.",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously.",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Watch interval in seconds.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    if args.watch:

        if args.interval < 30:

            logger.warning(
                "Minimum interval is 30 seconds."
            )

            args.interval = 30

        watch_mode(
            interval=args.interval,
            reset=args.reset,
        )

        return

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if args.reset:

        reset_chroma()

    model = load_model()

    success = run_ingestion(
        model=model,
        force_reset=False,
    )

    if not success:

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
