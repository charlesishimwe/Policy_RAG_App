"""
Policy RAG Copilot
Document Ingestion and ChromaDB Indexing

This script:
1. Reads policy documents from ./policies
2. Extracts text from PDF, TXT and Markdown files
3. Cleans the extracted text
4. Splits documents into overlapping chunks
5. Generates embeddings using SentenceTransformers
6. Stores chunks + embeddings + metadata in ChromaDB

The configuration MUST match app.py:
    CHROMA_PATH = "chroma_db"
    COLLECTION_NAME = "policy_docs"
    EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
"""

from pathlib import Path
import hashlib
import re
import sys

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "policies"

CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "policy_docs"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Batch size for ChromaDB
BATCH_SIZE = 64

# Supported file types
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
}


# ============================================================
# LOGGING HELPERS
# ============================================================

def print_header():
    print()
    print("=" * 70)
    print("POLICY RAG COPILOT - DOCUMENT INGESTION")
    print("=" * 70)
    print()
    print(f"Documents : {DATA_PATH}")
    print(f"ChromaDB  : {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding : {EMBED_MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap   : {CHUNK_OVERLAP}")
    print()
    print("=" * 70)
    print()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving useful
    paragraph and sentence structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", " ")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before/after newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.
    """

    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. "
            "Install it with: pip install pypdf"
        )

    text_parts = []

    try:
        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:
                page_text = page.extract_text()

            except Exception as exc:
                print(
                    f"  Warning: could not read page "
                    f"{page_number}: {exc}"
                )
                continue

            if page_text:
                text_parts.append(page_text)

    except Exception as exc:

        raise RuntimeError(
            f"Failed to read PDF '{file_path.name}': {exc}"
        ) from exc

    return "\n\n".join(text_parts)


# ============================================================
# TEXT / MARKDOWN EXTRACTION
# ============================================================

def extract_text_file(file_path: Path) -> str:
    """
    Read TXT or Markdown files.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ]

    last_error = None

    for encoding in encodings:

        try:

            return file_path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError as exc:

            last_error = exc

    raise RuntimeError(
        f"Could not decode '{file_path.name}': "
        f"{last_error}"
    )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_document(file_path: Path) -> str:
    """
    Extract text based on file extension.
    """

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        return extract_pdf(file_path)

    if extension in {".txt", ".md"}:

        return extract_text_file(file_path)

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
):
    """
    Split text into overlapping chunks.

    The implementation tries to split at natural boundaries
    such as paragraphs, sentences and spaces.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        # If this is not the final chunk,
        # try to find a natural break.
        if end < text_length:

            search_start = start + int(
                chunk_size * 0.60
            )

            candidate_breaks = [
                text.rfind("\n\n", search_start, end),
                text.rfind(". ", search_start, end),
                text.rfind("? ", search_start, end),
                text.rfind("! ", search_start, end),
                text.rfind("; ", search_start, end),
                text.rfind(", ", search_start, end),
                text.rfind(" ", search_start, end),
            ]

            valid_breaks = [
                position
                for position in candidate_breaks
                if position > start
            ]

            if valid_breaks:

                end = max(valid_breaks)

                # Keep punctuation when possible.
                if text[end:end + 2] in {
                    ". ",
                    "? ",
                    "! ",
                    "; ",
                }:

                    end += 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a source file.

    This is used to create deterministic document IDs.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# CHUNK ID
# ============================================================

def create_chunk_id(
    source_name: str,
    file_hash: str,
    chunk_index: int,
) -> str:
    """
    Create deterministic unique IDs.

    Example:
        employee_policy_abc123_chunk_0001
    """

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        source_name,
    )

    return (
        f"{safe_name}_"
        f"{file_hash[:12]}_"
        f"chunk_{chunk_index:04d}"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Load the local free SentenceTransformer model.
    """

    print(
        f"Loading embedding model: "
        f"{EMBED_MODEL_NAME}"
    )

    model = SentenceTransformer(
        EMBED_MODEL_NAME
    )

    print("Embedding model loaded.")
    print()

    return model


# ============================================================
# FIND DOCUMENTS
# ============================================================

def find_documents():
    """
    Find all supported policy documents.
    """

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Policy directory does not exist: "
            f"{DATA_PATH}\n\n"
            f"Create the directory and add your policy files."
        )

    files = []

    for path in DATA_PATH.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:

            files.append(path)

    files.sort(
        key=lambda path: str(path).lower()
    )

    return files


# ============================================================
# CHROMA CLIENT
# ============================================================

def create_chroma_client():
    """
    Create the ChromaDB persistent client.

    IMPORTANT:
    Do NOT use chromadb.config.Settings here.

    This exact configuration matches app.py and prevents
    the "different settings" Chroma error.
    """

    CHROMA_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    return client


# ============================================================
# GET COLLECTION
# ============================================================

def get_collection(client):
    """
    Get or create the policy collection.
    """

    try:

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Using existing collection: "
            f"{COLLECTION_NAME}"
        )

    except Exception:

        collection = client.create_collection(
            name=COLLECTION_NAME
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    return collection


# ============================================================
# PREPARE DOCUMENTS
# ============================================================

def prepare_documents(files):
    """
    Extract and chunk all policy documents.
    """

    all_documents = []
    all_metadatas = []
    all_ids = []

    successful_files = 0
    failed_files = 0

    print(
        f"Found {len(files)} supported document(s)."
    )
    print()

    for file_number, file_path in enumerate(
        files,
        start=1,
    ):

        relative_path = file_path.relative_to(
            DATA_PATH
        )

        print(
            f"[{file_number}/{len(files)}] "
            f"Processing: {relative_path}"
        )

        try:

            raw_text = extract_document(
                file_path
            )

            text = clean_text(
                raw_text
            )

            if not text:

                print(
                    "  WARNING: No text extracted. "
                    "Skipping."
                )

                failed_files += 1
                continue

            file_hash = calculate_file_hash(
                file_path
            )

            chunks = split_text(
                text
            )

            if not chunks:

                print(
                    "  WARNING: No chunks generated. "
                    "Skipping."
                )

                failed_files += 1
                continue

            for chunk_index, chunk in enumerate(
                chunks
            ):

                chunk_id = create_chunk_id(
                    file_path.stem,
                    file_hash,
                    chunk_index,
                )

                all_ids.append(
                    chunk_id
                )

                all_documents.append(
                    chunk
                )

                all_metadatas.append(
                    {
                        "source": file_path.name,
                        "source_path": str(
                            relative_path
                        ),
                        "chunk_id": chunk_index,
                        "file_hash": file_hash,
                        "file_type": (
                            file_path.suffix.lower()
                        ),
                    }
                )

            print(
                f"  Extracted characters: "
                f"{len(text):,}"
            )

            print(
                f"  Generated chunks: "
                f"{len(chunks)}"
            )

            successful_files += 1

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            failed_files += 1

    print()
    print(
        f"Successfully processed: "
        f"{successful_files}"
    )

    print(
        f"Failed/skipped: "
        f"{failed_files}"
    )

    print(
        f"Total chunks: "
        f"{len(all_documents)}"
    )

    print()

    return (
        all_documents,
        all_metadatas,
        all_ids,
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(
    model,
    documents,
):
    """
    Generate normalized embeddings for all chunks.
    """

    if not documents:
        return []

    print(
        f"Generating embeddings for "
        f"{len(documents)} chunks..."
    )

    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.tolist()

    print(
        "Embeddings generated successfully."
    )

    print()

    return embeddings


# ============================================================
# INDEX CHUNKS
# ============================================================

def index_chunks(
    collection,
    documents,
    metadatas,
    ids,
    embeddings,
):
    """
    Store documents, metadata and embeddings in ChromaDB.

    Existing IDs are updated/replaced using upsert, which makes
    the ingestion process safe to run multiple times.
    """

    if not documents:

        print(
            "No documents to index."
        )

        return

    total = len(documents)

    print(
        f"Indexing {total} chunks into ChromaDB..."
    )

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

        print(
            f"  Indexed {end}/{total} chunks"
        )

    print()
    print(
        "All chunks indexed successfully."
    )
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if CHUNK_OVERLAP >= CHUNK_SIZE:

        print(
            "ERROR: CHUNK_OVERLAP must be "
            "smaller than CHUNK_SIZE."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Find policy documents
    # --------------------------------------------------------

    try:

        files = find_documents()

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        sys.exit(1)

    if not files:

        print(
            "No supported policy documents found."
        )

        print()
        print(
            "Add PDF, TXT or MD files to:"
        )

        print(
            f"  {DATA_PATH}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    try:

        model = load_embedding_model()

    except Exception as exc:

        print(
            "ERROR loading embedding model:"
        )

        print(exc)

        sys.exit(1)

    # --------------------------------------------------------
    # Prepare documents
    # --------------------------------------------------------

    (
        documents,
        metadatas,
        ids,
    ) = prepare_documents(files)

    if not documents:

        print(
            "ERROR: No usable document chunks "
            "were generated."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    try:

        embeddings = generate_embeddings(
            model,
            documents,
        )

    except Exception as exc:

        print(
            "ERROR generating embeddings:"
        )

        print(exc)

        sys.exit(1)

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    try:

        client = create_chroma_client()

        collection = get_collection(
            client
        )

    except Exception as exc:

        print(
            "ERROR connecting to ChromaDB:"
        )

        print(exc)

        print()
        print(
            "If this is an old/incompatible "
            "chroma_db directory, stop Streamlit "
            "and rebuild the database."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    try:

        index_chunks(
            collection=collection,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    except Exception as exc:

        print(
            "ERROR indexing documents:"
        )

        print(exc)

        sys.exit(1)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    try:

        final_count = collection.count()

    except Exception as exc:

        print(
            "WARNING: Could not verify "
            f"collection count: {exc}"
        )

        final_count = None

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Database : {CHROMA_PATH}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Embedding : {EMBED_MODEL_NAME}"
    )

    print(
        f"Chunks processed: {len(documents)}"
    )

    if final_count is not None:

        print(
            f"Chunks in database: {final_count}"
        )

    print()
    print(
        "You can now start the application with:"
    )

    print()
    print(
        "    streamlit run app.py"
    )

    print()
    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
