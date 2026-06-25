"""
ingest.py — Load your policy documents into ChromaDB

Supports: .txt, .md, .pdf, .html files
Place your policy files in the ./policies/ folder, then run:
  python ingest.py

Run this ONCE before starting app.py (or re-run to refresh the database).
"""

import os
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
POLICIES_DIR      = "./policies"       # folder containing your policy docs
CHROMA_COLLECTION = "policy_docs"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
CHUNK_SIZE        = 500                # characters per chunk
CHUNK_OVERLAP     = 50                 # overlap between chunks


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print("  pypdf not installed. Run: pip install pypdf")
            return ""

    elif ext in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n")
        except ImportError:
            print("  beautifulsoup4 not installed. Run: pip install beautifulsoup4")
            return ""

    else:
        print(f"  Skipping unsupported file type: {ext}")
        return ""


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ── Main ingestion ─────────────────────────────────────────────────────────────
def ingest():
    if not os.path.exists(POLICIES_DIR):
        print(f"ERROR: Policies folder not found: {POLICIES_DIR}")
        print("Create a ./policies/ folder and add your .txt / .md / .pdf files.")
        return

    policy_files = [
        f for f in os.listdir(POLICIES_DIR)
        if os.path.isfile(os.path.join(POLICIES_DIR, f))
        and not f.startswith(".")
    ]

    if not policy_files:
        print(f"No files found in {POLICIES_DIR}. Add your policy documents and try again.")
        return

    print(f"Found {len(policy_files)} file(s) in {POLICIES_DIR}")

    # Load embedding model
    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete old collection to start fresh
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("Cleared existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for filename in policy_files:
        filepath = os.path.join(POLICIES_DIR, filename)
        print(f"\nProcessing: {filename}")

        text = extract_text(filepath)
        if not text.strip():
            print(f"  No text extracted from {filename}, skipping.")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"  No chunks created from {filename}, skipping.")
            continue

        print(f"  → {len(chunks)} chunk(s) created")

        # Embed all chunks
        embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()

        # Store in ChromaDB
        ids       = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

    print(f"\n✅ Done! {total_chunks} total chunks stored in ChromaDB.")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    ingest()
