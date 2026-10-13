
from pathlib import Path
import os
import re

import chromadb
from sentence_transformers import SentenceTransformer

# Configuration: keep these settings consistent with app.py
POLICIES_DIR = Path("policies")
DB_PATH = os.getenv("CHROMA_DB_PATH", "Vectordb")
COLLECTION_NAME = "company_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """Normalize whitespace and remove empty Markdown artifacts."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main() -> None:
    if not POLICIES_DIR.exists():
        raise FileNotFoundError(
            f"Missing {POLICIES_DIR}/. Create it and add Markdown policy files."
        )

    files = sorted(POLICIES_DIR.glob("*.md"))

    if not files:
        raise FileNotFoundError(
            f"No Markdown files found in {POLICIES_DIR}/."
        )

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)

    # Rebuild this project's collection on each ingestion run.
    # Other collections in the database are left untouched.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Synthetic company policy documents"},
    )

    documents = []
    metadatas = []
    ids = []

    for file_path in files:
        text = clean_text(file_path.read_text(encoding="utf-8"))
        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source": file_path.name,
                "title": file_path.stem.replace("_", " ").title(),
                "chunk_index": index,
            })
            ids.append(f"{file_path.stem}-{index}")

    if not documents:
        raise ValueError("No text chunks were created from the policy files.")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    batch_size = 100

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    print("\nIngestion completed successfully.")
    print(f"Policy files: {len(files)}")
    print(f"Text chunks: {len(documents)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database path: {DB_PATH}")


if __name__ == "__main__":
    main()
