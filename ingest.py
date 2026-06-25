import os
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# =========================
# CONFIG
# =========================
DATA_PATH = "policies"   # folder where your policy files live
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "policy_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================
# LOAD EMBEDDING MODEL
# =========================
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# =========================
# INIT CHROMA DB
# =========================
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# =========================
# FILE READERS
# =========================
def read_pdf(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_md(file_path):
    return read_txt(file_path)


def load_file(file_path):
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    elif ext == ".md":
        return read_md(file_path)
    else:
        return ""


# =========================
# TEXT CLEANING
# =========================
def clean_text(text: str) -> str:
    return " ".join(text.split())


# =========================
# CHUNKING FUNCTION
# =========================
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# =========================
# INGEST FUNCTION
# =========================
def ingest():
    files = list(Path(DATA_PATH).rglob("*"))

    print(f"Found {len(files)} files")

    for file_path in files:
        if file_path.is_dir():
            continue

        print(f"Ingesting: {file_path.name}")

        text = load_file(file_path)

        if not text or len(text.strip()) < 10:
            print(f"Skipping empty file: {file_path.name}")
            continue

        text = clean_text(text)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) == 0:
                continue

            embedding = model.encode(chunk).tolist()

            doc_id = str(uuid.uuid4())

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": file_path.name,
                    "chunk_id": i
                }]
            )

        print(f"✔ Done: {file_path.name} -> {len(chunks)} chunks")

    print("✅ Ingestion completed successfully!")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    ingest()
