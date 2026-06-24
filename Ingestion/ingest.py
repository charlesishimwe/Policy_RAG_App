import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 500
OVERLAP = 50

# simple embedding model (FREE + FAST)
model = SentenceTransformer("all-MiniLM-L6-v2")

# create chroma DB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="policy_docs")


# -----------------------------
# 1. READ PDF FILES
# -----------------------------
def load_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            docs.append({
                "text": text,
                "source": file
            })
    return docs


# -----------------------------
# 2. SIMPLE CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap  # overlap

    return chunks


# -----------------------------
# 3. EMBED + STORE
# -----------------------------
def embed_and_store(docs):
    for doc in docs:
        chunks = chunk_text(doc["text"])

        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()

            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "source": doc["source"],
                    "chunk_id": i
                }],
                ids=[f"{doc['source']}_{i}"]
            )

    print("✅ Ingestion complete. Data stored in ChromaDB!")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_pdfs(DATA_PATH)

    print(f"Loaded {len(docs)} documents")

    print("Chunking + embedding + storing...")
    embed_and_store(docs)
