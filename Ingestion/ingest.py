import os
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
POLICIES_DIR      = "./policies"
CHROMA_COLLECTION = "policy_docs"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50


# ── Text extraction (no pypdf, no beautifulsoup) ───────────────────────────────
def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext in (".html", ".htm"):
        # Uses Python built-in html.parser — no extra install needed
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.texts = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self.texts.append(stripped)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        parser = TextExtractor()
        parser.feed(raw)
        return "\n".join(parser.texts)

    elif ext == ".pdf":
        print(f"  ⚠️  Skipping {os.path.basename(filepath)} — PDF not supported.")
        print("     Rename it to .txt or .md to include it.")
        return ""

    else:
        print(f"  Skipping unsupported file: {os.path.basename(filepath)}")
        return ""


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────
def ingest():
    if not os.path.exists(POLICIES_DIR):
        print(f"ERROR: '{POLICIES_DIR}' folder not found.")
        print("Create a ./policies/ folder and add your .txt or .md files.")
        return

    files = [
        f for f in os.listdir(POLICIES_DIR)
        if os.path.isfile(os.path.join(POLICIES_DIR, f))
        and not f.startswith(".")
    ]

    if not files:
        print(f"No files found in {POLICIES_DIR}.")
        return

    print(f"Found {len(files)} file(s)\n")

    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )

    # Clear old data and start fresh
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("Cleared old collection.\n")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for filename in sorted(files):
        filepath = os.path.join(POLICIES_DIR, filename)
        print(f"Processing: {filename}")

        text = extract_text(filepath)
        if not text.strip():
            print(f"  No text found, skipping.\n")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"  No chunks created, skipping.\n")
            continue

        print(f"  → {len(chunks)} chunks created")

        embeddings = embed_model.encode(chunks, show_progress_bar=False).tolist()

        collection.add(
            ids       = [f"{filename}_chunk_{i}" for i in range(len(chunks))],
            documents = chunks,
            embeddings= embeddings,
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))],
        )

        total_chunks += len(chunks)
        print(f"  ✅ Stored in ChromaDB\n")

    print(f"Done! {total_chunks} total chunks stored in ChromaDB.")
    print("Now run:  streamlit run app.py")


if __name__ == "__main__":
    ingest()
