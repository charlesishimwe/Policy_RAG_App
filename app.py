import os
import traceback
import streamlit as st

# Safe imports (prevents full app crash)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
except Exception:
    chromadb = None


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Policy RAG Assistant", layout="wide")

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================
# SAFE UTILITIES
# =========================
def safe_read_pdf(file_path):
    """Read PDF safely without crashing app."""
    if PdfReader is None:
        return []

    try:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            try:
                text = page.extract_text()
                if text:
                    pages.append(text)
            except Exception:
                continue
        return pages
    except Exception as e:
        st.warning(f"PDF read error: {str(e)}")
        return []


def safe_chunk(text, chunk_size=500, overlap=100):
    """Robust text chunking."""
    try:
        chunks = []
        text = text.replace("\n", " ")
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return [c for c in chunks if c.strip()]
    except Exception:
        return [text[:500]] if text else []


# =========================
# INIT EMBEDDINGS MODEL
# =========================
@st.cache_resource
def load_embedder():
    try:
        if SentenceTransformer is None:
            return None
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        return None


# =========================
# INIT CHROMA DB
# =========================
@st.cache_resource
def load_vector_db():
    if chromadb is None:
        return None

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection("policies")
        return collection
    except Exception:
        return None


# =========================
# INGEST DOCUMENTS
# =========================
def ingest_documents(embedder, db):
    """Load all docs safely into vector DB."""
    if embedder is None or db is None:
        return

    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
            return

        doc_id = 0

        for file in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, file)

            if file.endswith(".pdf"):
                pages = safe_read_pdf(file_path)
                content_list = pages
            else:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content_list = [f.read()]
                except Exception:
                    continue

            for content in content_list:
                chunks = safe_chunk(content)

                for chunk in chunks:
                    try:
                        emb = embedder.encode(chunk).tolist()

                        db.add(
                            documents=[chunk],
                            embeddings=[emb],
                            ids=[f"doc_{doc_id}"],
                            metadatas=[{"source": file}]
                        )
                        doc_id += 1
                    except Exception:
                        continue

    except Exception as e:
        st.error(f"Ingestion error: {str(e)}")


# =========================
# RETRIEVAL
# =========================
def retrieve(query, embedder, db, k=3):
    if embedder is None or db is None:
        return []

    try:
        q_emb = embedder.encode(query).tolist()

        results = db.query(
            query_embeddings=[q_emb],
            n_results=k
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        return list(zip(docs, metas))

    except Exception:
        return []


# =========================
# GENERATION (SAFE MODE)
# =========================
def generate_answer(query, contexts):
    """Safe answer generator (no hallucination)."""

    try:
        if not contexts:
            return "I could not find relevant policy information in the documents."

        context_text = "\n\n".join(
            [f"[SOURCE: {m.get('source','unknown')}]\n{c}" for c, m in contexts]
        )

        prompt = f"""
You are a strict policy assistant.

RULES:
- Only use the provided context
- If answer is not in context say "Not found in policy documents"
- Always cite sources

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

        # OPTIONAL: If OpenAI key exists
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return resp.choices[0].message.content
            except Exception:
                pass

        # FALLBACK MODE (NO API)
        return f"""
⚠️ Running in fallback mode (no LLM API key).

Based on documents:

{context_text[:1500]}

QUESTION: {query}

NOTE: Install OpenAI key for better responses.
"""

    except Exception:
        return "Error generating response."


# =========================
# UI
# =========================
st.title("📄 Policy RAG Assistant (Zero-Crash Version)")

embedder = load_embedder()
db = load_vector_db()

# Ingest only once per session
if "ingested" not in st.session_state:
    ingest_documents(embedder, db)
    st.session_state.ingested = True


# Health status
st.sidebar.header("System Status")
st.sidebar.write("Embedder:", "OK" if embedder else "FAILED")
st.sidebar.write("Vector DB:", "OK" if db else "FAILED")
st.sidebar.write("Documents ingested:", st.session_state.get("ingested", False))


# Chat UI
query = st.text_input("Ask a policy question:")

if st.button("Ask"):
    try:
        with st.spinner("Thinking..."):
            contexts = retrieve(query, embedder, db)
            answer = generate_answer(query, contexts)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for doc, meta in contexts:
            st.markdown(f"**Source:** {meta.get('source','unknown')}")
            st.caption(doc[:300] + "...")

    except Exception as e:
        st.error("Something went wrong, but the app did NOT crash.")
        st.code(traceback.format_exc())


# Safe footer
st.markdown("---")
st.markdown("🔒 Built with crash-safe RAG architecture")
