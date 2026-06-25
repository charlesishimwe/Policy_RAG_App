"""
Policy RAG Chat App
Stack: Streamlit + OpenRouter (LLM) + ChromaDB (vector store) + sentence-transformers (embeddings)

Setup:
  pip install streamlit chromadb sentence-transformers requests python-dotenv

Set your API key in a .env file:
  OPENROUTER_API_KEY=sk-or-...

Run:
  streamlit run app.py
"""

import os
import time
import requests
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── Constants ─────────────────────────────────────────────────────────────────
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
# Free model on OpenRouter — swap for any other free model you prefer
LLM_MODEL        = "mistralai/mistral-7b-instruct:free"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"        # free, runs locally
CHROMA_COLLECTION = "policy_docs"
TOP_K             = 4                          # number of chunks to retrieve
MAX_TOKENS        = 600                        # max LLM output length

# Policy keywords — used by the guardrail to detect off-topic questions
POLICY_KEYWORDS = [
    "policy", "pto", "vacation", "leave", "remote", "work from home",
    "expense", "reimbursement", "security", "password", "holiday",
    "benefit", "conduct", "dress code", "overtime", "performance",
    "travel", "attendance", "harassment", "disciplin", "onboard",
    "termination", "promotion", "salary", "pay", "bonus",
]

# ── Cached resources (loaded once per session) ────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to vector database...")
def load_chroma_collection():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create so the app doesn't crash if the collection doesn't exist yet
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Guardrail: is the question about company policy? ──────────────────────────
def is_policy_question(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in POLICY_KEYWORDS)


# ── Retrieve relevant chunks from ChromaDB ────────────────────────────────────
def retrieve_chunks(question: str, collection, embed_model) -> dict:
    """Returns {'documents': [...], 'metadatas': [...], 'distances': [...]}"""
    query_vector = embed_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    return results


# ── Build the prompt ──────────────────────────────────────────────────────────
def build_prompt(question: str, chunks: list[str], sources: list[dict]) -> str:
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, sources), 1):
        source_name = meta.get("source", f"Document {i}")
        context_parts.append(f"[{i}] Source: {source_name}\n{chunk}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful company policy assistant. 
Your job is to answer employee questions accurately using ONLY the policy documents provided below.

Rules you must follow:
1. Answer ONLY from the context provided. Never use outside knowledge.
2. If the answer is not in the context, say exactly: "I could not find information about this in our policy documents."
3. Always end your answer with a "Sources:" line listing the document names you used.
4. Keep your answer concise and clear (under 200 words).
5. If multiple documents are relevant, mention all of them.

--- POLICY CONTEXT ---
{context}
--- END CONTEXT ---

Employee question: {question}

Answer:"""


# ── Call OpenRouter LLM ───────────────────────────────────────────────────────
def call_llm(prompt: str) -> tuple[str, float]:
    """Returns (answer_text, latency_seconds). Raises on API errors."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",   # required by OpenRouter
        "X-Title": "Policy RAG Chat",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,   # low temp = more factual, less creative
    }

    t0 = time.time()
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    latency = time.time() - t0

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    return answer, latency


# ── Full RAG pipeline ─────────────────────────────────────────────────────────
def answer_question(question: str, collection, embed_model) -> dict:
    """
    Returns:
      {
        "answer": str,
        "citations": [{"source": str, "snippet": str}],
        "latency": float,
        "num_chunks": int,
        "error": str | None,
      }
    """
    # Guardrail: off-topic check
    if not is_policy_question(question):
        return {
            "answer": "I can only answer questions about our company policies and procedures. "
                      "Please ask about topics like PTO, remote work, expenses, security, holidays, etc.",
            "citations": [],
            "latency": 0.0,
            "num_chunks": 0,
            "error": None,
        }

    # Retrieve
    try:
        results = retrieve_chunks(question, collection, embed_model)
    except Exception as e:
        return {
            "answer": "Sorry, I had trouble searching the policy documents. Please try again.",
            "citations": [],
            "latency": 0.0,
            "num_chunks": 0,
            "error": f"ChromaDB retrieval error: {e}",
        }

    chunks   = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not chunks:
        return {
            "answer": "I could not find any relevant policy documents for your question. "
                      "Please make sure your policy documents have been ingested into the database.",
            "citations": [],
            "latency": 0.0,
            "num_chunks": 0,
            "error": None,
        }

    # Generate
    prompt = build_prompt(question, chunks, metadatas)
    try:
        answer, latency = call_llm(prompt)
    except Exception as e:
        return {
            "answer": "Sorry, I could not generate an answer right now. Please try again in a moment.",
            "citations": [],
            "latency": 0.0,
            "num_chunks": len(chunks),
            "error": str(e),
        }

    # Build citations
    citations = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "Unknown document")
        snippet = chunk[:250].replace("\n", " ").strip()
        if snippet and source not in [c["source"] for c in citations]:
            citations.append({"source": source, "snippet": snippet + "…"})

    return {
        "answer": answer,
        "citations": citations,
        "latency": latency,
        "num_chunks": len(chunks),
        "error": None,
    }


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Policy RAG Chat",
        page_icon="📄",
        layout="centered",
    )

    st.title("📄 Policy RAG Chat App")
    st.caption("Ask anything about company policies below:")

    # Warn if API key is missing
    if not OPENROUTER_API_KEY:
        st.error(
            "⚠️ OPENROUTER_API_KEY is not set. "
            "Create a `.env` file in the same folder as app.py and add:\n\n"
            "`OPENROUTER_API_KEY=sk-or-your-key-here`"
        )

    # Load resources
    embed_model = load_embedding_model()
    collection  = load_chroma_collection()

    # Show how many chunks are indexed
    doc_count = collection.count()
    if doc_count == 0:
        st.warning(
            "⚠️ No documents found in the vector database. "
            "Run your ingestion script first to load your policy documents."
        )
    else:
        st.success(f"✅ {doc_count} policy chunks loaded and ready.")

    # Chat history stored in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Sources"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c['source']}**")
                        st.caption(c["snippet"])
            if msg.get("latency"):
                st.caption(f"⏱ Response time: {msg['latency']:.2f}s")

    # Chat input
    user_input = st.chat_input("Your question")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Searching policies and generating answer..."):
                result = answer_question(user_input, collection, embed_model)

            st.markdown(result["answer"])

            if result["error"]:
                st.error(f"Debug info: {result['error']}")

            if result["citations"]:
                with st.expander("📎 Sources"):
                    for c in result["citations"]:
                        st.markdown(f"**{c['source']}**")
                        st.caption(c["snippet"])

            if result["latency"] > 0:
                st.caption(f"⏱ Response time: {result['latency']:.2f}s")

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "latency": result["latency"],
        })


# ── Health check endpoint (for the /health requirement) ──────────────────────
# Streamlit doesn't have routes, so add a small Flask health server
# running on port 8502 in a background thread.
def start_health_server():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import threading

    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # suppress console noise

    server = HTTPServer(("0.0.0.0", 8502), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


if __name__ == "__main__":
    start_health_server()   # /health available at http://localhost:8502/health
    main()
