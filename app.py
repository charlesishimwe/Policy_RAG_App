import os
import time
import requests
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# Works both locally (.env) and on Streamlit Cloud (st.secrets)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") or st.secrets.get("OPENROUTER_API_KEY", "")

# ── Constants ──────────────────────────────────────────────────────────────────
OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL         = "mistralai/mistral-7b-instruct:free"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "policy_docs"
TOP_K             = 4
MAX_TOKENS        = 600

POLICY_KEYWORDS = [
    "policy", "pto", "vacation", "leave", "remote", "work from home",
    "expense", "reimbursement", "security", "password", "holiday",
    "benefit", "conduct", "dress code", "overtime", "performance",
    "travel", "attendance", "harassment", "disciplin", "onboard",
    "termination", "promotion", "salary", "pay", "bonus", "parental",
    "sick", "bereavement", "jury", "floating", "hybrid",
]

# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to vector database...")
def load_chroma_collection():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── Guardrail ──────────────────────────────────────────────────────────────────
def is_policy_question(question: str) -> bool:
    return any(kw in question.lower() for kw in POLICY_KEYWORDS)


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_chunks(question, collection, embed_model):
    vector = embed_model.encode(question).tolist()
    return collection.query(
        query_embeddings=[vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )


# ── Prompt builder ─────────────────────────────────────────────────────────────
def build_prompt(question, chunks, sources):
    context = "\n\n---\n\n".join(
        f"[{i+1}] Source: {m.get('source','Unknown')}\n{c}"
        for i, (c, m) in enumerate(zip(chunks, sources))
    )
    return f"""You are a helpful company policy assistant.
Answer ONLY using the policy documents provided below.

Rules:
1. Use ONLY the context below. Never use outside knowledge.
2. If the answer is not in the context, say: "I could not find information about this in our policy documents."
3. End your answer with a Sources line listing the document names used.
4. Keep answers under 200 words.

--- POLICY CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}
Answer:"""


# ── LLM call ───────────────────────────────────────────────────────────────────
def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://policy-rag-chat.streamlit.app",
        "X-Title": "Policy RAG Chat",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
    }
    t0 = time.time()
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    latency = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"].strip(), latency


# ── Full RAG pipeline ──────────────────────────────────────────────────────────
def answer_question(question, collection, embed_model):
    if not is_policy_question(question):
        return {
            "answer": "I can only answer questions about our company policies. "
                      "Try asking about PTO, remote work, expenses, security, holidays, etc.",
            "citations": [], "latency": 0.0, "error": None,
        }

    try:
        results = retrieve_chunks(question, collection, embed_model)
    except Exception as e:
        return {"answer": "Error searching documents. Please try again.",
                "citations": [], "latency": 0.0, "error": str(e)}

    chunks    = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas",  [[]])[0]

    if not chunks:
        return {
            "answer": "No relevant policy documents found. Make sure you have run ingest.py first.",
            "citations": [], "latency": 0.0, "error": None,
        }

    try:
        answer, latency = call_llm(build_prompt(question, chunks, metadatas))
    except Exception as e:
        return {"answer": "Could not generate an answer. Please try again.",
                "citations": [], "latency": 0.0, "error": str(e)}

    seen = set()
    citations = []
    for chunk, meta in zip(chunks, metadatas):
        src = meta.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            citations.append({"source": src, "snippet": chunk[:250].replace("\n", " ").strip() + "…"})

    return {"answer": answer, "citations": citations, "latency": latency, "error": None}


# ── Streamlit UI ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Policy RAG Chat", page_icon="📄", layout="centered")

    # ── Handle /health check via query params ──────────────────────────────────
    params = st.query_params
    if params.get("health") == "check":
        st.json({"status": "ok"})
        st.stop()

    st.title("📄 Policy RAG Chat App")
    st.caption("Ask anything about company policies below:")

    # API key warning
    if not OPENROUTER_API_KEY:
        st.error(
            "⚠️ OPENROUTER_API_KEY is not set.\n\n"
            "**Locally:** add it to a `.env` file.\n\n"
            "**Streamlit Cloud:** go to App Settings → Secrets and add:\n"
            "`OPENROUTER_API_KEY = 'sk-or-your-key-here'`"
        )

    embed_model = load_embedding_model()
    collection  = load_chroma_collection()

    doc_count = collection.count()
    if doc_count == 0:
        st.warning("⚠️ No documents in the database. Run `python ingest.py` first.")
    else:
        st.success(f"✅ {doc_count} policy chunks loaded and ready.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Sources"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c['source']}**")
                        st.caption(c["snippet"])
            if msg.get("latency"):
                st.caption(f"⏱ {msg['latency']:.2f}s")

    user_input = st.chat_input("Your question")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                result = answer_question(user_input, collection, embed_model)

            st.markdown(result["answer"])

            if result["error"]:
                st.error(f"Debug: {result['error']}")

            if result["citations"]:
                with st.expander("📎 Sources"):
                    for c in result["citations"]:
                        st.markdown(f"**{c['source']}**")
                        st.caption(c["snippet"])

            if result["latency"] > 0:
                st.caption(f"⏱ {result['latency']:.2f}s")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "latency": result["latency"],
        })


if __name__ == "__main__":
    main()
