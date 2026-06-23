import streamlit as st

# =========================
# MOCK RAG FUNCTION (replace later with your real pipeline)
# =========================
def get_rag_response(query: str):
    """
    This is a placeholder for your RAG system.
    Later you will connect:
    - vector DB (Chroma)
    - embeddings
    - retrieval
    - LLM generation
    """

    if not query:
        return {
            "answer": "Please enter a question about the policies.",
            "citations": []
        }

    # Simple fake response for now (no errors, safe default)
    return {
        "answer": f"This is a demo answer for: {query}",
        "citations": [
            {
                "source": "policy_document_1.pdf",
                "snippet": "Example policy snippet related to your question."
            }
        ]
    }


# =========================
# HEALTH CHECK (Step requirement)
# =========================
def health():
    return {"status": "ok"}


# =========================
# STREAMLIT UI (/ route)
# =========================
st.set_page_config(page_title="Policy RAG App", page_icon="📄")

st.title("📄 Policy RAG Chat App")

st.write("Ask anything about company policies below:")

# Chat input box
user_input = st.text_input("Your question")

# Button to submit
if st.button("Ask"):

    response = get_rag_response(user_input)

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Citations")

    if response["citations"]:
        for c in response["citations"]:
            st.markdown(f"- **Source:** {c['source']}")
            st.markdown(f"  - Snippet: {c['snippet']}")
    else:
        st.write("No citations found.")


# =========================
# OPTIONAL DEBUG (not Flask)
# =========================
if st.checkbox("Check system health"):
    st.json(health())
