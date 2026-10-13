
"""
Acme Corp Policy Assistant — Streamlit Application

Features:
  - Interactive policy question answering
  - Retrieval-Augmented Generation (RAG)
  - Source and snippet display
  - Index health information
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag import answer_question, get_collection, get_embed_model


load_dotenv()

st.set_page_config(
    page_title="Acme Corp Policy Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Acme Corp Policy Assistant")
st.caption("Enterprise Policy Intelligence powered by RAG")


@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    return get_embed_model()


@st.cache_resource
def load_policy_collection():
    """Load and cache the ChromaDB collection."""
    return get_collection()


# Initialize RAG components
with st.spinner("Initializing policy search..."):
    try:
        load_embedding_model()
        collection = load_policy_collection()
        indexed_chunks = collection.count()
        index_ready = indexed_chunks > 0
        initialization_error = None
    except Exception as exc:
        indexed_chunks = 0
        index_ready = False
        initialization_error = str(exc)


# Sidebar: system status
with st.sidebar:
    st.header("System Status")

    if initialization_error:
        st.error("RAG initialization failed")
        st.caption(initialization_error)
    elif index_ready:
        st.success("Policy index ready")
    else:
        st.warning("Policy index is empty")

    st.metric("Indexed chunks", indexed_chunks)

    api_key_env = os.getenv(
        "LLM_API_KEY_ENV",
        "OPENROUTER_API_KEY",
    )
    llm_configured = bool(os.getenv(api_key_env, ""))

    st.write(
        "LLM API: Configured"
        if llm_configured
        else "LLM API: Not configured"
    )

    st.caption(
        "Embedding model: all-MiniLM-L6-v2"
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=int(os.getenv("TOP_K", "5"))
        if os.getenv("TOP_K", "5").isdigit()
        and 1 <= int(os.getenv("TOP_K", "5")) <= 10
        else 5,
    )


# Main question-answering interface
st.subheader("Ask a policy question")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What is our company's vacation policy?"
    ),
    max_chars=2000,
    height=120,
)

if st.button(
    "Ask Policy Assistant",
    type="primary",
    disabled=not question.strip() or not index_ready,
):
    with st.spinner("Searching policies and generating an answer..."):
        try:
            result = answer_question(
                question.strip(),
                k=top_k,
            )

            st.session_state["last_result"] = result
            st.session_state["last_question"] = question.strip()

        except Exception:
            st.error(
                "An error occurred while processing your question. "
                "Check your RAG configuration and application logs."
            )
            st.exception(Exception("Policy question processing failed"))


# Display the most recent answer
result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("Answer")

    if isinstance(result, dict):
        answer = result.get(
            "answer",
            "No answer was returned.",
        )
        st.markdown(str(answer))

        latency_ms = result.get("latency_ms")
        retrieved_chunks = result.get("retrieved_chunks")

        col1, col2 = st.columns(2)

        if latency_ms is not None:
            col1.metric("Response time", f"{latency_ms} ms")

        if retrieved_chunks is not None:
            col2.metric("Retrieved chunks", retrieved_chunks)

        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources")

            for source in sources:
                if isinstance(source, dict):
                    title = source.get(
                        "title",
                        source.get("file", "Policy document"),
                    )
                    st.markdown(f"- {title}")
                else:
                    st.markdown(f"- {source}")

        snippets = result.get("snippets", [])
        if snippets:
            with st.expander("View supporting excerpts"):
                for index, snippet in enumerate(snippets, start=1):
                    if isinstance(snippet, dict):
                        source_title = snippet.get(
                            "source_title",
                            snippet.get("title", f"Source {index}"),
                        )
                        text = snippet.get("text", "")
                        similarity = snippet.get("similarity")

                        st.markdown(f"**{source_title}**")

                        if text:
                            st.write(text)

                        if similarity is not None:
                            st.caption(
                                f"Similarity: {similarity:.2f}"
                            )

                        st.divider()
    else:
        st.markdown(str(result))

elif not initialization_error and not index_ready:
    st.info(
        "No policy documents are indexed yet. "
        "Run your ingestion script to index the documents first."
    )

st.divider()
st.caption(
    "Acme Corp Policy Assistant | "
    "Verify important policy decisions against the official documents."
)  

