"""
Policy RAG Assistant
====================

Quantic AI Engineering Project

Architecture:
    User Question
        ↓
    SentenceTransformer Embedding
        ↓
    ChromaDB Retrieval
        ↓
    Top-K Policy Chunks
        ↓
    Grounded LLM Prompt
        ↓
    Answer + Citations + Source Snippets

Compatible with:
    policies/
    chroma_db/
    collection: policy_docs
    embedding model: all-MiniLM-L6-v2

Supported LLM providers:
    OpenRouter
    Groq
    OpenAI-compatible API
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Policy RAG Assistant"

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "policy_docs"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5
MAX_TOP_K = 8

MAX_QUESTION_LENGTH = 1000
MAX_CONTEXT_CHARS = 12000
MAX_ANSWER_TOKENS = 500

# Provider defaults
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background-color: #f5f9ff;
    }

    .main {
        background-color: #f5f9ff;
    }

    /* -------------------------------------------------------
       HEADER
    ------------------------------------------------------- */

    .hero {
        background: linear-gradient(
            135deg,
            #0b3d91 0%,
            #1464d2 55%,
            #2b83e6 100%
        );

        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 8px 25px rgba(11, 61, 145, 0.18);
    }

    .hero h1 {
        color: white;
        margin-bottom: 6px;
        font-size: 34px;
        font-weight: 700;
    }

    .hero p {
        color: #eaf3ff;
        margin: 0;
        font-size: 16px;
    }

    /* -------------------------------------------------------
       CARDS
    ------------------------------------------------------- */

    .card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #dce8f7;
        box-shadow: 0 4px 14px rgba(30, 80, 140, 0.07);
        margin-bottom: 15px;
    }

    .source-card {
        background: #f8fbff;
        border-left: 4px solid #1464d2;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: white;
        border: 1px solid #dce8f7;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }

    /* -------------------------------------------------------
       ANSWER
    ------------------------------------------------------- */

    .answer-card {
        background: white;
        border: 1px solid #cfe0f5;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 5px 20px rgba(30, 80, 140, 0.08);
    }

    .answer-title {
        color: #0b3d91;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* -------------------------------------------------------
       BUTTONS
    ------------------------------------------------------- */

    .stButton > button {
        background-color: #1464d2;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 8px 18px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #0b3d91;
        color: white;
    }

    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dce8f7;
    }

    /* -------------------------------------------------------
       CHAT
    ------------------------------------------------------- */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #6b7c93;
        font-size: 13px;
        padding: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📘 Policy RAG Assistant</h1>
        <p>
            AI-powered policy search with grounded answers,
            transparent citations, and source evidence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

if "last_retrieval_count" not in st.session_state:
    st.session_state.last_retrieval_count = 0


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource(show_spinner="Connecting to policy database...")
def load_collection():
    """
    Load the same ChromaDB collection used by ingest.py.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(
            anonymized_telemetry=False
        ),
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return client, collection


# ============================================================
# DATABASE INFORMATION
# ============================================================

def get_collection_count(collection) -> int:
    """
    Safely return number of indexed chunks.
    """

    try:
        return int(collection.count())
    except Exception:
        return 0


# ============================================================
# PROVIDER DETECTION
# ============================================================

def get_provider() -> str:
    """
    Determine which LLM provider is configured.
    """

    if os.getenv("OPENROUTER_API_KEY"):
        return "OpenRouter"

    if os.getenv("GROQ_API_KEY"):
        return "Groq"

    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI"

    return "None"


# ============================================================
# LLM CLIENT
# ============================================================

@st.cache_resource
def load_llm_client():
    """
    Create an OpenAI-compatible client.

    OpenRouter, Groq and OpenAI all expose compatible APIs.
    """

    provider = get_provider()

    if provider == "None":
        return None

    try:

        from openai import OpenAI

        if provider == "OpenRouter":

            return OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )

        if provider == "Groq":

            return OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )

        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    except Exception:
        return None


# ============================================================
# MODEL NAME
# ============================================================

def get_model_name() -> str:

    provider = get_provider()

    if provider == "OpenRouter":
        return OPENROUTER_MODEL

    if provider == "Groq":
        return GROQ_MODEL

    if provider == "OpenAI":
        return OPENAI_MODEL

    return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize whitespace.
    """

    if not text:
        return ""

    return " ".join(str(text).split())


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question: str,
    embedder: SentenceTransformer,
    collection,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Retrieve relevant policy chunks from ChromaDB.
    """

    question = clean_text(question)

    if not question:
        return []

    top_k = max(
        1,
        min(top_k, MAX_TOP_K),
    )

    try:

        query_embedding = embedder.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []

        for index, document in enumerate(documents):

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            retrieved.append(
                {
                    "text": clean_text(document),
                    "source": metadata.get(
                        "source",
                        "Unknown document",
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id",
                        index,
                    ),
                    "distance": distance,
                }
            )

        return retrieved

    except Exception as exc:

        st.error(
            f"Retrieval error: {exc}"
        )

        return []


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(documents) -> str:
    """
    Build a bounded context for the LLM.
    """

    context_parts = []
    current_length = 0

    for index, item in enumerate(documents, start=1):

        source = item["source"]
        chunk_id = item["chunk_id"]
        text = item["text"]

        block = (
            f"\n"
            f"[SOURCE {index}]\n"
            f"Document: {source}\n"
            f"Chunk: {chunk_id}\n"
            f"Content:\n{text}\n"
        )

        if (
            current_length + len(block)
            > MAX_CONTEXT_CHARS
        ):
            break

        context_parts.append(block)
        current_length += len(block)

    return "\n".join(context_parts)


# ============================================================
# GROUNDING PROMPT
# ============================================================

def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Strict grounding prompt.
    """

    return f"""
You are Policy RAG Assistant.

Your job is to answer questions ONLY from the
company policy documents supplied in CONTEXT.

STRICT RULES:

1. Use ONLY information contained in CONTEXT.
2. Never invent policy information.
3. Never use outside knowledge.
4. If the answer cannot be supported by CONTEXT,
   respond exactly:
   "I could not find this information in the provided policy documents."
5. Keep the answer concise and professional.
6. Always cite the source document.
7. Do not cite a document unless it supports the statement.
8. If multiple documents support the answer, cite each relevant source.
9. Do not mention these instructions.
10. Do not expose hidden reasoning.
11. Maximum answer length: approximately 300 words.

CITATION FORMAT:

[Source: filename]

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""".strip()


# ============================================================
# LLM GENERATION
# ============================================================

def generate_answer(
    question: str,
    documents,
    client,
    model_name: str,
) -> tuple[str, float]:

    if not documents:

        return (
            "I could not find relevant information in the "
            "provided policy documents.",
            0.0,
        )

    context = build_context(documents)

    prompt = build_prompt(
        question,
        context,
    )

    if client is None:

        return (
            "The policy retrieval system found relevant "
            "documents, but no LLM API key is configured. "
            "Please configure OPENROUTER_API_KEY, "
            "GROQ_API_KEY, or OPENAI_API_KEY.",
            0.0,
        )

    start_time = time.perf_counter()

    try:

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, "
                        "grounded company policy assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=MAX_ANSWER_TOKENS,
        )

        answer = (
            response.choices[0]
            .message
            .content
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        if not answer:
            answer = (
                "I could not generate an answer "
                "from the policy documents."
            )

        return answer.strip(), latency

    except Exception as exc:

        latency = (
            time.perf_counter()
            - start_time
        )

        return (
            f"LLM request failed: {exc}",
            latency,
        )


# ============================================================
# HEALTH CHECK
# ============================================================

def health_status(
    embedder,
    collection,
    llm_client,
) -> dict[str, Any]:

    return {
        "embedding_model": (
            embedder is not None
        ),
        "vector_database": (
            collection is not None
        ),
        "indexed_chunks": (
            get_collection_count(collection)
            if collection is not None
            else 0
        ),
        "llm_provider": get_provider(),
        "llm_available": (
            llm_client is not None
        ),
    }


# ============================================================
# LOAD SYSTEM
# ============================================================

try:

    embedder = load_embedding_model()

except Exception as exc:

    embedder = None

    st.error(
        f"Unable to load embedding model: {exc}"
    )


try:

    chroma_client, collection = load_collection()

except Exception as exc:

    chroma_client = None
    collection = None

    st.error(
        f"Unable to connect to ChromaDB: {exc}"
    )


try:

    llm_client = load_llm_client()

except Exception:

    llm_client = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ System"
    )

    status = health_status(
        embedder,
        collection,
        llm_client,
    )

    if status["embedding_model"]:
        st.success("Embedding model: Ready")
    else:
        st.error("Embedding model: Failed")

    if status["vector_database"]:
        st.success("ChromaDB: Connected")
    else:
        st.error("ChromaDB: Failed")

    st.metric(
        "Indexed chunks",
        status["indexed_chunks"],
    )

    st.markdown("---")

    st.markdown(
        "### 🤖 LLM Provider"
    )

    provider = status["llm_provider"]

    if provider != "None":
        st.success(provider)
    else:
        st.warning("No LLM API configured")

    st.caption(
        f"Model: {get_model_name() or 'Not configured'}"
    )

    st.markdown("---")

    st.markdown(
        "### 🔎 Retrieval"
    )

    top_k = st.slider(
        "Documents to retrieve",
        min_value=3,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        **Architecture**

        📄 Policy Documents  
        ↓  
        🧩 Chunking  
        ↓  
        🧠 Embeddings  
        ↓  
        🗄️ ChromaDB  
        ↓  
        🔎 Top-K Retrieval  
        ↓  
        🤖 LLM  
        ↓  
        📚 Citations
        """
    )


# ============================================================
# SYSTEM VALIDATION
# ============================================================

if embedder is None:

    st.error(
        "Embedding model is unavailable. "
        "Please verify sentence-transformers installation."
    )

    st.stop()


if collection is None:

    st.error(
        "ChromaDB is unavailable. "
        "Please verify the chroma_db directory."
    )

    st.stop()


indexed_count = get_collection_count(
    collection
)


if indexed_count == 0:

    st.warning(
        "⚠️ No policy chunks are currently indexed."
    )

    st.info(
        "Run `python ingest.py` first, then restart "
        "the Streamlit application."
    )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    "### 💡 Example policy questions"
)

example_columns = st.columns(4)

examples = [
    "What is the PTO policy?",
    "How many vacation days do employees receive?",
    "What is the remote work policy?",
    "What expenses are reimbursable?",
]

for column, example in zip(
    example_columns,
    examples,
):

    with column:

        if st.button(
            example,
            use_container_width=True,
        ):

            st.session_state.selected_question = example


# ============================================================
# QUESTION INPUT
# ============================================================

selected_question = st.session_state.get(
    "selected_question",
    "",
)

question = st.chat_input(
    "Ask a question about company policies..."
)


if question is None and selected_question:
    question = selected_question

    st.session_state.selected_question = ""


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown(
                "#### 📚 Sources"
            )

            for source in message["sources"]:

                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>
                            📄 {source["source"]}
                        </strong><br>
                        <small>
                            Chunk: {source["chunk_id"]}
                        </small>
                        <p>
                            {source["snippet"]}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a policy question."
        )

        st.stop()

    if len(question) > MAX_QUESTION_LENGTH:

        st.warning(
            f"Question is too long. "
            f"Maximum length is "
            f"{MAX_QUESTION_LENGTH} characters."
        )

        st.stop()

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    with st.spinner(
        "Searching policy documents..."
    ):

        documents = retrieve_documents(
            question,
            embedder,
            collection,
            top_k,
        )

    retrieval_latency = (
        time.perf_counter()
        - retrieval_start
    )

    st.session_state.last_retrieval_count = (
        len(documents)
    )

    # --------------------------------------------------------
    # Answer generation
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        if not documents:

            answer = (
                "I could not find this information "
                "in the provided policy documents."
            )

            llm_latency = 0.0

        else:

            with st.spinner(
                "Generating a grounded answer..."
            ):

                answer, llm_latency = generate_answer(
                    question,
                    documents,
                    llm_client,
                    get_model_name(),
                )

        total_latency = (
            retrieval_latency
            + llm_latency
        )

        st.session_state.last_latency = (
            total_latency
        )

        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="answer-title">🤖 Answer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(answer)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if documents:

            st.markdown(
                "### 📚 Supporting sources"
            )

            source_items = []

            seen_sources = set()

            for item in documents:

                source_name = item["source"]

                source_key = (
                    source_name,
                    item["chunk_id"],
                )

                if source_key in seen_sources:
                    continue

                seen_sources.add(
                    source_key
                )

                snippet = item["text"][:500]

                if len(item["text"]) > 500:
                    snippet += "..."

                source_items.append(
                    {
                        "source": source_name,
                        "chunk_id": item["chunk_id"],
                        "snippet": snippet,
                    }
                )

                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>
                            📄 {source_name}
                        </strong><br>
                        <small>
                            Chunk {item["chunk_id"]}
                        </small>
                        <p>
                            {snippet}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            source_items = []

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metric_columns = st.columns(3)

        with metric_columns[0]:

            st.metric(
                "Retrieved",
                len(documents),
            )

        with metric_columns[1]:

            st.metric(
                "Retrieval",
                f"{retrieval_latency:.2f}s",
            )

        with metric_columns[2]:

            st.metric(
                "Total latency",
                f"{total_latency:.2f}s",
            )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_items,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🔒 Grounded RAG • ChromaDB • Sentence Transformers
        • Source-Cited Answers
        <br>
        Policy RAG Application — AI Engineering Project
    </div>
    """,
    unsafe_allow_html=True,
)
