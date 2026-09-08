"""
Policy RAG Copilot
==================

Production-oriented Streamlit interface for the Policy RAG application.

Architecture:
    User Question
        ↓
    Input Validation
        ↓
    Query Embedding
        ↓
    ChromaDB Retrieval
        ↓
    Relevance Filtering
        ↓
    Context Construction
        ↓
    LLM Generation
        ↓
    Citation Validation
        ↓
    Answer + Sources + Metrics

Compatible with the current ingest.py:
    Chroma path: ./chroma_db
    Collection: policy_docs
    Embedding: all-MiniLM-L6-v2
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Groq, OpenRouter


# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_db"
POLICIES_DIR = BASE_DIR / "policies"

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "policy_docs",
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "8"))

MAX_QUERY_LENGTH = int(
    os.getenv("MAX_QUERY_LENGTH", "1000")
)

MAX_CONTEXT_CHARS = int(
    os.getenv("MAX_CONTEXT_CHARS", "18000")
)

MAX_RESPONSE_TOKENS = int(
    os.getenv("MAX_RESPONSE_TOKENS", "700")
)

MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.20")
)

APP_VERSION = os.getenv(
    "APP_VERSION",
    "2.0.0",
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("policy-rag-app")


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Policy Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM UI
# ============================================================================

st.markdown(
    """
<style>

:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --background: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --success: #16a34a;
    --warning: #d97706;
    --danger: #dc2626;
}

.stApp {
    background: var(--background);
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Header */

.hero {
    background:
        linear-gradient(
            135deg,
            #1e3a8a 0%,
            #2563eb 55%,
            #3b82f6 100%
        );
    padding: 2rem 2.2rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
}

.hero-title {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    color: #dbeafe;
    font-size: 1rem;
    margin-top: 0.5rem;
}

/* Cards */

.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    min-height: 105px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.metric-value {
    color: var(--text);
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 0.25rem;
}

/* Source cards */

.source-card {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.source-title {
    font-weight: 700;
    color: #1e293b;
}

.source-score {
    color: #2563eb;
    font-size: 0.85rem;
    font-weight: 600;
}

.source-snippet {
    color: #475569;
    font-size: 0.88rem;
    margin-top: 0.5rem;
    line-height: 1.55;
}

/* Status */

.status-good {
    color: #15803d;
    font-weight: 700;
}

.status-warning {
    color: #b45309;
    font-weight: 700;
}

.status-bad {
    color: #b91c1c;
    font-weight: 700;
}

/* Welcome */

.welcome-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.welcome-title {
    font-size: 1.2rem;
    font-weight: 750;
    color: var(--text);
}

.welcome-text {
    color: var(--muted);
    line-height: 1.6;
}

/* Footer */

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class RetrievedDocument:
    text: str
    score: float
    source: str
    filename: str
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    answer: str
    citations: List[Dict[str, Any]]
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    retrieved_count: int
    grounded_context: bool


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now() -> str:
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_query(query: str) -> str:
    """Normalize user query."""
    if not query:
        return ""

    query = query.replace("\x00", " ")
    query = re.sub(r"\s+", " ", query)
    return query.strip()


def validate_query(query: str) -> Tuple[bool, str]:
    """Validate incoming user query."""

    query = clean_query(query)

    if not query:
        return False, "Please enter a question."

    if len(query) > MAX_QUERY_LENGTH:
        return (
            False,
            f"Your question is too long. "
            f"Please keep it under {MAX_QUERY_LENGTH} characters.",
        )

    if len(query) < 3:
        return False, "Please enter a more specific question."

    return True, ""


def truncate_text(
    text: str,
    max_chars: int,
) -> str:
    """Safely truncate text."""
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n[Context truncated]"


def normalize_source(source: str) -> str:
    """Normalize source names for consistent citations."""
    if not source:
        return "Unknown source"

    return str(source).replace("\\", "/")


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

def create_llm():
    """
    Create LLM using environment configuration.

    Priority:
        1. Groq
        2. OpenRouter
    """

    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    groq_model = os.getenv(
        "GROQ_MODEL",
        "llama-3.1-8b-instant",
    )

    openrouter_model = os.getenv(
        "OPENROUTER_MODEL",
        "mistralai/mistral-7b-instruct",
    )

    temperature = float(
        os.getenv("LLM_TEMPERATURE", "0.1")
    )

    if groq_key:
        try:
            logger.info(
                "Initializing Groq model: %s",
                groq_model,
            )

            return Groq(
                groq_api_key=groq_key,
                model_name=groq_model,
                temperature=temperature,
                max_tokens=MAX_RESPONSE_TOKENS,
            )

        except Exception as exc:
            logger.exception(
                "Groq initialization failed: %s",
                exc,
            )

    if openrouter_key:
        try:
            logger.info(
                "Initializing OpenRouter model: %s",
                openrouter_model,
            )

            return OpenRouter(
                openrouter_api_key=openrouter_key,
                model=openrouter_model,
                temperature=temperature,
                max_tokens=MAX_RESPONSE_TOKENS,
            )

        except Exception as exc:
            logger.exception(
                "OpenRouter initialization failed: %s",
                exc,
            )

    raise RuntimeError(
        "No working LLM configuration found. "
        "Configure GROQ_API_KEY or OPENROUTER_API_KEY."
    )


# ============================================================================
# EMBEDDINGS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load embedding model once per Streamlit process.

    Caching prevents the model from being reloaded
    on every Streamlit rerun.
    """

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL_NAME,
    )

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


# ============================================================================
# CHROMA DATABASE
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """
    Create persistent Chroma client.

    IMPORTANT:
    This matches the current ingest.py implementation.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Opening ChromaDB: %s",
        CHROMA_DIR,
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


@st.cache_resource(show_spinner=False)
def get_collection():
    """
    Load the same collection used by ingest.py.
    """

    client = get_chroma_client()

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )

    except Exception:
        logger.warning(
            "Collection '%s' does not exist yet.",
            COLLECTION_NAME,
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    return collection


# ============================================================================
# RAG ENGINE
# ============================================================================

class PolicyRAGEngine:
    """Reliable policy RAG engine."""

    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.collection = get_collection()
        self.llm = create_llm()

        logger.info(
            "PolicyRAGEngine initialized successfully."
        )

    # ------------------------------------------------------------------------
    # DATABASE HEALTH
    # ------------------------------------------------------------------------

    def collection_count(self) -> int:
        """Return number of indexed chunks."""

        try:
            return int(self.collection.count())
        except Exception as exc:
            logger.error(
                "Unable to read Chroma count: %s",
                exc,
            )
            return 0

    # ------------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------------

    def retrieve_documents(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant policy chunks.

        Uses cosine distance from Chroma and converts
        it into an approximate similarity score.
        """

        query = clean_query(query)

        if not query:
            return []

        k = max(
            1,
            min(k, MAX_TOP_K),
        )

        count = self.collection_count()

        if count == 0:
            logger.warning(
                "Chroma collection is empty."
            )
            return []

        k = min(k, count)

        retrieval_start = time.perf_counter()

        try:
            query_embedding = (
                self.embedding_model.embed_query(
                    query
                )
            )

            results = self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=k,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

        except Exception as exc:
            logger.exception(
                "Chroma retrieval failed: %s",
                exc,
            )
            raise RuntimeError(
                "The policy search service is temporarily "
                "unavailable. Please try again."
            ) from exc

        elapsed = (
            time.perf_counter()
            - retrieval_start
        )

        logger.info(
            "Retrieved %d documents in %.3fs",
            len(results.get("documents", [[]])[0]),
            elapsed,
        )

        documents = results.get(
            "documents",
            [[]],
        )

        metadatas = results.get(
            "metadatas",
            [[]],
        )

        distances = results.get(
            "distances",
            [[]],
        )

        if not documents or not documents[0]:
            return []

        retrieved: List[RetrievedDocument] = []

        for index, text in enumerate(
            documents[0]
        ):

            metadata = {}

            if (
                metadatas
                and metadatas[0]
                and index < len(metadatas[0])
            ):
                metadata = (
                    metadatas[0][index]
                    or {}
                )

            distance = 1.0

            if (
                distances
                and distances[0]
                and index < len(distances[0])
            ):
                distance = safe_float(
                    distances[0][index],
                    1.0,
                )

            # For cosine distance:
            # similarity ≈ 1 - distance
            score = max(
                0.0,
                min(
                    1.0,
                    1.0 - distance,
                ),
            )

            source = normalize_source(
                metadata.get(
                    "source",
                    metadata.get(
                        "filename",
                        "Unknown source",
                    ),
                )
            )

            filename = str(
                metadata.get(
                    "filename",
                    Path(source).name,
                )
            )

            chunk_id = str(
                metadata.get(
                    "chunk_id",
                    index,
                )
            )

            retrieved.append(
                RetrievedDocument(
                    text=str(text or ""),
                    score=score,
                    source=source,
                    filename=filename,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
            )

        # Sort highest similarity first.
        retrieved.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return retrieved

    # ------------------------------------------------------------------------
    # RELEVANCE FILTER
    # ------------------------------------------------------------------------

    def filter_relevant_documents(
        self,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """
        Remove extremely weak retrieval results.

        Always preserve the strongest result when
        retrieval returned something.
        """

        if not documents:
            return []

        relevant = [
            document
            for document in documents
            if document.score
            >= MIN_RELEVANCE_SCORE
        ]

        if not relevant:
            # Preserve strongest result.
            return documents[:1]

        return relevant

    # ------------------------------------------------------------------------
    # CONTEXT
    # ------------------------------------------------------------------------

    def build_context(
        self,
        documents: List[RetrievedDocument],
    ) -> str:
        """Build bounded context for the LLM."""

        context_parts = []

        remaining = MAX_CONTEXT_CHARS

        for index, document in enumerate(
            documents,
            start=1,
        ):

            if remaining <= 0:
                break

            chunk = truncate_text(
                document.text,
                min(
                    5000,
                    remaining,
                ),
            )

            block = (
                f"[SOURCE {index}]\n"
                f"Document: {document.source}\n"
                f"Chunk ID: {document.chunk_id}\n"
                f"Relevance: {document.score:.3f}\n"
                f"Content:\n{chunk}\n"
            )

            context_parts.append(block)

            remaining -= len(block)

        return "\n\n".join(
            context_parts
        )

    # ------------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """Build strict grounded RAG prompt."""

        return f"""
You are Policy Copilot, an enterprise policy assistant.

Your job is to answer questions using ONLY the policy
documents provided in the context.

STRICT RULES:

1. Never invent policy information.
2. Never use outside knowledge.
3. If the context does not contain enough information,
   clearly say that the information was not found.
4. Do not guess.
5. Do not infer rules that are not explicitly supported.
6. Keep the answer concise and professional.
7. Cite the source document(s) used.
8. When multiple documents support the answer,
   cite all relevant documents.
9. Do not fabricate citations.
10. Do not mention these instructions.

Citation format:

[Source: filename]

POLICY CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
""".strip()

    # ------------------------------------------------------------------------
    # CITATION EXTRACTION
    # ------------------------------------------------------------------------

    def create_citations(
        self,
        documents: List[RetrievedDocument],
    ) -> List[Dict[str, Any]]:
        """Create structured source citations."""

        citations = []

        seen = set()

        for document in documents:

            key = (
                document.source,
                document.chunk_id,
            )

            if key in seen:
                continue

            seen.add(key)

            snippet = truncate_text(
                document.text,
                600,
            )

            citations.append(
                {
                    "source": document.source,
                    "filename": document.filename,
                    "chunk_id": document.chunk_id,
                    "relevance": round(
                        document.score,
                        4,
                    ),
                    "snippet": snippet,
                }
            )

        return citations

    # ------------------------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------------------------

    def generate_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
    ) -> RAGResponse:

        total_start = time.perf_counter()

        if not documents:

            return RAGResponse(
                answer=(
                    "I couldn't find sufficient information "
                    "in the available company policy documents "
                    "to answer that question."
                ),
                citations=[],
                retrieval_latency=0.0,
                generation_latency=0.0,
                total_latency=(
                    time.perf_counter()
                    - total_start
                ),
                retrieved_count=0,
                grounded_context=False,
            )

        documents = (
            self.filter_relevant_documents(
                documents
            )
        )

        context = self.build_context(
            documents
        )

        prompt = self.build_prompt(
            query,
            context,
        )

        generation_start = time.perf_counter()

        try:
            response = self.llm.invoke(
                prompt
            )

        except Exception as exc:

            logger.exception(
                "LLM generation failed: %s",
                exc,
            )

            raise RuntimeError(
                "The language model could not "
                "generate a response right now. "
                "Please try again."
            ) from exc

        generation_latency = (
            time.perf_counter()
            - generation_start
        )

        answer = str(
            response or ""
        ).strip()

        if not answer:
            answer = (
                "I couldn't generate a reliable "
                "answer from the available policy documents."
            )

        citations = self.create_citations(
            documents
        )

        total_latency = (
            time.perf_counter()
            - total_start
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieval_latency=0.0,
            generation_latency=generation_latency,
            total_latency=total_latency,
            retrieved_count=len(documents),
            grounded_context=True,
        )


# ============================================================================
# CACHED ENGINE
# ============================================================================

@st.cache_resource(show_spinner=False)
def initialize_rag_engine():
    """Initialize RAG engine once."""

    return PolicyRAGEngine()


# ============================================================================
# UI HELPERS
# ============================================================================

def render_metric_card(
    label: str,
    value: str,
):
    """Render custom metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(
    citations: List[Dict[str, Any]],
):
    """Render source cards."""

    if not citations:
        return

    st.markdown(
        "### 📚 Sources"
    )

    for index, citation in enumerate(
        citations,
        start=1,
    ):

        relevance = (
            citation["relevance"]
        )

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">
                    {index}. 📄
                    {citation["filename"]}
                </div>

                <div class="source-score">
                    Relevance:
                    {relevance:.1%}
                    &nbsp; • &nbsp;
                    Chunk:
                    {citation["chunk_id"]}
                </div>

                <div class="source-snippet">
                    {citation["snippet"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_system_status(
    engine: Optional[PolicyRAGEngine],
):
    """Render application health status."""

    st.sidebar.markdown(
        "## 🩺 System Status"
    )

    if engine is None:

        st.sidebar.markdown(
            '<span class="status-bad">'
            '● Offline'
            '</span>',
            unsafe_allow_html=True,
        )

        return

    try:
        count = engine.collection_count()

        if count > 0:

            st.sidebar.markdown(
                '<span class="status-good">'
                '● Operational'
                '</span>',
                unsafe_allow_html=True,
            )

            st.sidebar.caption(
                f"{count:,} indexed chunks"
            )

        else:

            st.sidebar.markdown(
                '<span class="status-warning">'
                '● No documents indexed'
                '</span>',
                unsafe_allow_html=True,
            )

    except Exception:

        st.sidebar.markdown(
            '<span class="status-bad">'
            '● Database error'
            '</span>',
            unsafe_allow_html=True,
        )


def export_conversation():
    """Export conversation as JSON."""

    messages = st.session_state.get(
        "messages",
        [],
    )

    payload = {
        "application": "Policy Copilot",
        "version": APP_VERSION,
        "exported_at": utc_now(),
        "messages": messages,
    }

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================================
# SESSION STATE
# ============================================================================

def initialize_session():
    """Initialize Streamlit session state."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "metrics" not in st.session_state:
        st.session_state.metrics = []

    if "last_error" not in st.session_state:
        st.session_state.last_error = None


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(
    engine: Optional[PolicyRAGEngine],
):
    """Render application settings."""

    with st.sidebar:

        st.markdown(
            "## 🛡️ Policy Copilot"
        )

        st.caption(
            "Enterprise RAG Policy Assistant"
        )

        st.divider()

        st.markdown(
            "### 🔎 Retrieval"
        )

        top_k = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=MAX_TOP_K,
            value=min(
                DEFAULT_TOP_K,
                MAX_TOP_K,
            ),
            step=1,
            help=(
                "Number of policy chunks retrieved "
                "from ChromaDB."
            ),
        )

        st.divider()

        st.markdown(
            "### 📊 Session Metrics"
        )

        metrics = st.session_state.metrics

        if metrics:

            total_queries = len(metrics)

            latencies = [
                item["total_latency"]
                for item in metrics
                if item.get("total_latency")
            ]

            average_latency = (
                sum(latencies)
                / len(latencies)
                if latencies
                else 0
            )

            st.metric(
                "Queries",
                total_queries,
            )

            st.metric(
                "Avg latency",
                f"{average_latency:.2f}s",
            )

        else:

            st.caption(
                "Metrics will appear after "
                "your first question."
            )

        st.divider()

        st.markdown(
            "### 🗄️ Knowledge Base"
        )

        if engine:

            try:

                count = (
                    engine.collection_count()
                )

                st.metric(
                    "Indexed chunks",
                    f"{count:,}",
                )

                st.caption(
                    f"Collection: `{COLLECTION_NAME}`"
                )

            except Exception:

                st.error(
                    "Unable to read database."
                )

        st.divider()

        st.markdown(
            "### ⚙️ Application"
        )

        st.caption(
            f"Version: {APP_VERSION}"
        )

        st.caption(
            f"Embedding: {EMBEDDING_MODEL_NAME}"
        )

        st.caption(
            f"Vector DB: ChromaDB"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🧹 Clear",
                use_container_width=True,
            ):

                st.session_state.messages = []
                st.session_state.metrics = []
                st.session_state.last_error = None

                st.rerun()

        with col2:

            st.download_button(
                "⬇️ Export",
                data=export_conversation(),
                file_name=(
                    "policy_copilot_conversation.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

    return top_k


# ============================================================================
# CHAT HISTORY
# ============================================================================

def render_chat_history():
    """Render existing conversation."""

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(role):

            st.markdown(content)

            if (
                role == "assistant"
                and message.get("citations")
            ):

                with st.expander(
                    "📚 View supporting sources"
                ):

                    render_sources(
                        message["citations"]
                    )

                latency = message.get(
                    "total_latency"
                )

                if latency:

                    st.caption(
                        f"⏱️ {latency:.2f}s"
                    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    initialize_session()

    # ------------------------------------------------------------------------
    # HERO
    # ------------------------------------------------------------------------

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🛡️ Policy Copilot
            </div>

            <div class="hero-subtitle">
                Ask questions about company policies
                and receive grounded, source-cited answers.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------------
    # INITIALIZE ENGINE
    # ------------------------------------------------------------------------

    engine = None

    try:

        with st.spinner(
            "Initializing Policy Copilot..."
        ):

            engine = (
                initialize_rag_engine()
            )

    except Exception as exc:

        logger.exception(
            "Application initialization failed: %s",
            exc,
        )

        st.error(
            "⚠️ Policy Copilot could not initialize."
        )

        st.info(
            "Check your API keys and make sure "
            "the policy database has been created."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(exc)
            )

        render_system_status(
            None
        )

        return

    # ------------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------------

    top_k = render_sidebar(
        engine
    )

    render_system_status(
        engine
    )

    # ------------------------------------------------------------------------
    # TOP METRICS
    # ------------------------------------------------------------------------

    try:

        chunk_count = (
            engine.collection_count()
        )

    except Exception:

        chunk_count = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        render_metric_card(
            "Knowledge chunks",
            f"{chunk_count:,}",
        )

    with col2:

        render_metric_card(
            "Top-K",
            str(top_k),
        )

    with col3:

        render_metric_card(
            "Embedding",
            "MiniLM",
        )

    with col4:

        render_metric_card(
            "RAG status",
            "Ready"
            if chunk_count > 0
            else "Empty",
        )

    # ------------------------------------------------------------------------
    # WELCOME
    # ------------------------------------------------------------------------

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="welcome-card">

                <div class="welcome-title">
                    👋 Welcome to Policy Copilot
                </div>

                <div class="welcome-text">
                    I search the indexed policy documents
                    and answer using only the information
                    available in the knowledge base.
                    Every answer is accompanied by supporting
                    source snippets when evidence is found.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "### 💡 Try asking"
        )

        examples = [
            "What is the PTO policy?",
            "How many vacation days do employees receive?",
            "What is the remote work policy?",
            "What are the password security requirements?",
            "Which employee expenses are reimbursable?",
            "What should I do if I cannot find a policy?",
        ]

        example_cols = st.columns(2)

        for index, example in enumerate(
            examples
        ):

            with example_cols[
                index % 2
            ]:

                if st.button(
                    example,
                    use_container_width=True,
                    key=f"example_{index}",
                ):

                    st.session_state[
                        "pending_question"
                    ] = example

                    st.rerun()

    # ------------------------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------------------------

    render_chat_history()

    # ------------------------------------------------------------------------
    # USER INPUT
    # ------------------------------------------------------------------------

    pending_question = (
        st.session_state.pop(
            "pending_question",
            None,
        )
    )

    user_input = st.chat_input(
        "Ask a question about company policies..."
    )

    if pending_question:
        user_input = pending_question

    if not user_input:
        return

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    valid, validation_message = (
        validate_query(
            user_input
        )
    )

    if not valid:

        st.warning(
            validation_message
        )

        return

    user_input = clean_query(
        user_input
    )

    # ------------------------------------------------------------------------
    # USER MESSAGE
    # ------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "timestamp": utc_now(),
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    # ------------------------------------------------------------------------
    # RAG PIPELINE
    # ------------------------------------------------------------------------

    with st.chat_message("assistant"):

        status = st.empty()

        try:

            status.markdown(
                "🔎 Searching policy documents..."
            )

            request_start = (
                time.perf_counter()
            )

            retrieval_start = (
                time.perf_counter()
            )

            retrieved_documents = (
                engine.retrieve_documents(
                    user_input,
                    k=top_k,
                )
            )

            retrieval_latency = (
                time.perf_counter()
                - retrieval_start
            )

            if not retrieved_documents:

                total_latency = (
                    time.perf_counter()
                    - request_start
                )

                answer = (
                    "I couldn't find sufficient "
                    "information in the available "
                    "company policy documents to "
                    "answer that question.\n\n"
                    "Please try asking about a policy "
                    "covered by the knowledge base."
                )

                citations = []

                generation_latency = 0.0

            else:

                status.markdown(
                    "🤖 Generating grounded answer..."
                )

                rag_response = (
                    engine.generate_answer(
                        user_input,
                        retrieved_documents,
                    )
                )

                answer = (
                    rag_response.answer
                )

                citations = (
                    rag_response.citations
                )

                generation_latency = (
                    rag_response.generation_latency
                )

                total_latency = (
                    time.perf_counter()
                    - request_start
                )

            status.empty()

            # --------------------------------------------------------------
            # ANSWER
            # --------------------------------------------------------------

            st.markdown(answer)

            # --------------------------------------------------------------
            # SOURCES
            # --------------------------------------------------------------

            if citations:

                with st.expander(
                    f"📚 Supporting sources "
                    f"({len(citations)})",
                    expanded=True,
                ):

                    render_sources(
                        citations
                    )

            # --------------------------------------------------------------
            # METRICS
            # --------------------------------------------------------------

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            with metric_col1:

                st.caption(
                    f"🔎 Retrieval: "
                    f"{retrieval_latency:.2f}s"
                )

            with metric_col2:

                st.caption(
                    f"🤖 Generation: "
                    f"{generation_latency:.2f}s"
                )

            with metric_col3:

                st.caption(
                    f"⏱️ Total: "
                    f"{total_latency:.2f}s"
                )

            # --------------------------------------------------------------
            # SAVE METRICS
            # --------------------------------------------------------------

            st.session_state.metrics.append(
                {
                    "timestamp": utc_now(),
                    "query": user_input,
                    "top_k": top_k,
                    "retrieved_count": len(
                        retrieved_documents
                    ),
                    "citation_count": len(
                        citations
                    ),
                    "retrieval_latency": (
                        retrieval_latency
                    ),
                    "generation_latency": (
                        generation_latency
                    ),
                    "total_latency": (
                        total_latency
                    ),
                }
            )

            # --------------------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # --------------------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "retrieved_count": len(
                        retrieved_documents
                    ),
                    "retrieval_latency": (
                        retrieval_latency
                    ),
                    "generation_latency": (
                        generation_latency
                    ),
                    "total_latency": (
                        total_latency
                    ),
                    "timestamp": utc_now(),
                }
            )

        except Exception as exc:

            status.empty()

            logger.exception(
                "Request failed: %s",
                exc,
            )

            error_message = (
                "⚠️ I couldn't complete that request "
                "because a backend component failed. "
                "Please try again."
            )

            st.error(
                error_message
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(exc)
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "citations": [],
                    "timestamp": utc_now(),
                }
            )

    # ------------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------------

    st.markdown(
        """
        <div class="footer">
            🛡️ Policy Copilot ·
            Retrieval-Augmented Generation ·
            Source-grounded responses
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
