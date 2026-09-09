"""
PolicyCopilot — Enterprise Policy Intelligence
Streamlit UI

IMPORTANT:
All HTML below is contained inside Streamlit markdown strings and is
used only for visual styling. It must never be written as standalone
text outside st.markdown(...).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

APP_NAME = "PolicyCopilot"
APP_SUBTITLE = "Enterprise Policy Intelligence"

TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

MAX_QUESTION_LENGTH = 1000


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger("policycopilot")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot | Enterprise Policy Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS ONLY
#
# This is the ONLY HTML block used by the application.
# It is styling markup, not visible application content.
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

:root {
    --blue: #155EEF;
    --deep-blue: #0B3B91;
    --light-blue: #EAF4FF;
    --soft-blue: #F4F8FF;

    --green: #16A34A;
    --light-green: #ECFDF3;

    --white: #FFFFFF;
    --black: #0B1220;

    --gray-700: #334155;
    --gray-600: #64748B;
    --gray-500: #94A3B8;
    --gray-300: #CBD5E1;

    --border: #D9E2EC;
    --background: #F4F7FB;

    --warning: #B45309;
    --warning-bg: #FFF7ED;
}

/* =========================================================
   APPLICATION
   ========================================================= */

.stApp {
    background:
        linear-gradient(
            180deg,
            #F8FBFF 0%,
            #F4F7FB 48%,
            #F8FAFC 100%
        );
    color: var(--black);
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #FFFFFF 0%,
            #F7FAFF 100%
        );

    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* =========================================================
   BRAND
   ========================================================= */

.brand-box {
    background:
        linear-gradient(
            135deg,
            #0B3B91 0%,
            #155EEF 100%
        );

    border-radius: 18px;
    padding: 18px;
    color: white;

    margin-bottom: 20px;

    box-shadow:
        0 8px 25px
        rgba(21, 94, 239, 0.18);
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    width: 45px;
    height: 45px;

    border-radius: 13px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);

    font-size: 1.25rem;
    font-weight: 900;
}

.brand-name {
    font-size: 1.18rem;
    font-weight: 850;
}

.brand-description {
    font-size: 0.73rem;
    opacity: 0.82;
    margin-top: 3px;
}

/* =========================================================
   TOP NAVIGATION
   ========================================================= */

.topbar {
    background: white;
    border: 1px solid var(--border);
    border-radius: 17px;

    padding: 13px 18px;
    margin-bottom: 17px;

    box-shadow:
        0 4px 18px
        rgba(15, 23, 42, 0.045);
}

.topbar-title {
    font-size: 1rem;
    font-weight: 850;
    color: var(--black);
}

.topbar-subtitle {
    font-size: 0.73rem;
    color: var(--gray-600);
    margin-top: 2px;
}

.status-pill {
    display: inline-block;

    padding: 5px 10px;

    border-radius: 999px;

    background: var(--light-green);
    color: var(--green);

    border: 1px solid #BBF7D0;

    font-size: 0.7rem;
    font-weight: 800;
}

/* =========================================================
   HERO
   ========================================================= */

.hero-box {
    background: white;

    border: 1px solid var(--border);
    border-radius: 22px;

    padding: 31px 34px;

    margin-bottom: 18px;

    box-shadow:
        0 8px 28px
        rgba(15, 23, 42, 0.045);
}

.hero-title {
    font-size: 2.05rem;

    line-height: 1.15;

    font-weight: 850;

    color: var(--black);

    letter-spacing: -0.8px;
}

.hero-title-accent {
    color: var(--blue);
}

.hero-description {
    max-width: 850px;

    color: var(--gray-600);

    font-size: 0.95rem;

    line-height: 1.65;

    margin-top: 10px;
}

/* =========================================================
   CARDS
   ========================================================= */

.card {
    background: white;

    border: 1px solid var(--border);

    border-radius: 16px;

    padding: 17px;

    height: 100%;

    box-shadow:
        0 4px 16px
        rgba(15, 23, 42, 0.035);
}

.card-label {
    font-size: 0.7rem;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.5px;

    color: var(--gray-600);
}

.card-value {
    font-size: 1.25rem;

    font-weight: 850;

    color: var(--black);

    margin-top: 6px;
}

.card-description {
    font-size: 0.73rem;

    line-height: 1.45;

    color: var(--gray-600);

    margin-top: 4px;
}

/* =========================================================
   CHAT
   ========================================================= */

.chat-frame {
    background: white;

    border: 1px solid var(--border);

    border-radius: 22px;

    padding: 22px;

    min-height: 410px;

    box-shadow:
        0 8px 28px
        rgba(15, 23, 42, 0.045);
}

.chat-header {
    display: flex;

    justify-content: space-between;

    align-items: center;

    padding-bottom: 14px;

    border-bottom: 1px solid #E8EEF5;

    margin-bottom: 16px;
}

.chat-title {
    font-size: 1rem;

    font-weight: 850;

    color: var(--black);
}

.chat-subtitle {
    font-size: 0.74rem;

    color: var(--gray-600);

    margin-top: 3px;
}

/* =========================================================
   MESSAGES
   ========================================================= */

.message {
    border-radius: 16px;

    padding: 14px 16px;

    margin: 10px 0;

    border: 1px solid var(--border);
}

.user-message {
    background: var(--soft-blue);

    border-color: #CFE0FF;
}

.assistant-message {
    background: white;
}

.message-label {
    font-size: 0.68rem;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 0.5px;

    color: var(--gray-600);

    margin-bottom: 5px;
}

.message-time {
    font-size: 0.65rem;

    color: var(--gray-500);

    margin-top: 7px;
}

/* =========================================================
   SOURCES
   ========================================================= */

.source-card {
    background: #FBFDFF;

    border: 1px solid var(--border);

    border-left: 4px solid var(--blue);

    border-radius: 12px;

    padding: 13px;

    margin-top: 8px;
}

.source-title {
    font-size: 0.81rem;

    font-weight: 800;

    color: var(--black);
}

.source-meta {
    font-size: 0.68rem;

    color: var(--gray-600);

    margin-top: 3px;
}

/* =========================================================
   EMPTY STATE
   ========================================================= */

.empty-state {
    text-align: center;

    padding: 58px 25px;

    color: var(--gray-600);
}

.empty-icon {
    font-size: 2.3rem;

    margin-bottom: 10px;
}

.empty-title {
    font-size: 1.1rem;

    font-weight: 800;

    color: var(--black);
}

.empty-description {
    max-width: 650px;

    margin: 7px auto 0 auto;

    font-size: 0.82rem;

    line-height: 1.55;
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: var(--gray-500);

    font-size: 0.68rem;

    padding: 25px 0 5px 0;
}

/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    border-radius: 10px !important;

    font-weight: 700 !important;

    border: 1px solid var(--border) !important;
}

.stButton > button:hover {
    border-color: var(--blue) !important;

    color: var(--blue) !important;
}

/* =========================================================
   INPUT
   ========================================================= */

div[data-baseweb="input"] > div {
    border-radius: 12px !important;

    border-color: var(--border) !important;
}

textarea {
    border-radius: 12px !important;
}

/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {
    background: white;

    border: 1px solid var(--border);

    border-radius: 14px;

    padding: 12px;
}

/* =========================================================
   EXPANDERS
   ========================================================= */

div[data-testid="stExpander"] {
    border: 1px solid var(--border);

    border-radius: 12px;

    background: white;
}

/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 900px) {

    .hero-title {
        font-size: 1.55rem;
    }

    .hero-box {
        padding: 22px;
    }

    .chat-frame {
        padding: 15px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:

    defaults = {
        "chat_history": [],
        "last_result": None,
        "policy_question": "",
        "top_k": TOP_K_DEFAULT,
        "rag_initialized": False,
        "rag_error": None,
        "total_questions": 0,
        "last_latency_ms": None,
        "current_page": "Chat",
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# RAG PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_pipeline(top_k: int):

    try:

        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=top_k
        )

        return pipeline, None

    except Exception as exc:

        LOGGER.exception(
            "RAG initialization failed"
        )

        return None, str(exc)


def get_rag_pipeline():

    pipeline, error = load_rag_pipeline(
        st.session_state.top_k
    )

    st.session_state.rag_initialized = (
        pipeline is not None
    )

    st.session_state.rag_error = error

    return pipeline, error


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def timestamp() -> str:

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def normalize_result(result: Any) -> Dict[str, Any]:

    if result is None:

        return {
            "answer": "",
            "citations": [],
            "sources": [],
        }

    if isinstance(result, dict):

        return {
            "answer": clean_text(
                result.get("answer", "")
            ),
            "citations": result.get(
                "citations", []
            ) or [],
            "sources": result.get(
                "sources", []
            ) or [],
        }

    return {
        "answer": clean_text(result),
        "citations": [],
        "sources": [],
    }


def normalize_source(source: Any) -> Dict[str, str]:

    if not isinstance(source, dict):

        return {
            "title": "Unknown Policy",
            "document_id": "unknown",
            "section": "Unknown Section",
            "source": "unknown",
            "snippet": clean_text(source),
        }

    return {
        "title": clean_text(
            source.get("title")
            or source.get("source")
            or "Unknown Policy"
        ),
        "document_id": clean_text(
            source.get("document_id")
            or source.get("source")
            or "unknown"
        ),
        "section": clean_text(
            source.get("section")
            or "Unknown Section"
        ),
        "source": clean_text(
            source.get("source")
            or "unknown"
        ),
        "snippet": clean_text(
            source.get("snippet")
            or source.get("document")
            or source.get("text")
            or ""
        ),
    }


def unique_sources(
    sources: List[Any],
) -> List[Dict[str, str]]:

    output = []

    seen = set()

    for item in sources:

        source = normalize_source(item)

        key = (
            source["document_id"],
            source["section"],
        )

        if key in seen:
            continue

        seen.add(key)

        output.append(source)

    return output


def clear_chat():

    st.session_state.chat_history = []

    st.session_state.last_result = None

    st.session_state.policy_question = ""

    st.session_state.last_latency_ms = None


def set_question(question: str):

    st.session_state.policy_question = (
        clean_text(question)
    )


def export_response() -> str:

    result = st.session_state.last_result

    if not result:
        return "No response available."

    lines = [
        "POLICYCOPILOT",
        "Enterprise Policy Intelligence",
        "=" * 70,
        "",
        f"Question: {result.get('question', '')}",
        f"Generated: {result.get('timestamp', '')}",
        "",
        "ANSWER",
        "-" * 70,
        result.get("answer", ""),
        "",
        "SOURCES",
        "-" * 70,
    ]

    for source in unique_sources(
        result.get("sources", [])
    ):

        lines.extend(
            [
                f"Document: {source['title']}",
                f"Section: {source['section']}",
                f"Document ID: {source['document_id']}",
                "",
                source["snippet"],
                "",
                "-" * 70,
            ]
        )

    return "\n".join(lines)


def run_question(question: str):

    question = clean_text(question)

    if not question:

        st.warning(
            "Please enter a policy question."
        )

        return

    if len(question) > MAX_QUESTION_LENGTH:

        st.error(
            f"Please keep your question under "
            f"{MAX_QUESTION_LENGTH} characters."
        )

        return

    pipeline, error = get_rag_pipeline()

    if pipeline is None:

        st.error(
            "PolicyCopilot could not initialize "
            "the policy intelligence service."
        )

        with st.expander(
            "Technical diagnostic"
        ):

            st.code(
                error or "Unknown error"
            )

        return

    start = time.perf_counter()

    try:

        with st.spinner(
            "Searching policy knowledge..."
        ):

            raw_result = pipeline.answer(
                question
            )

        latency_ms = round(
            (
                time.perf_counter() - start
            ) * 1000,
            2,
        )

        result = normalize_result(
            raw_result
        )

        answer = result.get(
            "answer",
            ""
        )

        if not answer:

            answer = (
                "I couldn't find sufficient "
                "information in the available "
                "company policy documents to "
                "answer that question."
            )

        result["answer"] = answer

        result["question"] = question

        result["timestamp"] = timestamp()

        result["latency_ms"] = latency_ms

        st.session_state.last_result = result

        st.session_state.last_latency_ms = (
            latency_ms
        )

        st.session_state.total_questions += 1

        # USER MESSAGE
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
                "question": question,
                "timestamp": result["timestamp"],
            }
        )

        # ASSISTANT MESSAGE
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "answer": answer,
                "timestamp": result["timestamp"],
                "sources": result.get(
                    "sources",
                    []
                ),
                "citations": result.get(
                    "citations",
                    []
                ),
                "latency_ms": latency_ms,
            }
        )

    except Exception as exc:

        LOGGER.exception(
            "Question processing failed"
        )

        st.error(
            "PolicyCopilot encountered an "
            "unexpected error while processing "
            "your question."
        )

        with st.expander(
            "Technical diagnostic"
        ):

            st.code(str(exc))


# ============================================================
# TOP NAVIGATION
# ============================================================

nav_left, nav_middle, nav_right = st.columns(
    [4, 4, 2]
)

with nav_left:

    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-title">
                PolicyCopilot
            </div>
            <div class="topbar-subtitle">
                Enterprise Policy Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_middle:

    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-subtitle">
                Workspace
            </div>
            <div class="topbar-title">
                Policy Knowledge Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_right:

    st.markdown(
        """
        <div class="topbar" style="text-align:center;">
            <span class="status-pill">
                ● Policy service
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-row">
                <div class="brand-logo">
                    P
                </div>
                <div>
                    <div class="brand-name">
                        PolicyCopilot
                    </div>
                    <div class="brand-description">
                        Enterprise Policy Intelligence
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Workspace"
    )

    page = st.radio(
        "Workspace",
        [
            "Chat",
            "Knowledge Base",
            "System Status",
            "Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if st.button(
        "＋ New Chat",
        type="primary",
        use_container_width=True,
    ):

        clear_chat()

        st.rerun()

    st.markdown(
        "### Recent Questions"
    )

    recent_questions = [
        item.get(
            "question",
            item.get("content", "")
        )
        for item in st.session_state.chat_history
        if item.get("role") == "user"
    ]

    if recent_questions:

        for index, question in enumerate(
            reversed(
                recent_questions[-5:]
            )
        ):

            label = question[:50]

            if len(question) > 50:
                label += "..."

            if st.button(
                label,
                key=f"recent_{index}",
                use_container_width=True,
            ):

                set_question(question)

                st.rerun()

    else:

        st.caption(
            "Your recent questions will appear here."
        )

    st.divider()

    st.markdown(
        "### AI Controls"
    )

    top_k = st.slider(
        "Retrieved policy chunks",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
    )

    if top_k != st.session_state.top_k:

        st.session_state.top_k = top_k

        load_rag_pipeline.clear()

        st.session_state.rag_initialized = False

    st.caption(
        f"Semantic retrieval • Top-K {top_k}"
    )

    st.divider()

    st.markdown(
        "### System Status"
    )

    if st.session_state.rag_initialized:

        st.success(
            "Policy service available"
        )

    else:

        st.info(
            "Service initializes when queried"
        )

    st.caption(
        f"Session questions: "
        f"{st.session_state.total_questions}"
    )

    st.divider()

    st.markdown(
        """
        **Trust principles**

        ✓ Policy grounded  
        ✓ Source cited  
        ✓ Evidence transparent  
        ✓ No unsupported claims
        """
    )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

if page == "Knowledge Base":

    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-title">
                Policy <span class="hero-title-accent">
                Knowledge Base</span>
            </div>

            <div class="hero-description">
                Explore the policy corpus and retrieval
                configuration powering PolicyCopilot.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Vector database",
            "ChromaDB"
        )

    with c2:
        st.metric(
            "Embedding",
            "MiniLM"
        )

    with c3:
        st.metric(
            "Top-K",
            st.session_state.top_k
        )

    with c4:
        st.metric(
            "Chunk size",
            CHUNK_SIZE
        )

    st.markdown(
        "### Indexed Policy Documents"
    )

    policies_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "policies",
    )

    if os.path.isdir(policies_dir):

        files = sorted(
            file_name
            for file_name in os.listdir(
                policies_dir
            )
            if file_name.lower().endswith(
                (
                    ".md",
                    ".txt",
                    ".html",
                    ".htm",
                    ".pdf",
                )
            )
        )

        if files:

            for file_name in files:

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">
                            📄 {file_name}
                        </div>

                        <div class="source-meta">
                            Policy corpus document
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.warning(
                "No supported policy documents "
                "were found."
            )

    else:

        st.warning(
            "The policies directory was not found."
        )

    st.markdown(
        """
        <div class="footer">
            PolicyCopilot • Knowledge Base
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SYSTEM STATUS
# ============================================================

if page == "System Status":

    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-title">
                System <span class="hero-title-accent">
                Status</span>
            </div>

            <div class="hero-description">
                Runtime visibility for the PolicyCopilot
                retrieval-augmented generation system.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline, error = get_rag_pipeline()

    c1, c2, c3 = st.columns(3)

    with c1:

        if pipeline:

            st.success(
                "RAG Pipeline\n\nAvailable"
            )

        else:

            st.warning(
                "RAG Pipeline\n\nNot initialized"
            )

    with c2:

        st.info(
            f"Vector Retrieval\n\nTop-K = "
            f"{st.session_state.top_k}"
        )

    with c3:

        if os.getenv(
            "OPENROUTER_API_KEY"
        ):

            st.success(
                "LLM Configuration\n\nConfigured"
            )

        else:

            st.warning(
                "LLM Configuration\n\nCheck API key"
            )

    st.markdown(
        "### Architecture Configuration"
    )

    config = {
        "Application": APP_NAME,
        "Embedding model": (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        "Vector database": "ChromaDB",
        "Top-K retrieval": st.session_state.top_k,
        "Chunk size": CHUNK_SIZE,
        "Chunk overlap": CHUNK_OVERLAP,
        "Maximum question length": (
            MAX_QUESTION_LENGTH
        ),
    }

    st.json(config)

    if error:

        st.markdown(
            "### Diagnostic"
        )

        with st.expander(
            "View backend diagnostic"
        ):

            st.code(error)

    st.markdown(
        "### Session Metrics"
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Questions",
            st.session_state.total_questions,
        )

    with m2:

        value = (
            f"{st.session_state.last_latency_ms} ms"
            if st.session_state.last_latency_ms
            else "—"
        )

        st.metric(
            "Last latency",
            value,
        )

    with m3:

        st.metric(
            "Top-K",
            st.session_state.top_k,
        )

    st.markdown(
        """
        <div class="footer">
            Runtime status reflects application
            configuration and service availability.
            It is not a security certification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SETTINGS
# ============================================================

if page == "Settings":

    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-title">
                Application <span class="hero-title-accent">
                Settings</span>
            </div>

            <div class="hero-description">
                Configure retrieval behavior for the
                PolicyCopilot session.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Retrieval Configuration"
    )

    selected_k = st.slider(
        "Top-K retrieved chunks",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
    )

    if st.button(
        "Apply Settings",
        type="primary",
    ):

        st.session_state.top_k = selected_k

        load_rag_pipeline.clear()

        st.session_state.rag_initialized = False

        st.success(
            "Retrieval settings updated."
        )

    st.markdown(
        "### Current Configuration"
    )

    st.json(
        {
            "TOP_K": st.session_state.top_k,
            "CHUNK_SIZE": CHUNK_SIZE,
            "CHUNK_OVERLAP": CHUNK_OVERLAP,
            "MAX_QUESTION_LENGTH": (
                MAX_QUESTION_LENGTH
            ),
        }
    )

    st.markdown(
        "### Conversation"
    )

    if st.button(
        "Clear Conversation"
    ):

        clear_chat()

        st.success(
            "Conversation cleared."
        )

        st.rerun()

    st.markdown(
        """
        <div class="footer">
            PolicyCopilot • Configuration
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-title">
            Ask about your
            <span class="hero-title-accent">
                company policies
            </span>
        </div>

        <div class="hero-description">
            Get accurate answers grounded in your
            organization's policy knowledge base,
            supported by transparent source citations
            and evidence.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CAPABILITY CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                Knowledge
            </div>

            <div class="card-value">
                Policy Grounded
            </div>

            <div class="card-description">
                Answers are based on indexed
                company policy content.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                Evidence
            </div>

            <div class="card-value">
                Source Cited
            </div>

            <div class="card-description">
                Supporting policy documents
                are presented with responses.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                Retrieval
            </div>

            <div class="card-value">
                Semantic Search
            </div>

            <div class="card-description">
                Relevant policy chunks are
                retrieved using embeddings.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:

    st.markdown(
        """
        <div class="card">

            <div class="card-label">
                Transparency
            </div>

            <div class="card-value">
                Evidence First
            </div>

            <div class="card-description">
                Inspect the policy evidence
                behind the generated answer.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown(
    "### Suggested questions"
)

suggestions = [
    "How many vacation days does an employee receive?",
    "What are the requirements for working remotely?",
    "What expenses are eligible for reimbursement?",
    "What should employees do after a security incident?",
    "What is the company travel approval process?",
    "What holidays are observed by the company?",
]

suggestion_columns = st.columns(3)

for index, suggestion in enumerate(
    suggestions
):

    with suggestion_columns[
        index % 3
    ]:

        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):

            set_question(suggestion)

            st.rerun()


# ============================================================
# CHAT CONTAINER
# ============================================================

st.markdown(
    """
    <div class="chat-frame">

        <div class="chat-header">

            <div>

                <div class="chat-title">
                    Policy Assistant
                </div>

                <div class="chat-subtitle">
                    Ask a question about the indexed
                    company policy corpus.
                </div>

            </div>

            <span class="status-pill">
                Top-K retrieval
            </span>

        </div>

    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.chat_history:

    for message in (
        st.session_state.chat_history
    ):

        role = message.get(
            "role"
        )

        content = clean_text(
            message.get(
                "content"
            )
            or message.get(
                "answer"
            )
            or message.get(
                "question"
            )
        )

        message_time = clean_text(
            message.get(
                "timestamp"
            )
        )

        if role == "user":

            st.markdown(
                """
                <div class="message user-message">
                    <div class="message-label">
                        You
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # IMPORTANT:
            # User text is rendered with Streamlit,
            # not inserted into HTML.
            st.write(content)

            if message_time:

                st.markdown(
                    f"""
                    <div class="message-time">
                        {message_time}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        elif role == "assistant":

            st.markdown(
                """
                <div class="message assistant-message">
                    <div class="message-label">
                        PolicyCopilot
                    </div>
                """,
                unsafe_allow_html=True,
            )

            # IMPORTANT:
            # Generated LLM output is NEVER inserted
            # directly into an HTML block.
            st.write(content)

            if message_time:

                st.markdown(
                    f"""
                    <div class="message-time">
                        {message_time}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

else:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                🛡️
            </div>

            <div class="empty-title">
                Your policy conversation starts here
            </div>

            <div class="empty-description">
                Ask about vacation, remote work,
                expenses, security, travel,
                benefits, holidays, conduct,
                or another topic covered by
                the policy knowledge base.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    "<div style='height:14px'></div>",
    unsafe_allow_html=True,
)

input_col, button_col = st.columns(
    [5, 1]
)

with input_col:

    question = st.text_area(
        "Policy question",
        value=st.session_state.policy_question,
        placeholder=(
            "Ask a question about company policies..."
        ),
        height=90,
        max_chars=MAX_QUESTION_LENGTH,
        label_visibility="collapsed",
        key="question_input",
    )

with button_col:

    st.markdown(
        "<div style='height:5px'></div>",
        unsafe_allow_html=True,
    )

    send = st.button(
        "Send",
        type="primary",
        use_container_width=True,
    )

    clear = st.button(
        "Clear",
        use_container_width=True,
    )

    st.caption(
        f"{len(question)} / "
        f"{MAX_QUESTION_LENGTH}"
    )


# ============================================================
# INPUT ACTIONS
# ============================================================

if clear:

    clear_chat()

    st.rerun()


if send:

    st.session_state.policy_question = (
        question
    )

    run_question(question)

    st.rerun()


# ============================================================
# LATEST RESPONSE
# ============================================================

result = st.session_state.last_result

if result:

    st.markdown(
        "<div style='height:14px'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Latest response"
    )

    answer_col, metrics_col = st.columns(
        [4, 1]
    )

    with answer_col:

        st.markdown(
            """
            <div class="card">

                <div class="card-label">
                    AI Response
                </div>

            """,
            unsafe_allow_html=True,
        )

        # NEVER put generated answer into HTML.
        st.write(
            result.get(
                "answer",
                ""
            )
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with metrics_col:

        latency = result.get(
            "latency_ms"
        )

        st.metric(
            "Latency",
            (
                f"{latency} ms"
                if latency is not None
                else "—"
            ),
        )

        sources = unique_sources(
            result.get(
                "sources",
                []
            )
        )

        st.metric(
            "Sources",
            len(sources),
        )


    # ========================================================
    # ACTIONS
    # ========================================================

    st.markdown(
        "### Response actions"
    )

    action1, action2, action3, action4 = (
        st.columns(4)
    )

    with action1:

        if st.button(
            "↻ Regenerate",
            use_container_width=True,
        ):

            original_question = result.get(
                "question",
                ""
            )

            if original_question:

                run_question(
                    original_question
                )

                st.rerun()

    with action2:

        st.download_button(
            "↗ Export",
            data=export_response(),
            file_name=(
                "policycopilot_response.txt"
            ),
            mime="text/plain",
            use_container_width=True,
        )

    with action3:

        st.info(
            f"{len(sources)} source(s)"
        )

    with action4:

        if st.button(
            "Clear Response",
            use_container_width=True,
        ):

            st.session_state.last_result = None

            st.rerun()


    # ========================================================
    # SOURCES
    # ========================================================

    if sources:

        st.markdown(
            "### Sources & evidence"
        )

        for index, source in enumerate(
            sources[:5],
            start=1,
        ):

            st.markdown(
                f"""
                <div class="source-card">

                    <div class="source-title">
                        {index}. {source["title"]}
                    </div>

                    <div class="source-meta">
                        Document ID:
                        {source["document_id"]}
                        &nbsp; • &nbsp;
                        Section:
                        {source["section"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            if source["snippet"]:

                with st.expander(
                    f"View evidence — "
                    f"{source['title']}"
                ):

                    # Source content is rendered
                    # safely with Streamlit.
                    st.write(
                        source["snippet"]
                    )

    else:

        st.warning(
            "No source metadata was returned "
            "with this response."
        )


    # ========================================================
    # CITATIONS
    # ========================================================

    citations = result.get(
        "citations",
        []
    )

    if citations:

        st.markdown(
            "### Citations"
        )

        for citation in citations:

            if isinstance(
                citation,
                dict
            ):

                title = clean_text(
                    citation.get(
                        "title"
                    )
                    or citation.get(
                        "document_id"
                    )
                    or "Policy source"
                )

                section = clean_text(
                    citation.get(
                        "section"
                    )
                    or "Unknown Section"
                )

                st.markdown(
                    f"• **{title}** — {section}"
                )

            else:

                st.markdown(
                    f"• {clean_text(citation)}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PolicyCopilot
        &nbsp;•&nbsp;
        Policy Grounded
        &nbsp;•&nbsp;
        Source Cited
        &nbsp;•&nbsp;
        Evidence Transparent
        <br>
        Enterprise RAG demonstration application
    </div>
    """,
    unsafe_allow_html=True,
)
