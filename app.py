"""
PolicyCopilot - Enterprise Policy RAG Assistant

Streamlit application for a policy-grounded RAG assistant.

Design goals:
- Enterprise-grade UI
- Graceful backend failure handling
- No raw HTML visible to users
- Source/citation transparency
- Retrieval and response latency visibility
- Session-based conversation history
- Safe configuration through environment variables
- Compatible with the existing RAGPipeline.answer() interface

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_NAME = "PolicyCopilot"
APP_SUBTITLE = "Enterprise Policy Intelligence"

TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

MAX_QUESTION_LENGTH = 1000
MAX_DISPLAY_SOURCES = 5

PAGE_TITLE = "PolicyCopilot | Enterprise Policy Intelligence"
PAGE_ICON = "🛡️"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

LOGGER = logging.getLogger("policycopilot")


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENTERPRISE THEME
# ============================================================

st.markdown(
    """
    <style>
        /* ====================================================
           GLOBAL
           ==================================================== */

        :root {
            --pc-blue: #155EEF;
            --pc-deep-blue: #0B3B91;
            --pc-light-blue: #EAF4FF;
            --pc-soft-blue: #F4F8FF;

            --pc-green: #16A34A;
            --pc-light-green: #ECFDF3;

            --pc-white: #FFFFFF;
            --pc-black: #0B1220;
            --pc-gray-700: #334155;
            --pc-gray-600: #64748B;
            --pc-gray-500: #94A3B8;
            --pc-gray-300: #CBD5E1;
            --pc-border: #D9E2EC;
            --pc-background: #F4F7FB;
            --pc-warning: #B45309;
            --pc-warning-bg: #FFF7ED;
            --pc-danger: #B91C1C;
            --pc-danger-bg: #FEF2F2;
        }

        .stApp {
            background:
                linear-gradient(
                    180deg,
                    #F7FAFF 0%,
                    #F4F7FB 45%,
                    #F8FAFC 100%
                );
            color: var(--pc-black);
        }

        .main .block-container {
            max-width: 1500px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        /* ====================================================
           SIDEBAR
           ==================================================== */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #FFFFFF 0%,
                    #F8FBFF 100%
                );

            border-right: 1px solid var(--pc-border);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1rem;
        }

        /* ====================================================
           TOP HEADER
           ==================================================== */

        .pc-header {
            background: var(--pc-white);
            border: 1px solid var(--pc-border);
            border-radius: 18px;
            padding: 14px 20px;
            margin-bottom: 18px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
        }

        .pc-header-title {
            font-size: 1.12rem;
            font-weight: 800;
            color: var(--pc-black);
            letter-spacing: -0.2px;
        }

        .pc-header-subtitle {
            font-size: 0.78rem;
            color: var(--pc-gray-600);
            margin-top: 2px;
        }

        .pc-status {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            background: var(--pc-light-green);
            color: var(--pc-green);
            border: 1px solid #BBF7D0;
        }

        .pc-status-neutral {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            background: var(--pc-light-blue);
            color: var(--pc-deep-blue);
            border: 1px solid #BFDBFE;
        }

        /* ====================================================
           BRAND
           ==================================================== */

        .pc-brand {
            background: linear-gradient(
                135deg,
                #0B3B91 0%,
                #155EEF 100%
            );
            border-radius: 18px;
            padding: 18px;
            color: white;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(21, 94, 239, 0.18);
        }

        .pc-brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .pc-brand-logo {
            width: 44px;
            height: 44px;
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.24);
            font-size: 1.25rem;
            font-weight: 900;
        }

        .pc-brand-name {
            font-size: 1.18rem;
            font-weight: 850;
        }

        .pc-brand-description {
            font-size: 0.74rem;
            opacity: 0.82;
            margin-top: 3px;
        }

        /* ====================================================
           HERO
           ==================================================== */

        .pc-hero {
            background: var(--pc-white);
            border: 1px solid var(--pc-border);
            border-radius: 22px;
            padding: 30px 32px;
            margin-bottom: 18px;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.05);
        }

        .pc-hero-title {
            font-size: 2.05rem;
            line-height: 1.15;
            font-weight: 850;
            color: var(--pc-black);
            letter-spacing: -0.8px;
        }

        .pc-hero-title span {
            color: var(--pc-blue);
        }

        .pc-hero-description {
            max-width: 850px;
            color: var(--pc-gray-600);
            font-size: 0.96rem;
            line-height: 1.6;
            margin-top: 10px;
        }

        /* ====================================================
           CARDS
           ==================================================== */

        .pc-card {
            background: var(--pc-white);
            border: 1px solid var(--pc-border);
            border-radius: 16px;
            padding: 17px;
            height: 100%;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.035);
        }

        .pc-card-title {
            font-size: 0.82rem;
            font-weight: 800;
            color: var(--pc-gray-700);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .pc-card-value {
            font-size: 1.35rem;
            font-weight: 850;
            color: var(--pc-black);
            margin-top: 7px;
        }

        .pc-card-description {
            font-size: 0.75rem;
            color: var(--pc-gray-600);
            margin-top: 3px;
        }

        /* ====================================================
           CHAT FRAME
           ==================================================== */

        .pc-chat-frame {
            background: var(--pc-white);
            border: 1px solid var(--pc-border);
            border-radius: 22px;
            padding: 22px;
            min-height: 400px;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.045);
        }

        .pc-chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 14px;
            border-bottom: 1px solid #E8EEF5;
            margin-bottom: 16px;
        }

        .pc-chat-title {
            font-size: 1rem;
            font-weight: 850;
            color: var(--pc-black);
        }

        .pc-chat-subtitle {
            font-size: 0.76rem;
            color: var(--pc-gray-600);
            margin-top: 3px;
        }

        /* ====================================================
           MESSAGE
           ==================================================== */

        .pc-message {
            border-radius: 16px;
            padding: 14px 16px;
            margin: 10px 0;
            border: 1px solid var(--pc-border);
        }

        .pc-user-message {
            background: var(--pc-soft-blue);
            border-color: #CFE0FF;
        }

        .pc-assistant-message {
            background: #FFFFFF;
        }

        .pc-message-label {
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--pc-gray-600);
            margin-bottom: 5px;
        }

        .pc-message-time {
            font-size: 0.68rem;
            color: var(--pc-gray-500);
            margin-top: 7px;
        }

        /* ====================================================
           SOURCE CARDS
           ==================================================== */

        .pc-source {
            background: #FBFDFF;
            border: 1px solid var(--pc-border);
            border-left: 4px solid var(--pc-blue);
            border-radius: 12px;
            padding: 13px;
            margin-top: 8px;
        }

        .pc-source-title {
            font-size: 0.82rem;
            font-weight: 800;
            color: var(--pc-black);
        }

        .pc-source-meta {
            font-size: 0.7rem;
            color: var(--pc-gray-600);
            margin-top: 3px;
        }

        .pc-source-snippet {
            font-size: 0.75rem;
            line-height: 1.5;
            color: var(--pc-gray-700);
            margin-top: 8px;
        }

        /* ====================================================
           EMPTY STATE
           ==================================================== */

        .pc-empty {
            text-align: center;
            padding: 55px 25px;
            color: var(--pc-gray-600);
        }

        .pc-empty-icon {
            font-size: 2.3rem;
            margin-bottom: 10px;
        }

        .pc-empty-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--pc-black);
        }

        .pc-empty-description {
            max-width: 650px;
            margin: 7px auto 0 auto;
            font-size: 0.82rem;
            line-height: 1.55;
        }

        /* ====================================================
           FOOTER
           ==================================================== */

        .pc-footer {
            text-align: center;
            color: var(--pc-gray-500);
            font-size: 0.7rem;
            padding: 25px 0 5px 0;
        }

        /* ====================================================
           BUTTONS
           ==================================================== */

        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid var(--pc-border);
            transition: all 0.15s ease;
        }

        .stButton > button:hover {
            border-color: var(--pc-blue);
            color: var(--pc-blue);
        }

        /* ====================================================
           INPUTS
           ==================================================== */

        div[data-baseweb="input"] > div {
            border-radius: 12px !important;
            border-color: var(--pc-border) !important;
        }

        textarea {
            border-radius: 12px !important;
        }

        /* ====================================================
           METRICS
           ==================================================== */

        div[data-testid="stMetric"] {
            background: var(--pc-white);
            border: 1px solid var(--pc-border);
            border-radius: 14px;
            padding: 12px;
        }

        /* ====================================================
           EXPANDERS
           ==================================================== */

        div[data-testid="stExpander"] {
            border: 1px solid var(--pc-border);
            border-radius: 12px;
            background: white;
        }

        /* ====================================================
           MOBILE
           ==================================================== */

        @media (max-width: 900px) {
            .pc-hero-title {
                font-size: 1.55rem;
            }

            .pc-hero {
                padding: 22px;
            }

            .pc-chat-frame {
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
    """Initialize all Streamlit session state values safely."""

    defaults = {
        "chat_history": [],
        "last_result": None,
        "policy_question": "",
        "top_k": TOP_K_DEFAULT,
        "rag_error": None,
        "rag_initialized": False,
        "last_latency_ms": None,
        "total_questions": 0,
        "current_page": "Chat",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# SAFE IMPORT / RAG INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_pipeline(top_k: int):
    """
    Lazily initialize the RAG pipeline.

    The import happens inside the function so a missing or broken
    RAG module does not prevent the Streamlit UI from rendering.
    """

    try:
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(top_k=top_k)

        return pipeline, None

    except Exception as exc:
        LOGGER.exception("RAG pipeline initialization failed")

        return None, (
            "The RAG backend could not be initialized. "
            "Please verify the RAG modules, vector database, "
            "embedding model, environment variables, and API configuration."
        )


def get_rag_pipeline():
    """Return cached RAG pipeline and error information."""

    pipeline, error = load_rag_pipeline(st.session_state.top_k)

    st.session_state.rag_initialized = pipeline is not None
    st.session_state.rag_error = error

    return pipeline, error


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def current_timestamp() -> str:
    """Return a human-readable local timestamp."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_text(value: Any) -> str:
    """Safely convert arbitrary values into displayable text."""

    if value is None:
        return ""

    return str(value).strip()


def normalize_result(result: Any) -> Dict[str, Any]:
    """
    Normalize RAG output into a predictable dictionary.

    Supports the existing expected structure:
        {
            "answer": "...",
            "citations": [...],
            "sources": [...]
        }
    """

    if result is None:
        return {
            "answer": "",
            "citations": [],
            "sources": [],
        }

    if isinstance(result, dict):
        return {
            "answer": clean_text(result.get("answer", "")),
            "citations": result.get("citations", []) or [],
            "sources": result.get("sources", []) or [],
        }

    return {
        "answer": clean_text(result),
        "citations": [],
        "sources": [],
    }


def safe_source(source: Any) -> Dict[str, str]:
    """Normalize one source object."""

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
            or source.get("document")
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
            or source.get("title")
            or "unknown"
        ),
        "snippet": clean_text(
            source.get("snippet")
            or source.get("document")
            or source.get("text")
            or ""
        ),
    }


def get_unique_sources(sources: List[Any]) -> List[Dict[str, str]]:
    """Remove duplicate sources while preserving order."""

    unique = []
    seen = set()

    for item in sources:
        source = safe_source(item)

        key = (
            source["document_id"],
            source["section"],
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(source)

    return unique


def set_question(question: str) -> None:
    """Put a suggested question into the input state."""

    st.session_state.policy_question = clean_text(question)


def clear_chat() -> None:
    """Clear current conversation."""

    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.session_state.last_latency_ms = None
    st.session_state.policy_question = ""


def format_answer_for_export() -> str:
    """Create a text export of the current response."""

    result = st.session_state.last_result

    if not result:
        return "No PolicyCopilot response is currently available."

    lines = [
        "POLICYCOPILOT RESPONSE",
        "=" * 70,
        "",
        f"Generated: {current_timestamp()}",
        "",
        "ANSWER",
        "-" * 70,
        clean_text(result.get("answer")),
        "",
        "SOURCES",
        "-" * 70,
    ]

    for source in get_unique_sources(result.get("sources", [])):
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


def answer_question(question: str) -> None:
    """Run the RAG pipeline and update the UI state."""

    question = clean_text(question)

    if not question:
        st.warning("Please enter a policy question.")
        return

    if len(question) > MAX_QUESTION_LENGTH:
        st.error(
            f"Your question is too long. "
            f"Please keep it under {MAX_QUESTION_LENGTH} characters."
        )
        return

    pipeline, error = get_rag_pipeline()

    if pipeline is None:
        st.error(
            "PolicyCopilot could not connect to the RAG service."
        )

        with st.expander("Technical details"):
            st.code(error or "Unknown RAG initialization error")

        return

    start_time = time.perf_counter()

    try:
        with st.spinner("Searching policy knowledge and generating an answer..."):
            raw_result = pipeline.answer(question)

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        result = normalize_result(raw_result)

        if not result["answer"]:
            result["answer"] = (
                "I couldn't find sufficient information in the "
                "available company policy documents to answer that question."
            )

        result["latency_ms"] = latency_ms
        result["timestamp"] = current_timestamp()
        result["question"] = question

        st.session_state.last_result = result
        st.session_state.last_latency_ms = latency_ms
        st.session_state.total_questions += 1

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
                "question": question,
                "timestamp": result["timestamp"],
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "answer": result["answer"],
                "timestamp": result["timestamp"],
                "sources": result.get("sources", []),
                "citations": result.get("citations", []),
                "latency_ms": latency_ms,
            }
        )

    except Exception as exc:
        LOGGER.exception("Question processing failed")

        st.error(
            "PolicyCopilot encountered an error while processing your question."
        )

        with st.expander("Technical details"):
            st.code(str(exc))


# ============================================================
# HEADER
# ============================================================

header_left, header_center, header_right = st.columns(
    [4, 4, 2],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        """
        <div class="pc-header">
            <div class="pc-header-title">PolicyCopilot</div>
            <div class="pc-header-subtitle">
                Enterprise Policy Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_center:
    st.markdown(
        """
        <div class="pc-header">
            <div style="font-size:0.78rem;color:#64748B;">
                Workspace
            </div>
            <div style="font-size:0.92rem;font-weight:800;color:#0B1220;">
                Policy Knowledge Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    status_text = (
        "● Service available"
        if st.session_state.rag_initialized
        else "● Service checking"
    )

    st.markdown(
        f"""
        <div class="pc-header" style="text-align:center;">
            <span class="pc-status-neutral">{status_text}</span>
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
        <div class="pc-brand">
            <div class="pc-brand-row">
                <div class="pc-brand-logo">P</div>
                <div>
                    <div class="pc-brand-name">PolicyCopilot</div>
                    <div class="pc-brand-description">
                        Enterprise Policy Intelligence
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workspace")

    page = st.radio(
        "Workspace navigation",
        [
            "Chat",
            "Knowledge Base",
            "System Status",
            "Settings",
        ],
        label_visibility="collapsed",
        key="current_page",
    )

    st.divider()

    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        clear_chat()
        st.rerun()

    st.markdown("### Recent Questions")

    recent_questions = [
        item.get("question", item.get("content", ""))
        for item in st.session_state.chat_history
        if item.get("role") == "user"
    ]

    if recent_questions:
        for index, question in enumerate(
            reversed(recent_questions[-5:])
        ):
            label = question[:48]
            if len(question) > 48:
                label += "..."

            if st.button(
                label,
                key=f"recent_question_{index}",
                use_container_width=True,
            ):
                set_question(question)
                st.rerun()
    else:
        st.caption("Your recent questions will appear here.")

    st.divider()

    st.markdown("### AI Controls")

    selected_top_k = st.slider(
        "Retrieved documents",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        help="Number of policy chunks retrieved for each question.",
    )

    if selected_top_k != st.session_state.top_k:
        st.session_state.top_k = selected_top_k
        load_rag_pipeline.clear()
        st.session_state.rag_initialized = False

    st.caption(
        f"Top-K retrieval: {st.session_state.top_k}"
    )

    st.divider()

    st.markdown("### System Status")

    if st.session_state.rag_initialized:
        st.success("RAG service available")
    else:
        st.info("RAG service initializes on first question")

    st.caption(
        f"Questions this session: "
        f"{st.session_state.total_questions}"
    )

    st.divider()

    st.markdown(
        """
        **Trust principles**

        • Policy-grounded answers  
        • Source citations  
        • No unsupported policy claims  
        • Transparent evidence  
        • Controlled retrieval
        """
    )


# ============================================================
# KNOWLEDGE BASE PAGE
# ============================================================

if page == "Knowledge Base":

    st.markdown(
        """
        <div class="pc-hero">
            <div class="pc-hero-title">
                Policy <span>Knowledge Base</span>
            </div>
            <div class="pc-hero-description">
                Explore the configuration used by PolicyCopilot for
                policy retrieval and evidence-based responses.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Retrieval", "Vector Search")

    with col2:
        st.metric("Top-K", st.session_state.top_k)

    with col3:
        st.metric("Chunk Size", CHUNK_SIZE)

    with col4:
        st.metric("Overlap", CHUNK_OVERLAP)

    st.markdown("### Retrieval Architecture")

    st.info(
        "PolicyCopilot retrieves semantically relevant policy chunks "
        "from the indexed corpus before generating an answer."
    )

    architecture_cols = st.columns(5)

    architecture = [
        ("01", "Question", "User policy question"),
        ("02", "Embedding", "Semantic representation"),
        ("03", "ChromaDB", "Vector retrieval"),
        ("04", "Context", "Relevant evidence"),
        ("05", "Answer", "Grounded response"),
    ]

    for column, (number, title, description) in zip(
        architecture_cols,
        architecture,
    ):
        with column:
            st.markdown(
                f"""
                <div class="pc-card">
                    <div class="pc-card-title">{number}</div>
                    <div class="pc-card-value" style="font-size:1rem;">
                        {title}
                    </div>
                    <div class="pc-card-description">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Policy Corpus")

    policies_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "policies",
    )

    if os.path.isdir(policies_path):

        policy_files = sorted(
            [
                file_name
                for file_name in os.listdir(policies_path)
                if file_name.lower().endswith(
                    (".md", ".txt", ".html", ".htm", ".pdf")
                )
            ]
        )

        if policy_files:

            for file_name in policy_files:
                full_path = os.path.join(
                    policies_path,
                    file_name,
                )

                try:
                    size_kb = round(
                        os.path.getsize(full_path) / 1024,
                        1,
                    )
                except OSError:
                    size_kb = 0

                st.markdown(
                    f"""
                    <div class="pc-source">
                        <div class="pc-source-title">
                            📄 {file_name}
                        </div>
                        <div class="pc-source-meta">
                            Policy document • {size_kb} KB
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.warning(
                "No policy documents were found in the policies folder."
            )

    else:
        st.warning(
            "The policies directory was not found."
        )

    st.markdown(
        """
        <div class="pc-footer">
            PolicyCopilot • Knowledge Base
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SYSTEM STATUS PAGE
# ============================================================

if page == "System Status":

    st.markdown(
        """
        <div class="pc-hero">
            <div class="pc-hero-title">
                System <span>Status</span>
            </div>
            <div class="pc-hero-description">
                Operational visibility for the PolicyCopilot RAG
                application.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline, error = get_rag_pipeline()

    col1, col2, col3 = st.columns(3)

    with col1:
        if pipeline is not None:
            st.success("RAG Pipeline\n\nAvailable")
        else:
            st.error("RAG Pipeline\n\nUnavailable")

    with col2:
        st.info(
            f"Vector Retrieval\n\nTop-K = {st.session_state.top_k}"
        )

    with col3:
        if os.getenv("OPENROUTER_API_KEY"):
            st.success("LLM Configuration\n\nAPI key configured")
        else:
            st.warning("LLM Configuration\n\nCheck API key")

    st.markdown("### Configuration")

    config_data = {
        "Application": APP_NAME,
        "Embedding model": "all-MiniLM-L6-v2",
        "Vector database": "ChromaDB",
        "Top-K": st.session_state.top_k,
        "Chunk size": CHUNK_SIZE,
        "Chunk overlap": CHUNK_OVERLAP,
        "LLM provider": "OpenRouter / configured generator",
        "Environment": "Configured through environment variables",
    }

    for key, value in config_data.items():
        left, right = st.columns([2, 5])

        with left:
            st.markdown(f"**{key}**")

        with right:
            st.write(value)

    if error:
        st.markdown("### Diagnostics")

        with st.expander("RAG initialization diagnostic"):
            st.code(error)

    st.markdown("### Session Metrics")

    metrics = st.columns(3)

    with metrics[0]:
        st.metric(
            "Questions",
            st.session_state.total_questions,
        )

    with metrics[1]:
        latency = st.session_state.last_latency_ms

        st.metric(
            "Last latency",
            f"{latency} ms" if latency else "—",
        )

    with metrics[2]:
        st.metric(
            "Top-K",
            st.session_state.top_k,
        )

    st.markdown(
        """
        <div class="pc-footer">
            Operational status is based on application configuration
            and runtime checks. It is not a security certification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SETTINGS PAGE
# ============================================================

if page == "Settings":

    st.markdown(
        """
        <div class="pc-hero">
            <div class="pc-hero-title">
                Application <span>Settings</span>
            </div>
            <div class="pc-hero-description">
                Configure retrieval behavior and review the current
                PolicyCopilot environment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Retrieval")

    new_top_k = st.slider(
        "Top-K documents",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
    )

    if st.button(
        "Apply Retrieval Settings",
        type="primary",
    ):
        st.session_state.top_k = new_top_k
        load_rag_pipeline.clear()
        st.session_state.rag_initialized = False

        st.success(
            "Retrieval settings updated. "
            "The RAG pipeline will reinitialize on the next query."
        )

    st.markdown("### Current Configuration")

    st.code(
        json.dumps(
            {
                "TOP_K": st.session_state.top_k,
                "CHUNK_SIZE": CHUNK_SIZE,
                "CHUNK_OVERLAP": CHUNK_OVERLAP,
                "MAX_QUESTION_LENGTH": MAX_QUESTION_LENGTH,
            },
            indent=2,
        ),
        language="json",
    )

    st.markdown("### Session")

    if st.button(
        "Clear Conversation",
        use_container_width=False,
    ):
        clear_chat()
        st.success("Conversation cleared.")
        st.rerun()

    st.markdown(
        """
        <div class="pc-footer">
            PolicyCopilot • Configuration
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# MAIN CHAT HERO
# ============================================================

st.markdown(
    """
    <div class="pc-hero">
        <div class="pc-hero-title">
            Ask about your <span>company policies</span>
        </div>

        <div class="pc-hero-description">
            Get accurate answers grounded in your organization's
            policy knowledge base, supported by transparent source
            citations and evidence.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TRUST / CAPABILITY CARDS
# ============================================================

trust1, trust2, trust3, trust4 = st.columns(4)

with trust1:
    st.markdown(
        """
        <div class="pc-card">
            <div class="pc-card-title">Knowledge</div>
            <div class="pc-card-value">Policy Grounded</div>
            <div class="pc-card-description">
                Responses are generated from indexed policy content.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with trust2:
    st.markdown(
        """
        <div class="pc-card">
            <div class="pc-card-title">Evidence</div>
            <div class="pc-card-value">Source Cited</div>
            <div class="pc-card-description">
                Retrieved documents are displayed with the response.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with trust3:
    st.markdown(
        """
        <div class="pc-card">
            <div class="pc-card-title">Retrieval</div>
            <div class="pc-card-value">Semantic Search</div>
            <div class="pc-card-description">
                Vector retrieval finds relevant policy passages.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with trust4:
    st.markdown(
        """
        <div class="pc-card">
            <div class="pc-card-title">Transparency</div>
            <div class="pc-card-value">Evidence First</div>
            <div class="pc-card-description">
                Users can inspect the supporting policy evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown("### Suggested questions")

suggestion_cols = st.columns(3)

suggestions = [
    "How many vacation days does an employee receive?",
    "What are the requirements for working remotely?",
    "What expenses are eligible for reimbursement?",
    "What should employees do if they suspect a security incident?",
    "What is the company travel approval process?",
    "What holidays are observed by the company?",
]

for index, suggestion in enumerate(suggestions):

    with suggestion_cols[index % 3]:

        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            set_question(suggestion)
            st.rerun()


# ============================================================
# CHAT FRAME
# ============================================================

st.markdown(
    """
    <div class="pc-chat-frame">
        <div class="pc-chat-header">
            <div>
                <div class="pc-chat-title">
                    Policy Assistant
                </div>
                <div class="pc-chat-subtitle">
                    Ask a question about the indexed company policies.
                </div>
            </div>
            <span class="pc-status-neutral">
                Top-K retrieval enabled
            </span>
        </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.chat_history:

    for message in st.session_state.chat_history:

        role = message.get("role")
        content = clean_text(
            message.get("content")
            or message.get("answer")
            or message.get("question")
        )

        timestamp = clean_text(
            message.get("timestamp")
        )

        if role == "user":

            st.markdown(
                f"""
                <div class="pc-message pc-user-message">
                    <div class="pc-message-label">
                        You
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(content)

            if timestamp:
                st.markdown(
                    f"""
                    <div class="pc-message-time">
                        {timestamp}
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
                f"""
                <div class="pc-message pc-assistant-message">
                    <div class="pc-message-label">
                        PolicyCopilot
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(content)

            if timestamp:
                st.markdown(
                    f"""
                    <div class="pc-message-time">
                        {timestamp}
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
        <div class="pc-empty">
            <div class="pc-empty-icon">🛡️</div>
            <div class="pc-empty-title">
                Your policy conversation starts here
            </div>
            <div class="pc-empty-description">
                Ask about vacation, remote work, expenses, security,
                travel, benefits, holidays, conduct, or other topics
                covered by the policy corpus.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INPUT AREA
# ============================================================

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

input_left, input_right = st.columns([5, 1])

with input_left:

    question = st.text_area(
        "Policy question",
        value=st.session_state.policy_question,
        placeholder=(
            "Ask a question about company policies..."
        ),
        height=90,
        max_chars=MAX_QUESTION_LENGTH,
        label_visibility="collapsed",
        key="policy_question_input",
    )

with input_right:

    st.markdown("")

    send_clicked = st.button(
        "Send",
        type="primary",
        use_container_width=True,
    )

    clear_clicked = st.button(
        "Clear",
        use_container_width=True,
    )

    st.caption(
        f"{len(question)} / {MAX_QUESTION_LENGTH}"
    )


# ============================================================
# INPUT ACTIONS
# ============================================================

if clear_clicked:
    clear_chat()
    st.rerun()

if send_clicked:

    st.session_state.policy_question = question

    answer_question(question)

    st.rerun()


# ============================================================
# LAST RESPONSE / EVIDENCE
# ============================================================

result = st.session_state.last_result

if result:

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown("### Latest response")

    answer_col, metric_col = st.columns([4, 1])

    with answer_col:

        st.markdown(
            """
            <div class="pc-card">
                <div class="pc-card-title">
                    AI Response
                </div>
            """,
            unsafe_allow_html=True,
        )

        # IMPORTANT:
        # Use st.write instead of injecting generated answer into HTML.
        st.write(result.get("answer", ""))

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with metric_col:

        latency = result.get("latency_ms")

        st.metric(
            "Latency",
            f"{latency} ms" if latency is not None else "—",
        )

        source_count = len(
            get_unique_sources(
                result.get("sources", [])
            )
        )

        st.metric(
            "Sources",
            source_count,
        )


    # ========================================================
    # ACTIONS
    # ========================================================

    st.markdown("### Response actions")

    action1, action2, action3, action4 = st.columns(4)

    with action1:

        if st.button(
            "↻ Regenerate",
            use_container_width=True,
        ):

            original_question = result.get(
                "question",
                "",
            )

            if original_question:
                answer_question(original_question)
                st.rerun()
            else:
                st.warning(
                    "No original question is available."
                )

    with action2:

        st.download_button(
            "↗ Export",
            data=format_answer_for_export(),
            file_name="policycopilot_response.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with action3:

        source_count = len(
            get_unique_sources(
                result.get("sources", [])
            )
        )

        st.info(
            f"{source_count} source(s)"
        )

    with action4:

        if st.button(
            "▣ Clear Response",
            use_container_width=True,
        ):
            st.session_state.last_result = None
            st.rerun()


    # ========================================================
    # EVIDENCE / SOURCES
    # ========================================================

    sources = get_unique_sources(
        result.get("sources", [])
    )

    if sources:

        st.markdown("### Sources & evidence")

        for index, source in enumerate(
            sources[:MAX_DISPLAY_SOURCES],
            start=1,
        ):

            title = source["title"]
            section = source["section"]
            document_id = source["document_id"]
            snippet = source["snippet"]

            st.markdown(
                f"""
                <div class="pc-source">
                    <div class="pc-source-title">
                        {index}. {title}
                    </div>

                    <div class="pc-source-meta">
                        Document ID: {document_id}
                        &nbsp; • &nbsp;
                        Section: {section}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if snippet:

                with st.expander(
                    f"View evidence — {title}"
                ):

                    st.write(snippet)

        if len(sources) > MAX_DISPLAY_SOURCES:
            st.caption(
                f"{len(sources) - MAX_DISPLAY_SOURCES} "
                "additional sources were retrieved but are not displayed."
            )

    else:

        st.warning(
            "No source metadata was returned with this response. "
            "For production use, every policy answer should include "
            "traceable document metadata."
        )


    # ========================================================
    # CITATIONS
    # ========================================================

    citations = result.get("citations", [])

    if citations:

        st.markdown("### Citations")

        for citation in citations:

            if isinstance(citation, dict):

                citation_title = clean_text(
                    citation.get("title")
                    or citation.get("document_id")
                    or "Policy source"
                )

                citation_section = clean_text(
                    citation.get("section")
                    or "Unknown Section"
                )

                st.markdown(
                    f"- **{citation_title}** — {citation_section}"
                )

            else:

                st.markdown(
                    f"- {clean_text(citation)}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="pc-footer">
        PolicyCopilot &nbsp;•&nbsp;
        Policy Grounded &nbsp;•&nbsp;
        Source Cited &nbsp;•&nbsp;
        Evidence Transparent
        <br>
        Enterprise RAG demonstration application
    </div>
    """,
    unsafe_allow_html=True,
)
