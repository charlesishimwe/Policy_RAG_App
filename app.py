import html
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "PolicyCopilot"
APP_SUBTITLE = "Enterprise Policy Intelligence"

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    str(BASE_DIR / "chroma_db")
)

TOP_K = int(os.getenv("TOP_K", "5"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    ""
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

COLLECTION_NAMES = [
    "policy_docs",
    "policy_documents",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PolicyCopilot | %(levelname)s | %(message)s"
)

logger = logging.getLogger("PolicyCopilot")


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "recent_questions": [],
    "last_sources": [],
    "last_latency": 0.0,
    "last_retrieval_count": 0,
    "pending_question": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
<style>

:root {
    --blue: #155EEF;
    --blue-dark: #0B3B91;
    --blue-light: #EAF4FF;
    --black: #0B1220;
    --white: #FFFFFF;
    --gray-bg: #F5F7FA;
    --gray-border: #E2E8F0;
    --gray-text: #64748B;
}

.stApp {
    background: var(--gray-bg);
    color: var(--black);
}

.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--gray-border);
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 5px 4px 22px 4px;
}

.sidebar-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: linear-gradient(
        135deg,
        #155EEF,
        #0B3B91
    );

    color: white;
    font-weight: 900;
    font-size: 17px;

    box-shadow:
        0 8px 18px rgba(21,94,239,0.20);
}

.sidebar-title {
    font-size: 16px;
    font-weight: 850;
    color: #0B1220;
}

.sidebar-subtitle {
    font-size: 10px;
    color: #64748B;
    margin-top: 2px;
}

.sidebar-section {
    margin-top: 20px;
    margin-bottom: 8px;

    color: #94A3B8;

    font-size: 10px;
    font-weight: 850;

    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.sidebar-status {
    display: flex;
    align-items: center;
    gap: 8px;

    padding: 10px 12px;

    border-radius: 10px;

    background: #F8FAFC;

    border: 1px solid #E2E8F0;

    color: #334155;

    font-size: 10px;
    font-weight: 750;
}

.status-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #155EEF;
}


/* ============================================================
   TOP BAR
   ============================================================ */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 13px 17px;

    background: #FFFFFF;

    border: 1px solid #E2E8F0;

    border-radius: 15px;

    margin-bottom: 18px;

    box-shadow:
        0 5px 20px rgba(15,23,42,0.045);
}

.topbar-left {
    display: flex;
    align-items: center;
    gap: 11px;
}

.topbar-logo {
    width: 38px;
    height: 38px;

    border-radius: 10px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #155EEF;

    color: #FFFFFF;

    font-size: 15px;
    font-weight: 900;
}

.topbar-name {
    font-size: 14px;
    font-weight: 850;
    color: #0B1220;
}

.topbar-subtitle {
    font-size: 10px;
    color: #64748B;
}

.topbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-badge {
    padding: 7px 11px;

    border-radius: 999px;

    background: #F8FAFC;

    border: 1px solid #E2E8F0;

    color: #334155;

    font-size: 10px;
    font-weight: 800;
}

.user-avatar {
    width: 34px;
    height: 34px;

    border-radius: 50%;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #0B1220;

    color: white;

    font-size: 10px;
    font-weight: 850;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;

    padding: 35px;

    border-radius: 21px;

    background:
        linear-gradient(
            135deg,
            #0B3B91 0%,
            #155EEF 60%,
            #246BDE 100%
        );

    color: white;

    margin-bottom: 20px;

    box-shadow:
        0 16px 40px rgba(11,59,145,0.18);
}

.hero-title {
    position: relative;
    z-index: 2;

    font-size: clamp(30px, 4vw, 46px);

    line-height: 1.05;

    font-weight: 900;

    letter-spacing: -0.035em;
}

.hero-description {
    position: relative;
    z-index: 2;

    max-width: 760px;

    margin-top: 15px;

    color: rgba(255,255,255,0.90);

    font-size: 14px;

    line-height: 1.7;
}

.hero-eyebrow {
    display: inline-block;

    padding: 6px 10px;

    border-radius: 999px;

    background: rgba(255,255,255,0.12);

    border: 1px solid rgba(255,255,255,0.18);

    font-size: 9px;
    font-weight: 850;

    text-transform: uppercase;
    letter-spacing: 0.13em;
}

.hero-features {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;

    margin-top: 22px;

    color: rgba(255,255,255,0.85);

    font-size: 10px;
    font-weight: 750;
}


/* ============================================================
   METRICS
   ============================================================ */

.metric-card {
    background: #FFFFFF;

    border: 1px solid #E2E8F0;

    border-radius: 14px;

    padding: 16px;

    min-height: 100px;

    box-shadow:
        0 4px 15px rgba(15,23,42,0.035);
}

.metric-label {
    color: #64748B;

    font-size: 9px;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 0.1em;
}

.metric-value {
    color: #0B1220;

    font-size: 23px;

    font-weight: 900;

    margin-top: 8px;
}

.metric-note {
    color: #94A3B8;

    font-size: 10px;
}


/* ============================================================
   CHAT
   ============================================================ */

.chat-container {
    background: #FFFFFF;

    border: 1px solid #E2E8F0;

    border-radius: 19px;

    overflow: hidden;

    box-shadow:
        0 8px 28px rgba(15,23,42,0.045);
}

.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 16px 19px;

    border-bottom: 1px solid #EEF2F7;
}

.chat-title {
    font-size: 14px;
    font-weight: 850;
    color: #0B1220;
}

.chat-subtitle {
    font-size: 10px;
    color: #64748B;
    margin-top: 3px;
}


/* ============================================================
   MESSAGES
   ============================================================ */

.message-row {
    display: flex;
    gap: 11px;

    padding: 18px 20px;

    border-bottom: 1px solid #F1F5F9;
}

.message-avatar {
    flex: 0 0 auto;

    width: 34px;
    height: 34px;

    border-radius: 10px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 10px;
    font-weight: 900;
}

.ai-avatar {
    background: #EAF4FF;
    color: #155EEF;
}

.user-avatar-small {
    background: #0B1220;
    color: #FFFFFF;
}

.message-content {
    flex: 1;
}

.message-name {
    color: #0B1220;

    font-size: 11px;
    font-weight: 850;
}

.message-time {
    color: #94A3B8;

    font-size: 9px;

    margin-left: 7px;
}

.message-text {
    color: #334155;

    font-size: 13px;

    line-height: 1.7;

    margin-top: 5px;
}


/* ============================================================
   SOURCE
   ============================================================ */

.source-card {
    padding: 13px;

    margin-bottom: 9px;

    background: #F8FAFC;

    border: 1px solid #E2E8F0;

    border-radius: 12px;
}

.source-title {
    color: #0B1220;

    font-size: 11px;

    font-weight: 850;
}

.source-meta {
    color: #64748B;

    font-size: 9px;

    margin-top: 3px;
}

.source-snippet {
    margin-top: 9px;

    padding: 9px;

    background: #FFFFFF;

    border: 1px solid #E8EEF5;

    border-radius: 8px;

    color: #475569;

    font-size: 10px;

    line-height: 1.6;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 39px;

    border-radius: 9px;

    border: 1px solid #D7E1EC;

    background: #FFFFFF;

    color: #1E293B;

    font-size: 11px;

    font-weight: 750;
}

.stButton > button:hover {
    border-color: #155EEF;

    color: #155EEF;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #155EEF,
        #0B3B91
    );

    color: white;

    border: none;
}


/* ============================================================
   TEXT AREA
   ============================================================ */

.stTextArea textarea {
    background: #FFFFFF !important;

    color: #0B1220 !important;

    border: 1px solid #D7E1EC !important;

    border-radius: 12px !important;

    font-size: 12px !important;
}

.stTextArea textarea:focus {
    border-color: #155EEF !important;

    box-shadow:
        0 0 0 3px rgba(21,94,239,0.09) !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.app-footer {
    text-align: center;

    padding: 25px 0 5px;

    color: #94A3B8;

    font-size: 9px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    return str(value).strip()


def safe_html(value: Any) -> str:

    return html.escape(
        clean_text(value)
    )


def clean_document_text(text: str) -> str:

    text = clean_text(text)

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# CHROMA
# ============================================================

@st.cache_resource(show_spinner=False)
def get_chroma_client():

    return chromadb.PersistentClient(
        path=CHROMA_PATH
    )


def get_collection():

    try:

        client = get_chroma_client()

        collections = client.list_collections()

        names = []

        for collection in collections:

            try:
                names.append(collection.name)
            except Exception:
                pass

        for name in COLLECTION_NAMES:

            if name in names:

                return client.get_collection(name)

        return None

    except Exception as exc:

        logger.exception(
            "Chroma error: %s",
            exc
        )

        return None


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question: str
) -> List[Dict[str, Any]]:

    if not question.strip():
        return []

    collection = get_collection()

    if collection is None:
        return []

    try:

        model = get_embedding_model()

        embedding = model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        result = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result.get(
            "documents",
            [[]]
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]]
        )[0]

        distances = result.get(
            "distances",
            [[]]
        )[0]

        output = []

        for i, document in enumerate(
            documents
        ):

            output.append(
                {
                    "text": clean_document_text(
                        document
                    ),
                    "metadata":
                        metadatas[i]
                        if i < len(metadatas)
                        else {},
                    "distance":
                        distances[i]
                        if i < len(distances)
                        else None,
                }
            )

        return [
            item
            for item in output
            if len(item["text"]) >= 20
        ]

    except Exception as exc:

        logger.exception(
            "Retrieval failed: %s",
            exc
        )

        return []


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    documents: List[Dict[str, Any]]
) -> str:

    parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
        )

        title = (
            metadata.get("title")
            or metadata.get("document_title")
            or metadata.get("source")
            or "Policy Document"
        )

        section = (
            metadata.get("section")
            or metadata.get("section_title")
            or "Policy Section"
        )

        document_id = (
            metadata.get("document_id")
            or metadata.get("id")
            or metadata.get("source")
            or f"POLICY-{index:03d}"
        )

        parts.append(
            f"""
SOURCE {index}

Document ID:
{document_id}

Policy:
{title}

Section:
{section}

Evidence:
{document["text"]}
""".strip()
        )

    return "\n\n--------------------\n\n".join(
        parts
    )


# ============================================================
# LLM
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> Optional[str]:

    if not OPENROUTER_API_KEY:
        return None

    system_prompt = """
You are PolicyCopilot.

You answer ONLY from the supplied policy context.

Rules:
- Never use outside knowledge.
- Never invent policy.
- If the answer is not supported, clearly state that
  the policy corpus does not provide enough information.
- Always cite the relevant policy document.
- Include the policy section when available.
- Keep answers concise and professional.
- Do not fabricate citations.
"""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "POLICY CONTEXT:\n\n"
                    + context
                    + "\n\nQUESTION:\n\n"
                    + question
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 700,
    }

    headers = {
        "Authorization":
            f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":
            "application/json",
        "HTTP-Referer":
            "https://github.com/charlesishimwe/Policy_RAG_App",
        "X-Title":
            "PolicyCopilot",
    }

    try:

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:

            logger.warning(
                "LLM HTTP %s",
                response.status_code
            )

            return None

        data = response.json()

        choices = data.get(
            "choices",
            []
        )

        if not choices:
            return None

        answer = (
            choices[0]
            .get("message", {})
            .get("content")
        )

        return (
            clean_text(answer)
            if answer
            else None
        )

    except Exception as exc:

        logger.exception(
            "Generation failed: %s",
            exc
        )

        return None


# ============================================================
# FALLBACK
# ============================================================

def fallback_answer(
    question: str,
    documents: List[Dict[str, Any]]
) -> str:

    if not documents:

        return (
            "I can only answer questions covered by "
            "the company policy corpus. No relevant "
            "policy information was found."
        )

    question_words = set(
        word.lower()
        for word in re.findall(
            r"[A-Za-zÀ-ÿ0-9]+",
            question
        )
        if len(word) > 2
    )

    matches = []

    for document in documents:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            document["text"]
        )

        for sentence in sentences:

            sentence_words = set(
                word.lower()
                for word in re.findall(
                    r"[A-Za-zÀ-ÿ0-9]+",
                    sentence
                )
                if len(word) > 2
            )

            score = len(
                question_words
                &
                sentence_words
            )

            if score:
                matches.append(
                    (
                        score,
                        sentence
                    )
                )

    matches.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    selected = []

    for _, sentence in matches:

        if sentence not in selected:

            selected.append(
                sentence
            )

        if len(selected) == 3:
            break

    if not selected:

        selected = [
            documents[0]["text"][:700]
        ]

    return (
        "Based on the retrieved policy evidence:\n\n"
        + " ".join(selected)
    )


# ============================================================
# SOURCES
# ============================================================

def build_sources(
    documents: List[Dict[str, Any]]
) -> List[Dict[str, str]]:

    sources = []

    seen = set()

    for document in documents:

        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
        )

        title = (
            metadata.get("title")
            or metadata.get("document_title")
            or metadata.get("source")
            or "Policy Document"
        )

        section = (
            metadata.get("section")
            or metadata.get("section_title")
            or "Policy Section"
        )

        document_id = (
            metadata.get("document_id")
            or metadata.get("id")
            or metadata.get("source")
            or "POLICY"
        )

        key = (
            str(document_id),
            str(section)
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "document_id":
                    clean_text(document_id),
                "title":
                    clean_text(title),
                "section":
                    clean_text(section),
                "snippet":
                    clean_text(
                        document["text"]
                    )[:800],
            }
        )

    return sources


# ============================================================
# PIPELINE
# ============================================================

def ask_policy(
    question: str
) -> Dict[str, Any]:

    start = time.perf_counter()

    documents = retrieve(
        question
    )

    if not documents:

        return {
            "answer":
                "I can only answer questions covered by "
                "the company policy corpus. I could not "
                "find relevant policy information.",
            "sources": [],
            "latency":
                time.perf_counter() - start,
            "retrieval_count": 0,
        }

    context = build_context(
        documents
    )

    answer = generate_answer(
        question,
        context
    )

    if not answer:

        answer = fallback_answer(
            question,
            documents
        )

    return {
        "answer": answer,
        "sources":
            build_sources(documents),
        "latency":
            time.perf_counter() - start,
        "retrieval_count":
            len(documents),
    }


# ============================================================
# STATUS
# ============================================================

def get_status():

    collection = get_collection()

    if collection is None:

        return {
            "connected": False,
            "chunks": 0,
        }

    try:

        return {
            "connected": True,
            "chunks": collection.count(),
        }

    except Exception:

        return {
            "connected": True,
            "chunks": 0,
        }


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    """
<div class="topbar">

    <div class="topbar-left">

        <div class="topbar-logo">
            AI
        </div>

        <div>

            <div class="topbar-name">
                PolicyCopilot
            </div>

            <div class="topbar-subtitle">
                Enterprise Policy Intelligence
            </div>

        </div>

    </div>

    <div class="topbar-right">

        <div class="status-badge">
            Knowledge service
        </div>

        <div class="user-avatar">
            AI
        </div>

    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">

    <div class="sidebar-logo">
        AI
    </div>

    <div>

        <div class="sidebar-title">
            PolicyCopilot
        </div>

        <div class="sidebar-subtitle">
            Policy Intelligence Platform
        </div>

    </div>

</div>
""",
        unsafe_allow_html=True
    )

    if st.button(
        "＋  New conversation",
        type="primary",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown(
        '<div class="sidebar-section">Workspace</div>',
        unsafe_allow_html=True
    )

    workspace = st.radio(
        "Workspace",
        [
            "Assistant",
            "Knowledge base",
            "Settings",
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-section">Recent questions</div>',
        unsafe_allow_html=True
    )

    if st.session_state.recent_questions:

        for index, question in enumerate(
            st.session_state.recent_questions[:6]
        ):

            label = question[:40]

            if len(question) > 40:
                label += "…"

            if st.button(
                label,
                key=f"recent_{index}",
                use_container_width=True
            ):

                st.session_state.pending_question = (
                    question
                )

                st.rerun()

    else:

        st.caption(
            "Your recent questions will appear here."
        )

    st.markdown(
        '<div class="sidebar-section">System</div>',
        unsafe_allow_html=True
    )

    status = get_status()

    if status["connected"]:

        st.markdown(
            """
<div class="sidebar-status">

<span class="status-dot"></span>

Knowledge base connected

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
<div class="sidebar-status">

<span
class="status-dot"
style="background:#64748B;">
</span>

Knowledge base unavailable

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# SETTINGS
# ============================================================

if workspace == "Settings":

    st.title("Settings")

    st.write(
        "PolicyCopilot application configuration."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Top-K",
            TOP_K
        )

    with c2:
        st.metric(
            "Embedding",
            "MiniLM"
        )

    with c3:
        st.metric(
            "Generation",
            "OpenRouter"
            if OPENROUTER_API_KEY
            else "Local"
        )

    st.stop()


# ============================================================
# KNOWLEDGE BASE
# ============================================================

if workspace == "Knowledge base":

    st.title("Knowledge base")

    status = get_status()

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "Knowledge status",
            "Connected"
            if status["connected"]
            else "Unavailable"
        )

    with c2:

        st.metric(
            "Indexed chunks",
            f"{status['chunks']:,}"
        )

    st.markdown(
        """
<div class="metric-card">

<div class="metric-label">
Architecture
</div>

<div class="metric-value">
RAG
</div>

<div class="metric-note">
Semantic retrieval + grounded generation
</div>

</div>
""",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# HERO
# ============================================================

# IMPORTANT:
# The HTML tags below are rendered by Streamlit.
# Customers see ONLY the visual result.
# They do NOT see the HTML source code.

st.markdown(
    """
<div class="hero">

    <div class="hero-eyebrow">
        AI-powered policy intelligence
    </div>

    <div class="hero-title">
        Ask your policies.<br>
        Get grounded answers.
    </div>

    <div class="hero-description">
        Get accurate answers grounded in your organization's
        policy knowledge base, supported by transparent source
        citations and evidence.
    </div>

    <div class="hero-features">
        <span>✓ Grounded answers</span>
        <span>✓ Source citations</span>
        <span>✓ Semantic retrieval</span>
        <span>✓ Enterprise-ready</span>
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# METRICS
# ============================================================

status = get_status()

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
Policy chunks
</div>

<div class="metric-value">
{status["chunks"]:,}
</div>

<div class="metric-note">
Indexed knowledge
</div>

</div>
""",
        unsafe_allow_html=True
    )

with m2:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
Retrieval depth
</div>

<div class="metric-value">
{TOP_K}
</div>

<div class="metric-note">
Top policy matches
</div>

</div>
""",
        unsafe_allow_html=True
    )

with m3:

    latency = (
        f"{st.session_state.last_latency:.2f}s"
        if st.session_state.last_latency
        else "—"
    )

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
Last response
</div>

<div class="metric-value">
{latency}
</div>

<div class="metric-note">
End-to-end latency
</div>

</div>
""",
        unsafe_allow_html=True
    )

with m4:

    st.markdown(
        f"""
<div class="metric-card">

<div class="metric-label">
Generation
</div>

<div class="metric-value">
{"AI" if OPENROUTER_API_KEY else "Local"}
</div>

<div class="metric-note">
{"OpenRouter" if OPENROUTER_API_KEY else "Fallback mode"}
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="metric-label">
Suggested questions
</div>
""",
    unsafe_allow_html=True
)

suggestions = [
    "How many vacation days do employees receive?",
    "What is the remote work policy?",
    "Which business expenses can employees claim?",
    "What are the company's security requirements?",
]

cols = st.columns(4)

for index, suggestion in enumerate(
    suggestions
):

    with cols[index]:

        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                suggestion
            )

            st.rerun()


# ============================================================
# CHAT
# ============================================================

st.markdown(
    """
<div class="chat-container">

<div class="chat-header">

<div>

<div class="chat-title">
Policy Assistant
</div>

<div class="chat-subtitle">
Answers are grounded in the indexed policy knowledge base.
</div>

</div>

<div class="status-badge">
Ready
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
<div class="message-row">

<div class="message-avatar ai-avatar">
AI
</div>

<div class="message-content">

<div class="message-name">
PolicyCopilot
<span class="message-time">
Ready
</span>
</div>

<div class="message-text">
Welcome to PolicyCopilot. Ask a question about company
policies and I will retrieve the relevant policy evidence
before providing a grounded answer.
</div>

</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY MESSAGES
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant"
    )

    content = message.get(
        "content",
        ""
    )

    if role == "user":

        st.markdown(
            f"""
<div class="message-row">

<div class="message-avatar user-avatar-small">
U
</div>

<div class="message-content">

<div class="message-name">
You
</div>

<div class="message-text">
{safe_html(content)}
</div>

</div>

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
<div class="message-row">

<div class="message-avatar ai-avatar">
AI
</div>

<div class="message-content">

<div class="message-name">
PolicyCopilot
</div>

<div class="message-text">
""",
            unsafe_allow_html=True
        )

        st.markdown(
            content
        )

        st.markdown(
            """
</div>
</div>
</div>
""",
            unsafe_allow_html=True
        )

        sources = message.get(
            "sources",
            []
        )

        if sources:

            with st.expander(
                f"View {len(sources)} source"
                + (
                    "s"
                    if len(sources) != 1
                    else ""
                )
            ):

                for source in sources:

                    st.markdown(
                        f"""
<div class="source-card">

<div class="source-title">
{safe_html(source["title"])}
</div>

<div class="source-meta">
{safe_html(source["document_id"])}
•
{safe_html(source["section"])}
</div>

<div class="source-snippet">
{safe_html(source["snippet"])}
</div>

</div>
""",
                        unsafe_allow_html=True
                    )


# ============================================================
# CLOSE CHAT
# ============================================================

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

pending = st.session_state.pop(
    "pending_question",
    None
)

question = st.text_area(
    "Policy question",
    value=pending or "",
    placeholder=(
        "Ask about vacation, remote work, expenses, "
        "security, travel, benefits, or another policy..."
    ),
    height=90,
    label_visibility="collapsed"
)


# ============================================================
# ACTION BUTTONS
# ============================================================

c1, c2, c3 = st.columns(
    [1, 1, 6]
)

with c1:

    clear = st.button(
        "Clear",
        use_container_width=True
    )

with c2:

    export = st.button(
        "Export",
        use_container_width=True
    )

with c3:

    send = st.button(
        "Send question  →",
        type="primary",
        use_container_width=True
    )


# ============================================================
# CLEAR
# ============================================================

if clear:

    st.session_state.messages = []

    st.session_state.last_sources = []

    st.session_state.last_latency = 0

    st.session_state.last_retrieval_count = 0

    st.rerun()


# ============================================================
# EXPORT
# ============================================================

if export:

    if not st.session_state.messages:

        st.info(
            "There is no conversation to export yet."
        )

    else:

        lines = [
            "PolicyCopilot Conversation",
            "==========================",
            "",
        ]

        for message in st.session_state.messages:

            role = (
                "USER"
                if message["role"] == "user"
                else "POLICYCOPILOT"
            )

            lines.append(role)

            lines.append(
                message["content"]
            )

            lines.append("")

        st.download_button(
            "Download conversation",
            data="\n".join(lines),
            file_name="policycopilot_conversation.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# SEND QUESTION
# ============================================================

if send:

    question = clean_text(
        question
    )

    if not question:

        st.warning(
            "Please enter a policy question."
        )

    else:

        st.session_state.recent_questions.insert(
            0,
            question
        )

        st.session_state.recent_questions = list(
            dict.fromkeys(
                st.session_state.recent_questions
            )
        )[:10]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
                "timestamp": time.time(),
            }
        )

        with st.spinner(
            "Searching policy knowledge..."
        ):

            result = ask_policy(
                question
            )

        st.session_state.last_latency = (
            result["latency"]
        )

        st.session_state.last_sources = (
            result["sources"]
        )

        st.session_state.last_retrieval_count = (
            result["retrieval_count"]
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "timestamp": time.time(),
            }
        )

        st.rerun()


# ============================================================
# RESPONSE DETAILS
# ============================================================

if st.session_state.messages:

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    with st.expander(
        "Response details"
    ):

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Response time",
                f"{st.session_state.last_latency:.2f}s"
            )

        with c2:

            st.metric(
                "Sources retrieved",
                st.session_state.last_retrieval_count
            )

        with c3:

            st.metric(
                "Retrieval",
                f"Top-{TOP_K}"
            )


# ============================================================
# TRUST CARDS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
<div class="metric-card">

<div class="metric-label">
Semantic retrieval
</div>

<div class="metric-value">
RAG
</div>

<div class="metric-note">
Natural-language policy search
</div>

</div>
""",
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
<div class="metric-card">

<div class="metric-label">
Evidence first
</div>

<div class="metric-value">
Grounded
</div>

<div class="metric-note">
Relevant evidence before generation
</div>

</div>
""",
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
<div class="metric-card">

<div class="metric-label">
Transparency
</div>

<div class="metric-value">
Citations
</div>

<div class="metric-note">
Policy sources accompany answers
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="app-footer">

PolicyCopilot • Enterprise Policy Intelligence

<br>

Grounded AI • Semantic Retrieval • Transparent Evidence

</div>
""",
    unsafe_allow_html=True
)
