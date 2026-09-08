```python
"""
Policy RAG Assistant
--------------------
Streamlit application for the Quantic AI Engineering Project.

Architecture:
    User Question
        ↓
    SentenceTransformer Embedding
        ↓
    ChromaDB Retrieval
        ↓
    Grounded Prompt
        ↓
    OpenAI / OpenRouter / Groq-compatible LLM
        ↓
    Answer + Citations + Evidence

Designed to fail gracefully instead of crashing the Streamlit application.
"""

from __future__ import annotations

import os
import time
import html
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

# -------------------------------------------------------------------
# OPTIONAL / SAFE IMPORTS
# -------------------------------------------------------------------

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

APP_TITLE = "Policy RAG Assistant"

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "chroma_db",
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "policy_docs",
)

POLICY_DIR = Path(
    os.getenv(
        "POLICY_DIR",
        "policies",
    )
)

TOP_K = int(
    os.getenv(
        "TOP_K",
        "4",
    )
)

MAX_ANSWER_CHARS = 3500


# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# CUSTOM UI
# -------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* -------------------------------------------------------------
       GLOBAL
    ------------------------------------------------------------- */

    .stApp {
        background: #f5f7fa;
        color: #172033;
    }

    .main {
        background: #f5f7fa;
    }

    /* -------------------------------------------------------------
       HEADER
       ------------------------------------------------------------- */

    .hero {
        background: linear-gradient(
            135deg,
            #0b3d91 0%,
            #1565c0 55%,
            #1976d2 100%
        );

        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 24px;

        box-shadow:
            0 8px 25px rgba(11, 61, 145, 0.15);
    }

    .hero-title {
        color: white;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #e8f1ff;
        font-size: 15px;
    }

    /* -------------------------------------------------------------
       CARDS
       ------------------------------------------------------------- */

    .card {
        background: white;
        border: 1px solid #dfe5ec;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 14px;

        box-shadow:
            0 3px 12px rgba(20, 35, 55, 0.06);
    }

    .source-card {
        background: #f8fafc;
        border-left: 4px solid #1565c0;
        border-radius: 10px;
        padding: 14px;
        margin-top: 10px;
    }

    .metric-card {
        background: white;
        border: 1px solid #dfe5ec;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }

    .metric-label {
        color: #687386;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .metric-value {
        color: #0b3d91;
        font-size: 25px;
        font-weight: 800;
    }

    /* -------------------------------------------------------------
       CHAT
       ------------------------------------------------------------- */

    .answer-box {
        background: white;
        border: 1px solid #d8e0ea;
        border-radius: 15px;
        padding: 22px;
        line-height: 1.65;

        box-shadow:
            0 4px 15px rgba(20, 35, 55, 0.06);
    }

    .question-box {
        background: #eaf2ff;
        border: 1px solid #c9dcfa;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 14px;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 5px;
    }

    .badge-blue {
        background: #e3efff;
        color: #0b3d91;
    }

    .badge-gray {
        background: #edf0f3;
        color: #4d5968;
    }

    /* -------------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dfe5ec;
    }

    /* -------------------------------------------------------------
       BUTTONS
       ------------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #0b3d91;
        font-weight: 700;
    }

    /* -------------------------------------------------------------
       FOOTER
       ------------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #7b8491;
        font-size: 12px;
        padding: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# SAFE HELPERS
# -------------------------------------------------------------------

def safe_text(value: Any) -> str:
    """Convert arbitrary values to safe text."""
    try:
        return str(value or "").strip()
    except Exception:
        return ""


def safe_error_message(exc: Exception) -> str:
    """Return a user-friendly error without exposing secrets."""
    message = safe_text(exc)

    if len(message) > 500:
        message = message[:500] + "..."

    return message or "Unknown application error."


# -------------------------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load the Sentence Transformer once and cache it.
    """

    if SentenceTransformer is None:
        return None

    try:
        return SentenceTransformer(MODEL_NAME)

    except Exception:
        return None


# -------------------------------------------------------------------
# CHROMA
# -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_collection():
    """
    Connect safely to the existing ChromaDB collection created
    by ingest.py.
    """

    if chromadb is None:
        return None

    try:

        Path(CHROMA_PATH).mkdir(
            parents=True,
            exist_ok=True,
        )

        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=(
                Settings(anonymized_telemetry=False)
                if Settings
                else None
            ),
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME
        )

        return collection

    except Exception:
        return None


# -------------------------------------------------------------------
# DATABASE STATUS
# -------------------------------------------------------------------

def get_collection_count(collection) -> int:

    if collection is None:
        return 0

    try:
        return int(collection.count())

    except Exception:
        return 0


# -------------------------------------------------------------------
# RETRIEVAL
# -------------------------------------------------------------------

def retrieve_documents(
    question: str,
    model,
    collection,
    k: int = TOP_K,
) -> List[Dict[str, Any]]:

    if not question.strip():
        return []

    if model is None:
        return []

    if collection is None:
        return []

    try:

        embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = (
            results.get("documents", [[]])[0]
            or []
        )

        metadata = (
            results.get("metadatas", [[]])[0]
            or []
        )

        distances = (
            results.get("distances", [[]])[0]
            or []
        )

        retrieved = []

        for index, document in enumerate(documents):

            meta = (
                metadata[index]
                if index < len(metadata)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            retrieved.append(
                {
                    "document": safe_text(document),
                    "source": safe_text(
                        meta.get(
                            "source",
                            "Unknown source",
                        )
                    ),
                    "chunk_id": meta.get(
                        "chunk_id",
                        index,
                    ),
                    "distance": distance,
                }
            )

        return retrieved

    except Exception:
        return []


# -------------------------------------------------------------------
# RELEVANCE CHECK
# -------------------------------------------------------------------

def has_relevant_context(
    documents: List[Dict[str, Any]],
) -> bool:

    if not documents:
        return False

    valid = 0

    for item in documents:

        text = safe_text(
            item.get("document")
        )

        if len(text) >= 40:
            valid += 1

    return valid > 0


# -------------------------------------------------------------------
# PROMPT
# -------------------------------------------------------------------

def build_prompt(
    question: str,
    documents: List[Dict[str, Any]],
) -> str:

    context_blocks = []

    for index, item in enumerate(documents, start=1):

        source = item.get(
            "source",
            "Unknown",
        )

        chunk_id = item.get(
            "chunk_id",
            index,
        )

        text = item.get(
            "document",
            "",
        )

        context_blocks.append(
            f"""
SOURCE {index}
Document: {source}
Chunk: {chunk_id}

{text}
"""
        )

    context = "\n".join(context_blocks)

    return f"""
You are a strict company-policy RAG assistant.

Your job is to answer ONLY from the supplied policy evidence.

RULES:

1. Use ONLY the context below.
2. Never invent policy information.
3. Never use outside knowledge.
4. If the answer cannot be supported by the context, say:
   "I could not find this information in the policy documents."
5. Keep the answer concise and professional.
6. Every factual answer must include citations using:
   [Source: filename]
7. If multiple documents support the answer, cite each relevant source.
8. Do not cite a document that does not support the statement.
9. Do not make assumptions.
10. Maximum answer length: 250 words.

POLICY CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# -------------------------------------------------------------------
# LLM CLIENT
# -------------------------------------------------------------------

def get_llm_configuration():

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ),
            "model": os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini",
            ),
            "provider": "OpenAI",
        }

    # OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        return {
            "api_key": os.getenv(
                "OPENROUTER_API_KEY"
            ),
            "base_url": (
                "https://openrouter.ai/api/v1"
            ),
            "model": os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-4o-mini",
            ),
            "provider": "OpenRouter",
        }

    # Groq
    if os.getenv("GROQ_API_KEY"):
        return {
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": (
                "https://api.groq.com/openai/v1"
            ),
            "model": os.getenv(
                "GROQ_MODEL",
                "llama-3.1-8b-instant",
            ),
            "provider": "Groq",
        }

    return None


# -------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------

def generate_answer(
    question: str,
    documents: List[Dict[str, Any]],
) -> Tuple[str, str]:

    if not has_relevant_context(documents):

        return (
            "I could not find this information in the policy documents.",
            "No relevant evidence",
        )

    configuration = get_llm_configuration()

    if configuration is None:

        return (
            "The policy evidence was retrieved successfully, "
            "but no LLM API key is configured. "
            "Please configure OPENAI_API_KEY, "
            "OPENROUTER_API_KEY, or GROQ_API_KEY.",
            "LLM not configured",
        )

    if OpenAI is None:

        return (
            "The OpenAI-compatible client is not installed. "
            "Please run: pip install openai",
            "LLM client unavailable",
        )

    try:

        client = OpenAI(
            api_key=configuration["api_key"],
            base_url=configuration["base_url"],
        )

        response = client.chat.completions.create(
            model=configuration["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded policy assistant. "
                        "Never hallucinate."
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(
                        question,
                        documents,
                    ),
                },
            ],
            temperature=0,
            max_tokens=500,
        )

        answer = safe_text(
            response.choices[0].message.content
        )

        if not answer:

            return (
                "The model returned an empty response.",
                configuration["provider"],
            )

        return (
            answer[:MAX_ANSWER_CHARS],
            configuration["provider"],
        )

    except Exception as exc:

        return (
            "I was unable to contact the language model. "
            "The application is still running. "
            "Please verify your API configuration.",
            f"{configuration['provider']} unavailable",
        )


# -------------------------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------------------------

def initialize_session():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_latency" not in st.session_state:
        st.session_state.last_latency = None


# -------------------------------------------------------------------
# DISPLAY SOURCES
# -------------------------------------------------------------------

def display_sources(
    documents: List[Dict[str, Any]],
):

    if not documents:
        return

    st.markdown(
        "### 📚 Retrieved evidence"
    )

    for index, item in enumerate(
        documents,
        start=1,
    ):

        source = html.escape(
            safe_text(
                item.get(
                    "source",
                    "Unknown source",
                )
            )
        )

        chunk_id = html.escape(
            safe_text(
                item.get(
                    "chunk_id",
                    index,
                )
            )
        )

        text = html.escape(
            safe_text(
                item.get(
                    "document",
                    "",
                )
            )
        )

        preview = text[:700]

        if len(text) > 700:
            preview += "..."

        st.markdown(
            f"""
            <div class="source-card">

                <strong>Source {index}</strong>

                <br>

                <span class="badge badge-blue">
                    {source}
                </span>

                <span class="badge badge-gray">
                    Chunk {chunk_id}
                </span>

                <p>{preview}</p>

            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

def render_sidebar(
    model,
    collection,
):

    st.sidebar.markdown(
        """
        <div style="
            color:#0b3d91;
            font-size:23px;
            font-weight:800;
            margin-bottom:15px;
        ">
            📘 Policy RAG
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "### System status"
    )

    model_ok = model is not None
    database_ok = collection is not None

    if model_ok:
        st.sidebar.success(
            "Embedding model: Ready"
        )
    else:
        st.sidebar.error(
            "Embedding model: Unavailable"
        )

    if database_ok:
        st.sidebar.success(
            "ChromaDB: Connected"
        )
    else:
        st.sidebar.error(
            "ChromaDB: Unavailable"
        )

    count = get_collection_count(
        collection
    )

    st.sidebar.metric(
        "Indexed chunks",
        count,
    )

    configuration = get_llm_configuration()

    if configuration:

        st.sidebar.info(
            f"LLM: {configuration['provider']}"
        )

    else:

        st.sidebar.warning(
            "No LLM API configured"
        )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        **RAG pipeline**

        1. User question
        2. Embedding
        3. ChromaDB retrieval
        4. Grounded prompt
        5. LLM generation
        6. Source citations
        """
    )

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# -------------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------------

def main():

    initialize_session()

    model = load_embedding_model()

    collection = load_collection()

    render_sidebar(
        model,
        collection,
    )

    # ---------------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                📘 Policy RAG Assistant
            </div>

            <div class="hero-subtitle">
                Ask questions about company policies and procedures.
                Answers are grounded in the indexed policy corpus
                and supported with source evidence.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Indexed chunks
                </div>

                <div class="metric-value">
                    {get_collection_count(collection)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Retrieval
                </div>

                <div class="metric-value">
                    Top-{TOP_K}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:

        latency = st.session_state.last_latency

        latency_text = (
            f"{latency:.2f}s"
            if latency
            else "—"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Last latency
                </div>

                <div class="metric-value">
                    {latency_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---------------------------------------------------------------
    # SYSTEM WARNING
    # ---------------------------------------------------------------

    if model is None:

        st.warning(
            "Embedding model is unavailable. "
            "Install sentence-transformers and restart the app."
        )

    if collection is None:

        st.error(
            "ChromaDB could not be initialized. "
            "The UI remains available, but retrieval is disabled."
        )

    elif get_collection_count(collection) == 0:

        st.info(
            "The vector database is empty. "
            "Run `python ingest.py` first to index your policy documents."
        )

    # ---------------------------------------------------------------
    # EXAMPLE QUESTIONS
    # ---------------------------------------------------------------

    st.markdown(
        "### 💡 Example questions"
    )

    example1, example2, example3 = st.columns(3)

    examples = [
        "What is the PTO policy?",
        "What are the remote work requirements?",
        "What expenses are reimbursable?",
    ]

    with example1:

        if st.button(
            examples[0],
            use_container_width=True,
        ):

            st.session_state.example_question = examples[0]

    with example2:

        if st.button(
            examples[1],
            use_container_width=True,
        ):

            st.session_state.example_question = examples[1]

    with example3:

        if st.button(
            examples[2],
            use_container_width=True,
        ):

            st.session_state.example_question = examples[2]

    # ---------------------------------------------------------------
    # QUESTION INPUT
    # ---------------------------------------------------------------

    default_question = st.session_state.pop(
        "example_question",
        "",
    )

    question = st.text_area(
        "Ask a policy question",
        value=default_question,
        height=100,
        placeholder=(
            "Example: How many vacation days "
            "do employees receive?"
        ),
    )

    ask = st.button(
        "🔎 Ask Policy Assistant",
        type="primary",
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # PROCESS QUESTION
    # ---------------------------------------------------------------

    if ask:

        clean_question = safe_text(
            question
        )

        if not clean_question:

            st.warning(
                "Please enter a policy question."
            )

            return

        if len(clean_question) > 1000:

            st.warning(
                "Please keep your question under "
                "1000 characters."
            )

            return

        start_time = time.perf_counter()

        with st.spinner(
            "Searching policy documents..."
        ):

            documents = retrieve_documents(
                clean_question,
                model,
                collection,
                TOP_K,
            )

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer, provider = generate_answer(
                clean_question,
                documents,
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        st.session_state.last_latency = elapsed

        st.session_state.messages.append(
            {
                "question": clean_question,
                "answer": answer,
                "sources": documents,
                "latency": elapsed,
                "provider": provider,
            }
        )

    # ---------------------------------------------------------------
    # CONVERSATION
    # ---------------------------------------------------------------

    if st.session_state.messages:

        st.markdown(
            "### 💬 Conversation"
        )

        for message in reversed(
            st.session_state.messages
        ):

            st.markdown(
                f"""
                <div class="question-box">
                    <strong>Question</strong>
                    <br>
                    {html.escape(message["question"])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="answer-box">

                    <strong>
                        🤖 Policy Assistant
                    </strong>

                    <br><br>

                    {html.escape(message["answer"]).replace(chr(10), "<br>")}

                </div>
                """,
                unsafe_allow_html=True,
            )

            latency = message.get(
                "latency",
                0,
            )

            provider = message.get(
                "provider",
                "Unknown",
            )

            st.caption(
                f"⏱️ {latency:.2f}s  •  "
                f"Provider: {provider}  •  "
                f"Top-{TOP_K} retrieval"
            )

            display_sources(
                message.get(
                    "sources",
                    [],
                )
            )

            st.markdown("---")

    else:

        st.markdown(
            """
            <div class="card">

                <h3 style="color:#0b3d91;">
                    👋 Welcome
                </h3>

                <p>
                    Ask a question about the company
                    policy corpus. The assistant will
                    retrieve relevant policy passages,
                    generate a grounded response, and
                    display the supporting sources.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------
    # FOOTER
    # ---------------------------------------------------------------

    st.markdown(
        """
        <div class="footer">

            🔒 Grounded RAG • ChromaDB •
            Sentence Transformers • LLM

            <br>

            Policy RAG Application —
            Quantic AI Engineering Project

        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# GLOBAL CRASH PROTECTION
# -------------------------------------------------------------------

if __name__ == "__main__":

    try:

        main()

    except Exception:

        st.error(
            "The application encountered an unexpected "
            "problem, but the Streamlit process is still running."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                traceback.format_exc()
            )
```

This version is aligned with the actual collection name and embedding model used by your current `ingest.py`, rather than creating a second incompatible RAG database. Your repository's `ingest.py` uses `chroma_db`, collection `policy_docs`, 800-character chunks, 150-character overlap, and `all-MiniLM-L6-v2`.

**Important:** no application can honestly be guaranteed to “never crash.” This version is designed to **fail gracefully** for missing models, missing ChromaDB, empty databases, missing API keys, failed LLM calls, malformed documents, and unexpected runtime exceptions.

---

# 2. What your GitHub already has vs. what remains

Based on the repository and the Quantic requirements you pasted:

| Quantic requirement                 | Current status | What remains                                                  |
| ----------------------------------- | -------------- | ------------------------------------------------------------- |
| Virtual environment                 | ✅              | Nothing                                                       |
| `requirements.txt`                  | ✅              | Verify versions                                               |
| README                              | ✅              | Improve final documentation                                   |
| Policy corpus                       | ⚠️             | Verify 5–20 files / 30–120 pages                              |
| PDF/TXT/MD ingestion                | ✅/⚠️           | Current ingestion supports PDF/TXT/MD                         |
| Chunking                            | ✅              | Already implemented                                           |
| Embeddings                          | ✅              | `all-MiniLM-L6-v2`                                            |
| ChromaDB                            | ✅              | Already implemented                                           |
| Top-k retrieval                     | ✅              | Improve UI/evaluation                                         |
| RAG generation                      | ✅              | Improve groundedness                                          |
| Citations                           | ⚠️             | Need evaluation proving citation accuracy                     |
| Outside-corpus refusal              | ⚠️             | New `app.py` strengthens this                                 |
| Web UI                              | ✅              | New UI substantially improves it                              |
| `/`                                 | ✅              | Streamlit UI                                                  |
| `/chat`                             | ❌              | Still needs implementation if grader strictly checks endpoint |
| `/health`                           | ❌              | Still needs implementation if grader strictly checks endpoint |
| Deployment                          | ⚠️             | Optional but strongly recommended                             |
| CI/CD                               | ✅              | GitHub Actions exists                                         |
| Evaluation set 15–30 questions      | ❌              | **Remaining**                                                 |
| Groundedness measurement            | ❌              | **Remaining**                                                 |
| Citation accuracy measurement       | ❌              | **Remaining**                                                 |
| p50/p95 latency                     | ❌              | **Remaining**                                                 |
| Ablation study                      | ❌              | Optional                                                      |
| Design documentation                | ⚠️             | Need `design-and-evaluation.md`                               |
| AI tooling documentation            | ❌/⚠️           | Need `ai-tooling.md`                                          |
| Demo video                          | ❌              | **Remaining**                                                 |
| Final PDF submission                | ❌              | **Remaining**                                                 |
| GitHub shared with `quantic-grader` | ❓              | Must verify                                                   |

The existing CI workflow is already part of the repository, and your README describes the required build/import check.

---

# 3. Your remaining project steps

You are basically at the **evaluation + documentation + submission** stage.

## STEP 1 — Replace `app.py`

Replace your current:

```text
app.py
```

with the version above.

Then run:

```bash
pip install -r requirements.txt
```

Then:

```bash
python ingest.py
```

Then:

```bash
streamlit run app.py
```

Your application should open at:

```text
http://localhost:8501
```

---

# STEP 2 — Verify your policy corpus

Your Quantic requirement is:

> approximately **5–20 short policy documents totaling 30–120 pages**.

You need policies covering several subjects.

Recommended corpus:

```text
policies/
│
├── PTO_Policy.pdf
├── Remote_Work_Policy.pdf
├── Expense_Reimbursement_Policy.pdf
├── Information_Security_Policy.pdf
├── Password_Policy.pdf
├── Holiday_Policy.pdf
├── Code_of_Conduct.pdf
├── Travel_Policy.pdf
├── Leave_of_Absence_Policy.pdf
└── Employee_Benefits_Policy.pdf
```

This is important because your evaluation questions should cover multiple policy domains.

---

# STEP 3 — Run ingestion

Your existing `ingest.py` already implements:

* PDF reading
* TXT reading
* Markdown reading
* text cleaning
* chunking
* embeddings
* ChromaDB persistence

and stores the vectors in:

```text
chroma_db/
```

The current implementation specifically creates the `policy_docs` collection.

Run:

```bash
python ingest.py
```

You should see something similar to:

```text
Loading embedding model...
Found 10 files

Ingesting: PTO_Policy.pdf
✔ Done: PTO_Policy.pdf -> XX chunks

Ingesting: Remote_Work_Policy.pdf
✔ Done: Remote_Work_Policy.pdf -> XX chunks

...

✅ Ingestion completed successfully!
```

---

# STEP 4 — Test the RAG manually

Use at least these questions:

```text
What is the PTO policy?
```

```text
How many vacation days do employees receive?
```

```text
What is the remote work policy?
```

```text
What expenses are reimbursable?
```

```text
What are the password requirements?
```

```text
What holidays are recognized?
```

And most importantly, test an **out-of-scope question**:

```text
Who is the president of France?
```

Expected behavior:

> I could not find this information in the policy documents.

This is important because Quantic specifically requires the application to refuse questions outside the corpus.

---

# STEP 5 — Build the evaluation dataset

This is one of your biggest remaining requirements.

Create:

```text
evaluation/
    evaluation_questions.json
```

You need **15–30 questions**.

I recommend **25**.

Example structure:

```json
[
  {
    "id": 1,
    "question": "How many vacation days do employees receive?",
    "expected_source": "PTO_Policy.pdf",
    "gold_answer": "Employees receive..."
  },
  {
    "id": 2,
    "question": "What is the remote work policy?",
    "expected_source": "Remote_Work_Policy.pdf",
    "gold_answer": "Employees may..."
  }
]
```

Your questions should cover:

* PTO
* vacation
* holidays
* remote work
* expenses
* security
* passwords
* travel
* employee conduct
* leave
* benefits
* reimbursement

Plus several questions that **should be refused**.

---

# STEP 6 — Measure groundedness

You need to report:

> **Groundedness: % of answers whose content is factually consistent with and fully supported by retrieved evidence.**

For example:

```text
25 questions

Grounded answers = 23

Groundedness =
23 / 25 × 100

= 92%
```

Your final report could say:

```text
Groundedness: 92%
```

Do **not** invent this number.

You need to actually test the 25 answers and calculate it.

---

# STEP 7 — Measure citation accuracy

This is another required metric.

For each answer, check:

```text
Does the cited source actually support the answer?
```

Example:

Question:

```text
How many vacation days do employees receive?
```

Answer:

```text
Employees receive 15 vacation days per year.

[Source: PTO_Policy.pdf]
```

If `PTO_Policy.pdf` actually contains that information:

```text
Citation = Correct
```

If it cites:

```text
Remote_Work_Policy.pdf
```

then:

```text
Citation = Incorrect
```

Calculate:

```text
Correct citations / Total questions × 100
```

---

# STEP 8 — Measure latency

This is explicitly required by Quantic.

You need **10–20 queries**.

Record:

```text
Query 1 = 2.31 sec
Query 2 = 1.94 sec
Query 3 = 2.72 sec
...
```

Then calculate:

```text
p50 latency
p95 latency
```

Your report should eventually contain something like:

| Metric       |   Result |
| ------------ | -------: |
| Queries      |       20 |
| p50 latency  | X.XX sec |
| p95 latency  | X.XX sec |
| Mean latency | X.XX sec |

Again, use your **real measured results**, not fabricated values.

---

# STEP 9 — Add `design-and-evaluation.md`

This is explicitly required by your project instructions.

Create:

```text
design-and-evaluation.md
```

It should explain:

```text
1. Architecture
2. Document ingestion
3. Chunking
4. Embeddings
5. Vector database
6. Retrieval
7. Prompting
8. Guardrails
9. LLM
10. Evaluation methodology
11. Groundedness
12. Citation accuracy
13. Latency
14. Results
15. Limitations
16. Future improvements
```

---

# STEP 10 — Add `ai-tooling.md`

This is also explicitly required.

Document how you used AI tools during development.

For example:

```text
# AI Tooling

## ChatGPT

Used for:
- Architecture design
- RAG implementation
- Error handling
- UI development
- Evaluation design
- Documentation

## GitHub Copilot

Used for:
- Code completion
- Refactoring
- Debugging suggestions

## What worked well

AI tools accelerated:
- RAG pipeline development
- Streamlit UI implementation
- Error handling
- Documentation

## What required human validation

Generated code was tested manually to verify:
- Retrieval correctness
- Citation accuracy
- API behavior
- Application stability
- Evaluation results
```

Be truthful about which tools you actually used.

---

# STEP 11 — Deployment

Deployment is optional according to the project instructions, but **I strongly recommend doing it** if you want to demonstrate a stronger project.

Your README already identifies Render/Railway as deployment options.

You want:

```text
GitHub
   ↓
GitHub Actions
   ↓
Build/Test
   ↓
Render/Railway
   ↓
Public RAG application
```

---

# STEP 12 — CI/CD

You already have the GitHub Actions foundation.

Your workflow should at minimum execute:

```bash
pip install -r requirements.txt
```

then:

```bash
python -c "import app"
```

Better:

```bash
python -m compileall .
```

and:

```bash
python -c "import app"
```

Then optionally:

```bash
pytest -q
```

---

# STEP 13 — Demo video

This is **not finished yet** based on the repository information you provided.

Quantic requires a **5–10 minute screen-share demonstration**.

Your demo should show:

### Minute 0–1

Introduction:

```text
My name is Charles Ishimwe Hagenimana.

This project is a Retrieval-Augmented Generation
policy assistant designed to answer questions
using a controlled corpus of company policies.
```

### Minute 1–2

Show architecture:

```text
Policy Documents
       ↓
Parsing
       ↓
Chunking
       ↓
Sentence Transformers
       ↓
ChromaDB
       ↓
Retriever
       ↓
LLM
       ↓
Grounded Answer + Citation
```

### Minute 2–5

Actually use the application.

Ask:

```text
What is the PTO policy?
```

Then:

```text
What is the remote work policy?
```

Then:

```text
What expenses are reimbursable?
```

Show the **sources**.

### Minute 5–6

Demonstrate guardrails:

```text
Who won the 2026 World Cup?
```

The system should refuse because that isn't in the policy corpus.

### Minute 6–7

Show:

```text
Groundedness
Citation Accuracy
Latency p50
Latency p95
```

### Minute 7–8

Show GitHub:

```text
app.py
ingest.py
policies/
chroma_db/
requirements.txt
.github/
README.md
design-and-evaluation.md
ai-tooling.md
```

### Minute 8–10

Explain:

```text
Architecture decisions
Evaluation
CI/CD
Deployment
Future improvements
```

---

# 14. Final submission PDF

Your final PDF should contain only the important submission information and the two required links.

Something like:

```text
QUANTIC AI ENGINEERING PROJECT

Policy RAG Assistant

Charles Ishimwe Hagenimana

GitHub Repository:
https://github.com/charlesishimwe/Policy_RAG_App

Demo Video:
[YOUR VIDEO LINK]

Project Summary:

A Retrieval-Augmented Generation application
that answers questions about company policies
using semantic retrieval, ChromaDB,
Sentence Transformers and an LLM.

Evaluation:

Groundedness: XX%
Citation Accuracy: XX%
Latency p50: XX sec
Latency p95: XX sec

Technology:

Python
Streamlit
ChromaDB
Sentence Transformers
OpenAI-compatible LLM
GitHub Actions
```

---

# 15. Your project status now

I would consider your project approximately:

**Core engineering: ~80–90% complete**

**Submission/evaluation: ~40–50% complete**

The biggest missing pieces are **not another RAG implementation**. You now need to prove that the system works.

### Remaining priority order

```text
1. Replace app.py                         ← DO NOW
2. Verify policy corpus
3. Run ingestion
4. Test application
5. Create 25-question evaluation set
6. Measure groundedness
7. Measure citation accuracy
8. Measure p50/p95 latency
9. Create design-and-evaluation.md
10. Create ai-tooling.md
11. Verify GitHub Actions
12. Deploy to Render/Railway (recommended)
13. Record 5–10 minute demo
14. Create final PDF
15. Share repository with quantic-grader
16. Submit
```

The key point is: **don't keep rebuilding the RAG pipeline unnecessarily.** Your repository already contains the ingestion/vector/RAG foundation. The remaining work is to turn it into a polished, demonstrably evaluated submission.

If you want, the **next step should be Step 5: I can give you the complete `evaluation_questions.json` + evaluation script that automatically calculates groundedness, citation accuracy, p50 and p95 latency for this exact repository.**
