import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

APP_NAME = "PolicyCopilot"
BASE_DIR = Path(__file__).resolve().parent

TOP_K = int(os.getenv("TOP_K", "5"))


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS - Blue & White UI
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #f4f8ff;
    }

    /* Header */
    .main-header {
        background: #ffffff;
        border-bottom: 1px solid #dce7f7;
        padding: 18px 30px;
        border-radius: 0 0 15px 15px;
        margin-bottom: 35px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .logo {
        width: 48px;
        height: 48px;
        background: #1769e0;
        color: white;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 700;
    }

    .brand-title {
        color: #12315c;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }

    .brand-subtitle {
        color: #718096;
        font-size: 13px;
        margin-top: 2px;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 20px 10px 30px 10px;
    }

    .hero-title {
        color: #102f5e;
        font-size: 42px;
        font-weight: 700;
        line-height: 1.2;
    }

    .hero-title span {
        color: #1769e0;
    }

    .hero-text {
        color: #63728a;
        font-size: 17px;
        line-height: 1.6;
        max-width: 750px;
        margin: auto;
    }

    /* Chat card */
    .chat-card {
        background: white;
        border: 1px solid #dce7f7;
        border-radius: 18px;
        padding: 30px;
        box-shadow: 0 12px 35px rgba(27, 76, 140, 0.08);
        margin-bottom: 25px;
    }

    .section-title {
        color: #17365f;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    /* Answer */
    .answer-card {
        background: #f8fbff;
        border-left: 5px solid #1769e0;
        border-radius: 10px;
        padding: 20px;
        color: #24364f;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 15px;
    }

    /* Source */
    .source-card {
        background: #ffffff;
        border: 1px solid #dce7f7;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }

    .source-title {
        color: #1769e0;
        font-weight: 700;
        font-size: 14px;
    }

    .source-meta {
        color: #75859b;
        font-size: 12px;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .source-snippet {
        color: #465b75;
        font-size: 13px;
        line-height: 1.5;
    }

    /* Status */
    .status-online {
        background: #eaf3ff;
        color: #1769e0;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
    }

    .status-offline {
        background: #fff4f4;
        color: #c0392b;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
    }

    /* Metrics */
    .metric-card {
        background: white;
        border: 1px solid #dce7f7;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }

    .metric-value {
        color: #1769e0;
        font-size: 24px;
        font-weight: 700;
    }

    .metric-label {
        color: #718096;
        font-size: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8190a5;
        font-size: 12px;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "rag_error" not in st.session_state:
    st.session_state.rag_error = None


# ============================================================
# Load RAG pipeline
# ============================================================

@st.cache_resource
def load_rag_pipeline():
    """
    Load the RAG pipeline once and cache it.

    The application will still start if the RAG pipeline
    cannot be imported or initialized.
    """

    try:

        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=TOP_K
        )

        return pipeline, None

    except Exception as exc:

        return None, str(exc)


rag_pipeline, rag_error = load_rag_pipeline()

st.session_state.rag_pipeline = rag_pipeline
st.session_state.rag_error = rag_error


# ============================================================
# Header
# ============================================================

rag_ready = rag_pipeline is not None

status_html = (
    '<div class="status-online">● RAG Online</div>'
    if rag_ready
    else
    '<div class="status-offline">● RAG Not Ready</div>'
)

st.markdown(
    f"""
    <div class="main-header">

        <div class="brand">

            <div class="logo">
                P
            </div>

            <div>
                <div class="brand-title">
                    {APP_NAME}
                </div>

                <div class="brand-subtitle">
                    Enterprise Policy Intelligence
                </div>
            </div>

        </div>

        <div style="margin-top: 12px;">
            {status_html}
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Hero
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            Ask your <span>company policies</span>
        </div>

        <div class="hero-text">
            Get accurate, policy-grounded answers with
            transparent source citations.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔵 PolicyCopilot"
    )

    st.markdown(
        """
        **RAG Configuration**

        - Vector Database: ChromaDB
        - Embeddings: Sentence Transformers
        - Retrieval: Top-K
        - LLM: OpenRouter
        - Framework: Streamlit
        """
    )

    st.divider()

    st.markdown(
        "### System Status"
    )

    if rag_ready:
        st.success("RAG pipeline ready")
    else:
        st.error("RAG pipeline unavailable")

        if rag_error:
            with st.expander("Technical details"):
                st.code(
                    rag_error
                )

    st.divider()

    st.markdown(
        "### Guardrails"
    )

    st.write(
        "✓ Corpus-only answers"
    )

    st.write(
        "✓ Source citations"
    )

    st.write(
        "✓ Evidence snippets"
    )

    st.write(
        "✓ Out-of-scope refusal"
    )

    st.write(
        "✓ Output limits"
    )


# ============================================================
# Main chat area
# ============================================================

with st.container():

    st.markdown(
        '<div class="chat-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Policy Assistant</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Ask a question about company policies and procedures."
    )

    # --------------------------------------------------------
    # Example questions
    # --------------------------------------------------------

    st.markdown(
        "#### Example questions"
    )

    col1, col2, col3, col4 = st.columns(4)

    example_questions = [
        "How many vacation days do employees receive?",
        "How many vacation days can be carried over?",
        "What are the requirements for remote work?",
        "What are the rules for business expenses?",
    ]

    with col1:
        example_1 = st.button(
            "Vacation entitlement",
            use_container_width=True,
        )

    with col2:
        example_2 = st.button(
            "PTO carry-over",
            use_container_width=True,
        )

    with col3:
        example_3 = st.button(
            "Remote work",
            use_container_width=True,
        )

    with col4:
        example_4 = st.button(
            "Expenses",
            use_container_width=True,
        )


    # --------------------------------------------------------
    # Determine selected example
    # --------------------------------------------------------

    selected_question = ""

    if example_1:
        selected_question = example_questions[0]

    elif example_2:
        selected_question = example_questions[1]

    elif example_3:
        selected_question = example_questions[2]

    elif example_4:
        selected_question = example_questions[3]


    # --------------------------------------------------------
    # Question input
    # --------------------------------------------------------

    question = st.text_area(
        "Ask a policy question",
        value=selected_question,
        height=130,
        placeholder=(
            "Example: How many vacation days can "
            "employees carry over?"
        ),
    )


    # --------------------------------------------------------
    # Ask button
    # --------------------------------------------------------

    ask_button = st.button(
        "🔍 Ask PolicyCopilot",
        type="primary",
        use_container_width=True,
    )


    # ========================================================
    # Process question
    # ========================================================

    if ask_button:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a policy question."
            )

        elif len(question) > 1000:

            st.error(
                "Question is too long. "
                "Maximum length is 1000 characters."
            )

        elif rag_pipeline is None:

            st.error(
                "The RAG pipeline is not ready."
            )

            st.info(
                "Make sure the ingestion process has "
                "created the vector database and that "
                "all dependencies are installed."
            )

            if rag_error:
                with st.expander(
                    "Technical error"
                ):
                    st.code(
                        rag_error
                    )

        else:

            start_time = time.perf_counter()

            with st.spinner(
                "Searching policies and generating answer..."
            ):

                try:

                    result = rag_pipeline.answer(
                        question
                    )

                    latency_ms = round(
                        (
                            time.perf_counter()
                            - start_time
                        )
                        * 1000,
                        2,
                    )


                    # ------------------------------------------------
                    # Normalize response
                    # ------------------------------------------------

                    if isinstance(
                        result,
                        dict,
                    ):

                        answer = result.get(
                            "answer",
                            "No answer was generated.",
                        )

                        citations = result.get(
                            "citations",
                            [],
                        )

                        sources = result.get(
                            "sources",
                            [],
                        )

                    else:

                        answer = str(result)

                        citations = []

                        sources = []


                    # ------------------------------------------------
                    # Store conversation
                    # ------------------------------------------------

                    st.session_state.messages.append(
                        {
                            "question": question,
                            "answer": answer,
                            "citations": citations,
                            "sources": sources,
                            "latency_ms": latency_ms,
                        }
                    )


                except Exception as exc:

                    st.error(
                        "An error occurred while "
                        "processing your question."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.exception(exc)


    # --------------------------------------------------------
    # Display latest answer
    # --------------------------------------------------------

    if st.session_state.messages:

        latest = st.session_state.messages[-1]

        st.divider()

        st.markdown(
            "### Answer"
        )

        st.markdown(
            f"""
            <div class="answer-card">
                {latest["answer"]}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        st.markdown(
            "### Sources & Evidence"
        )

        sources = latest.get(
            "sources",
            [],
        )

        if sources:

            for source in sources:

                title = (
                    source.get(
                        "title"
                    )
                    or source.get(
                        "document_id"
                    )
                    or source.get(
                        "source"
                    )
                    or "Policy document"
                )

                document_id = source.get(
                    "document_id",
                    "",
                )

                section = source.get(
                    "section",
                    "",
                )

                filename = source.get(
                    "source",
                    "",
                )

                snippet = (
                    source.get(
                        "snippet"
                    )
                    or source.get(
                        "content"
                    )
                    or ""
                )

                metadata = " · ".join(
                    value
                    for value in [
                        document_id,
                        section,
                        filename,
                    ]
                    if value
                )

                st.markdown(
                    f"""
                    <div class="source-card">

                        <div class="source-title">
                            {title}
                        </div>

                        <div class="source-meta">
                            {metadata}
                        </div>

                        <div class="source-snippet">
                            {snippet}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No source information was returned."
            )


        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div style="
                text-align:right;
                color:#718096;
                font-size:12px;
                margin-top:15px;
            ">
                Response latency:
                {latest.get("latency_ms", "N/A")} ms
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        PolicyCopilot · AI Engineering Project ·
        Retrieval-Augmented Generation
    </div>
    """,
    unsafe_allow_html=True,
)
