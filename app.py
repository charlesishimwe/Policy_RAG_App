import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_NAME = "PolicyCopilot"
APP_SUBTITLE = "Enterprise Policy Intelligence"

BASE_DIR = Path(__file__).resolve().parent

TOP_K = int(os.getenv("TOP_K", "5"))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="P",
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
        background: #F8FAFC;
        color: #0F172A;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0F172A;
    }

    .sidebar-section {
        margin-top: 18px;
        margin-bottom: 8px;
        color: #64748B;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .tech-item {
        padding: 8px 0;
        color: #334155;
        font-size: 13px;
    }


    /* -------------------------------------------------------
       BRAND
    ------------------------------------------------------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 34px;
    }

    .logo {
        width: 44px;
        height: 44px;
        border-radius: 12px;

        background: linear-gradient(
            135deg,
            #2563EB,
            #1D4ED8
        );

        color: white;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 21px;
        font-weight: 800;

        box-shadow:
            0 5px 15px rgba(37, 99, 235, 0.20);
    }

    .brand-title {
        color: #0F172A;
        font-size: 20px;
        font-weight: 750;
        letter-spacing: -0.3px;
    }

    .brand-subtitle {
        color: #64748B;
        font-size: 12px;
        margin-top: 2px;
    }


    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero {
        padding: 30px 0 20px 0;
    }

    .hero-title {
        color: #0F172A;
        font-size: 42px;
        font-weight: 750;
        line-height: 1.15;
        letter-spacing: -1.2px;
        margin-bottom: 12px;
    }

    .hero-title span {
        color: #2563EB;
    }

    .hero-text {
        color: #64748B;
        font-size: 16px;
        line-height: 1.6;
        max-width: 680px;
    }


    /* -------------------------------------------------------
       QUESTION CARD
    ------------------------------------------------------- */

    .question-label {
        color: #334155;
        font-size: 14px;
        font-weight: 650;
        margin-bottom: 7px;
    }

    div[data-testid="stTextArea"] textarea {
        background: #FFFFFF;
        color: #0F172A;

        border: 1px solid #CBD5E1;
        border-radius: 12px;

        font-size: 15px;
        line-height: 1.5;

        padding: 14px;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #2563EB;
        box-shadow:
            0 0 0 2px rgba(37, 99, 235, 0.10);
    }


    /* -------------------------------------------------------
       BUTTONS
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #CBD5E1;

        background: #FFFFFF;
        color: #334155;

        font-weight: 600;

        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
        background: #EFF6FF;
    }

    div.stButton > button[kind="primary"] {
        background: #2563EB;
        border-color: #2563EB;
        color: #FFFFFF;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1D4ED8;
        border-color: #1D4ED8;
        color: #FFFFFF;
    }


    /* -------------------------------------------------------
       ANSWER
    ------------------------------------------------------- */

    .answer-card {
        background: #FFFFFF;

        border: 1px solid #E2E8F0;
        border-radius: 14px;

        padding: 24px;

        margin-top: 20px;

        box-shadow:
            0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .answer-title {
        color: #0F172A;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .answer-text {
        color: #334155;
        font-size: 15px;
        line-height: 1.7;
    }


    /* -------------------------------------------------------
       SOURCE CARDS
    ------------------------------------------------------- */

    .source-card {
        background: #F8FAFC;

        border: 1px solid #E2E8F0;
        border-radius: 10px;

        padding: 14px 16px;

        margin-top: 10px;
    }

    .source-title {
        color: #0F172A;
        font-size: 14px;
        font-weight: 700;
    }

    .source-meta {
        color: #64748B;
        font-size: 12px;
        margin-top: 4px;
    }

    .source-snippet {
        color: #475569;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 8px;
    }


    /* -------------------------------------------------------
       METRICS
    ------------------------------------------------------- */

    .metric-card {
        background: #FFFFFF;

        border: 1px solid #E2E8F0;
        border-radius: 10px;

        padding: 12px 15px;
    }

    .metric-label {
        color: #64748B;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        color: #0F172A;
        font-size: 18px;
        font-weight: 700;
        margin-top: 3px;
    }


    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;

        color: #94A3B8;

        font-size: 12px;

        margin-top: 50px;
        padding-top: 20px;

        border-top: 1px solid #E2E8F0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RAG PIPELINE
# ============================================================

@st.cache_resource
def load_rag_pipeline():
    """
    Load the RAG pipeline once and keep it in memory.

    Returns:
        pipeline, error
    """

    try:
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=TOP_K
        )

        return pipeline, None

    except Exception as exc:
        return None, str(exc)


rag_pipeline, pipeline_error = load_rag_pipeline()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="logo">
                P
            </div>

            <div>
                <div class="brand-title">
                    PolicyCopilot
                </div>

                <div class="brand-subtitle">
                    Enterprise Policy Intelligence
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="tech-item">
            <strong>Vector Database</strong><br>
            ChromaDB
        </div>

        <div class="tech-item">
            <strong>Embeddings</strong><br>
            Sentence Transformers
        </div>

        <div class="tech-item">
            <strong>Retrieval</strong><br>
            Top-K = {TOP_K}
        </div>

        <div class="tech-item">
            <strong>LLM</strong><br>
            OpenRouter
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Guardrails</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tech-item">
            ✓ Corpus-grounded answers
        </div>

        <div class="tech-item">
            ✓ Source citations
        </div>

        <div class="tech-item">
            ✓ No policy invention
        </div>

        <div class="tech-item">
            ✓ Out-of-scope refusal
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            Ask about your <span>company policies</span>
        </div>

        <div class="hero-text">
            Get clear answers grounded in official company policies,
            with transparent source citations.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    '<div class="question-label">Try an example</div>',
    unsafe_allow_html=True,
)

example_columns = st.columns(4)

examples = [
    "How many vacation days do employees receive?",
    "Can unused vacation days be carried over?",
    "What is the remote work policy?",
    "How are business expenses reimbursed?",
]

for column, example in zip(example_columns, examples):

    with column:

        if st.button(
            example,
            use_container_width=True,
        ):
            st.session_state["question"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    height=120,
    placeholder=(
        "Ask a question about vacation, remote work, "
        "expenses, security, benefits, travel..."
    ),
    label_visibility="collapsed",
)


# ============================================================
# ASK BUTTON
# ============================================================

ask = st.button(
    "Ask PolicyCopilot",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask:

    question = question.strip()

    # -----------------------------------------------
    # Validate question
    # -----------------------------------------------

    if not question:

        st.warning(
            "Please enter a question about company policies."
        )

        st.stop()

    if len(question) > 1000:

        st.warning(
            "Please keep your question under 1,000 characters."
        )

        st.stop()

    # -----------------------------------------------
    # Pipeline unavailable
    # -----------------------------------------------

    if rag_pipeline is None:

        st.error(
            "PolicyCopilot is temporarily unavailable. "
            "Please try again shortly."
        )

        # Technical information remains hidden from normal UI.
        # It can be enabled during development if required.

        with st.expander("Developer diagnostics"):

            st.code(
                pipeline_error
                if pipeline_error
                else "Unknown RAG pipeline error."
            )

        st.stop()

    # -----------------------------------------------
    # Run RAG
    # -----------------------------------------------

    start_time = time.perf_counter()

    with st.spinner("Searching company policies..."):

        try:

            result = rag_pipeline.answer(
                question
            )

        except Exception as exc:

            st.error(
                "PolicyCopilot could not process the request."
            )

            with st.expander("Developer diagnostics"):

                st.code(str(exc))

            st.stop()

    latency = time.perf_counter() - start_time

    # -----------------------------------------------
    # Extract result
    # -----------------------------------------------

    answer = result.get(
        "answer",
        "No answer was generated."
    )

    citations = result.get(
        "citations",
        []
    )

    sources = result.get(
        "sources",
        []
    )

    # ===================================================
    # ANSWER CARD
    # ===================================================

    st.markdown(
        f"""
        <div class="answer-card">

            <div class="answer-title">
                PolicyCopilot Answer
            </div>

            <div class="answer-text">
                {answer}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ===================================================
    # METRICS
    # ===================================================

    st.markdown("")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Response time
                </div>

                <div class="metric-value">
                    {latency:.2f}s
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
                    Sources
                </div>

                <div class="metric-value">
                    {len(sources)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Retrieval
                </div>

                <div class="metric-value">
                    Top {TOP_K}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===================================================
    # SOURCES
    # ===================================================

    if sources:

        st.markdown("### Sources")

        for index, source in enumerate(
            sources,
            start=1,
        ):

            title = source.get(
                "title",
                source.get(
                    "source",
                    "Policy document",
                ),
            )

            document_id = source.get(
                "document_id",
                "",
            )

            section = source.get(
                "section",
                "Policy section",
            )

            snippet = source.get(
                "snippet",
                "",
            )

            # Keep snippets reasonably short
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            st.markdown(
                f"""
                <div class="source-card">

                    <div class="source-title">
                        {index}. {title}
                    </div>

                    <div class="source-meta">
                        {document_id}
                        &nbsp; • &nbsp;
                        {section}
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
            "No policy sources were returned for this answer."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PolicyCopilot · Enterprise Policy Intelligence
        <br>
        Answers are generated only from the configured policy corpus.
    </div>
    """,
    unsafe_allow_html=True,
)
