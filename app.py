import html
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "PolicyCopilot"
APP_SUBTITLE = "Enterprise Policy Intelligence"

TOP_K = int(os.getenv("TOP_K", "5"))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL BLUE / WHITE UI
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    :root {
        --blue: #2563EB;
        --blue-dark: #1D4ED8;
        --blue-light: #EFF6FF;
        --blue-border: #BFDBFE;

        --white: #FFFFFF;
        --background: #F8FAFC;

        --text: #0F172A;
        --text-secondary: #64748B;
        --text-light: #94A3B8;

        --border: #E2E8F0;
        --success: #16A34A;
    }

    .stApp {
        background: var(--background);
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 30px;
        padding-bottom: 50px;
    }

    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 28px;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 32px;
    }

    .sidebar-logo {
        width: 42px;
        height: 42px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: linear-gradient(
            135deg,
            #2563EB,
            #1D4ED8
        );

        color: white;

        font-size: 20px;
        font-weight: 800;

        box-shadow:
            0 5px 14px rgba(37, 99, 235, 0.20);
    }

    .sidebar-title {
        color: var(--text);
        font-size: 19px;
        font-weight: 750;
    }

    .sidebar-subtitle {
        color: var(--text-secondary);
        font-size: 11px;
        margin-top: 2px;
    }

    .sidebar-heading {
        color: var(--text-secondary);
        font-size: 10px;
        font-weight: 750;

        text-transform: uppercase;
        letter-spacing: 0.10em;

        margin-top: 24px;
        margin-bottom: 10px;
    }

    .sidebar-item {
        color: #334155;
        font-size: 13px;
        padding: 7px 0;
        line-height: 1.4;
    }

    .sidebar-item strong {
        color: #0F172A;
    }

    .guardrail {
        display: flex;
        gap: 8px;
        align-items: center;

        color: #334155;
        font-size: 12px;

        margin: 8px 0;
    }

    .check {
        color: var(--success);
        font-weight: 800;
    }


    /* ======================================================
       TOP BRAND
    ====================================================== */

    .top-brand {
        display: flex;
        align-items: center;
        gap: 13px;

        margin-bottom: 42px;
    }

    .top-logo {
        width: 46px;
        height: 46px;

        border-radius: 13px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: linear-gradient(
            135deg,
            #2563EB,
            #1D4ED8
        );

        color: #FFFFFF;

        font-size: 22px;
        font-weight: 800;

        box-shadow:
            0 6px 18px rgba(37, 99, 235, 0.20);
    }

    .top-title {
        color: var(--text);
        font-size: 21px;
        font-weight: 750;
    }

    .top-subtitle {
        color: var(--text-secondary);
        font-size: 12px;
        margin-top: 2px;
    }


    /* ======================================================
       HERO
    ====================================================== */

    .hero-container {
        background: #FFFFFF;

        border: 1px solid var(--border);

        border-radius: 18px;

        padding: 42px 46px;

        margin-bottom: 22px;

        box-shadow:
            0 8px 30px rgba(15, 23, 42, 0.035);
    }

    .hero-title {
        color: var(--text);

        font-size: 43px;
        line-height: 1.12;

        font-weight: 800;

        letter-spacing: -1.5px;

        margin-bottom: 14px;
    }

    .hero-title span {
        color: var(--blue);
    }

    .hero-description {
        max-width: 700px;

        color: var(--text-secondary);

        font-size: 16px;
        line-height: 1.65;
    }


    /* ======================================================
       QUESTION SECTION
    ====================================================== */

    .section-title {
        color: var(--text);

        font-size: 15px;
        font-weight: 700;

        margin-top: 25px;
        margin-bottom: 10px;
    }

    div[data-testid="stTextArea"] textarea {
        background: #FFFFFF !important;

        color: #0F172A !important;

        border: 1px solid #CBD5E1 !important;

        border-radius: 12px !important;

        font-size: 15px !important;

        line-height: 1.55 !important;

        padding: 15px !important;

        box-shadow:
            0 2px 8px rgba(15, 23, 42, 0.025);
    }

    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #2563EB !important;

        box-shadow:
            0 0 0 3px rgba(37, 99, 235, 0.10) !important;
    }


    /* ======================================================
       BUTTONS
    ====================================================== */

    .stButton > button {
        border-radius: 10px !important;

        font-weight: 600 !important;

        min-height: 42px !important;

        transition: all 0.15s ease !important;
    }

    .stButton > button:not([kind="primary"]) {
        background: #FFFFFF !important;

        color: #334155 !important;

        border: 1px solid #E2E8F0 !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: #EFF6FF !important;

        color: #2563EB !important;

        border-color: #BFDBFE !important;
    }

    .stButton > button[kind="primary"] {
        background: #2563EB !important;

        color: #FFFFFF !important;

        border: 1px solid #2563EB !important;

        font-size: 15px !important;

        box-shadow:
            0 4px 12px rgba(37, 99, 235, 0.18);
    }

    .stButton > button[kind="primary"]:hover {
        background: #1D4ED8 !important;

        border-color: #1D4ED8 !important;
    }


    /* ======================================================
       ANSWER
    ====================================================== */

    .answer-container {
        background: #FFFFFF;

        border: 1px solid var(--border);

        border-left: 4px solid var(--blue);

        border-radius: 14px;

        padding: 24px 26px;

        margin-top: 25px;

        box-shadow:
            0 6px 24px rgba(15, 23, 42, 0.035);
    }

    .answer-heading {
        display: flex;
        align-items: center;
        gap: 9px;

        color: var(--text);

        font-size: 16px;
        font-weight: 750;

        margin-bottom: 13px;
    }

    .answer-icon {
        width: 28px;
        height: 28px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 8px;

        background: var(--blue-light);

        color: var(--blue);

        font-size: 13px;
        font-weight: 800;
    }

    .answer-text {
        color: #334155;

        font-size: 15px;

        line-height: 1.75;
    }


    /* ======================================================
       METRICS
    ====================================================== */

    .metric {
        background: #FFFFFF;

        border: 1px solid var(--border);

        border-radius: 11px;

        padding: 14px 16px;

        margin-top: 14px;
    }

    .metric-label {
        color: var(--text-secondary);

        font-size: 10px;

        text-transform: uppercase;

        letter-spacing: 0.07em;
    }

    .metric-value {
        color: var(--text);

        font-size: 18px;

        font-weight: 750;

        margin-top: 4px;
    }


    /* ======================================================
       SOURCES
    ====================================================== */

    .sources-heading {
        color: var(--text);

        font-size: 17px;
        font-weight: 750;

        margin-top: 30px;
        margin-bottom: 12px;
    }

    .source-card {
        background: #FFFFFF;

        border: 1px solid var(--border);

        border-radius: 11px;

        padding: 17px 19px;

        margin-bottom: 10px;
    }

    .source-number {
        display: inline-flex;

        width: 24px;
        height: 24px;

        align-items: center;
        justify-content: center;

        border-radius: 7px;

        background: var(--blue-light);

        color: var(--blue);

        font-size: 11px;
        font-weight: 750;

        margin-right: 8px;
    }

    .source-title {
        color: var(--text);

        font-size: 14px;
        font-weight: 700;
    }

    .source-meta {
        color: var(--text-secondary);

        font-size: 11px;

        margin-top: 7px;
    }

    .source-snippet {
        color: #475569;

        font-size: 13px;

        line-height: 1.55;

        margin-top: 10px;

        padding-top: 10px;

        border-top: 1px solid #F1F5F9;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;

        color: #94A3B8;

        font-size: 11px;

        margin-top: 55px;

        padding-top: 22px;

        border-top: 1px solid var(--border);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    """
    Load the RAG pipeline once.

    Streamlit caches the initialized pipeline so the embedding
    model and vector database are not recreated on every click.
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
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                P
            </div>

            <div>
                <div class="sidebar-title">
                    PolicyCopilot
                </div>

                <div class="sidebar-subtitle">
                    Enterprise Policy Intelligence
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">Technology</div>',
        unsafe_allow_html=True,
    )

    technologies = [
        ("RAG Framework", "Custom RAG Pipeline"),
        ("Vector Database", "ChromaDB"),
        ("Embeddings", "Sentence Transformers"),
        ("Language Model", "OpenRouter"),
        ("Interface", "Streamlit"),
    ]

    for name, value in technologies:

        st.markdown(
            f"""
            <div class="sidebar-item">
                <strong>{name}</strong><br>
                {value}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-heading">Retrieval</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-item">
            <strong>Top-K</strong><br>
            {TOP_K} policy chunks
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">AI Guardrails</div>',
        unsafe_allow_html=True,
    )

    guardrails = [
        "Corpus-grounded responses",
        "Source citations",
        "No policy invention",
        "Out-of-scope refusal",
    ]

    for item in guardrails:

        st.markdown(
            f"""
            <div class="guardrail">
                <span class="check">✓</span>
                <span>{item}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN BRAND
# ============================================================

st.markdown(
    """
    <div class="top-brand">

        <div class="top-logo">
            P
        </div>

        <div>
            <div class="top-title">
                PolicyCopilot
            </div>

            <div class="top-subtitle">
                Enterprise Policy Intelligence
            </div>
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-container">

        <div class="hero-title">
            Ask about your <span>company policies</span>
        </div>

        <div class="hero-description">
            Get clear, policy-grounded answers from your organization's
            policy knowledge base, with transparent source citations.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLES
# ============================================================

st.markdown(
    '<div class="section-title">Popular questions</div>',
    unsafe_allow_html=True,
)

examples = [
    "How many vacation days do employees receive?",
    "Can unused vacation days be carried over?",
    "What is the remote work policy?",
    "How are business expenses reimbursed?",
]

columns = st.columns(4)

for column, example in zip(columns, examples):

    with column:

        if st.button(
            example,
            use_container_width=True,
            key=f"example_{example}",
        ):
            st.session_state["question"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Question",
    value=st.session_state.get("question", ""),
    placeholder=(
        "Ask a question about vacation, remote work, "
        "expenses, security, benefits, travel..."
    ),
    height=125,
    label_visibility="collapsed",
)


# ============================================================
# ASK
# ============================================================

ask_button = st.button(
    "Ask PolicyCopilot",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RAG EXECUTION
# ============================================================

if ask_button:

    question = question.strip()

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CHECK PIPELINE
    # --------------------------------------------------------

    if rag_pipeline is None:

        st.error(
            "PolicyCopilot is temporarily unavailable. "
            "Please try again shortly."
        )

        # Developer-only diagnostic information.
        # This can be removed before final submission if desired.

        with st.expander("Developer diagnostics"):

            st.code(
                pipeline_error
                or "Unknown RAG pipeline initialization error."
            )

        st.stop()

    # --------------------------------------------------------
    # EXECUTE RAG
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:

        with st.spinner(
            "Searching policies and generating an answer..."
        ):

            result = rag_pipeline.answer(
                question
            )

    except Exception as exc:

        st.error(
            "PolicyCopilot could not process your question."
        )

        with st.expander("Developer diagnostics"):

            st.code(str(exc))

        st.stop()

    latency = time.perf_counter() - start_time

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    answer = result.get(
        "answer",
        "No answer was generated.",
    )

    sources = result.get(
        "sources",
        [],
    )

    citations = result.get(
        "citations",
        [],
    )

    # Escape HTML so generated text cannot break the UI.
    safe_answer = html.escape(
        str(answer)
    ).replace(
        "\n",
        "<br>",
    )

    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        f"""
        <div class="answer-container">

            <div class="answer-heading">

                <div class="answer-icon">
                    AI
                </div>

                PolicyCopilot Answer

            </div>

            <div class="answer-text">
                {safe_answer}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.markdown(
            f"""
            <div class="metric">
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
            <div class="metric">
                <div class="metric-label">
                    Sources retrieved
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
            <div class="metric">
                <div class="metric-label">
                    Retrieval depth
                </div>

                <div class="metric-value">
                    Top {TOP_K}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # SOURCES
    # ========================================================

    if sources:

        st.markdown(
            '<div class="sources-heading">Sources & Evidence</div>',
            unsafe_allow_html=True,
        )

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

            # Protect UI from HTML in document content.
            title = html.escape(str(title))
            document_id = html.escape(str(document_id))
            section = html.escape(str(section))
            snippet = html.escape(str(snippet))

            if len(snippet) > 600:
                snippet = snippet[:600] + "..."

            st.markdown(
                f"""
                <div class="source-card">

                    <div>
                        <span class="source-number">
                            {index}
                        </span>

                        <span class="source-title">
                            {title}
                        </span>
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

    elif citations:

        st.markdown(
            '<div class="sources-heading">Citations</div>',
            unsafe_allow_html=True,
        )

        for citation in citations:

            title = html.escape(
                str(
                    citation.get(
                        "title",
                        "Policy document",
                    )
                )
            )

            section = html.escape(
                str(
                    citation.get(
                        "section",
                        "Policy section",
                    )
                )
            )

            st.info(
                f"{title} — {section}"
            )

    else:

        st.info(
            "No supporting policy sources were returned."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PolicyCopilot · Enterprise Policy Intelligence
        <br>
        Responses are grounded in the configured company policy corpus.
    </div>
    """,
    unsafe_allow_html=True,
)
