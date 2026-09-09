import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "PolicyCopilot"
APP_SUBTITLE = "Enterprise Policy Intelligence"

TOP_K = int(os.getenv("TOP_K", "5"))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background-color: #F7F9FC;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 28px;
        padding-bottom: 60px;
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


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5EAF1;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 25px;
    }

    .sidebar-logo {
        width: 42px;
        height: 42px;
        border-radius: 11px;

        background: linear-gradient(
            135deg,
            #2563EB,
            #1D4ED8
        );

        color: #FFFFFF;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 20px;
        font-weight: 800;

        margin-bottom: 12px;
    }

    .sidebar-brand-name {
        color: #0F172A;
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 2px;
    }

    .sidebar-brand-description {
        color: #64748B;
        font-size: 12px;
        margin-bottom: 28px;
    }

    .sidebar-heading {
        color: #64748B;
        font-size: 10px;
        font-weight: 750;

        text-transform: uppercase;
        letter-spacing: 0.10em;

        margin-top: 25px;
        margin-bottom: 9px;
    }

    .sidebar-line {
        color: #334155;
        font-size: 13px;
        line-height: 1.45;

        padding: 6px 0;
    }

    .sidebar-line strong {
        color: #0F172A;
    }

    .guardrail {
        color: #334155;
        font-size: 12px;

        padding: 5px 0;
    }

    .guardrail span {
        color: #16A34A;
        font-weight: 800;
        margin-right: 6px;
    }


    /* =====================================================
       BRAND HEADER
       ===================================================== */

    .brand-wrapper {
        display: flex;
        align-items: center;
        gap: 12px;

        margin-bottom: 35px;
    }

    .brand-logo {
        width: 46px;
        height: 46px;
        border-radius: 13px;

        background: linear-gradient(
            135deg,
            #2563EB,
            #1D4ED8
        );

        color: #FFFFFF;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 22px;
        font-weight: 800;

        box-shadow:
            0 5px 16px rgba(37, 99, 235, 0.18);
    }

    .brand-name {
        color: #0F172A;
        font-size: 21px;
        font-weight: 750;
    }

    .brand-description {
        color: #64748B;
        font-size: 12px;
        margin-top: 2px;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        background: #FFFFFF;

        border: 1px solid #E3E8EF;

        border-radius: 18px;

        padding: 40px 45px;

        margin-bottom: 24px;

        box-shadow:
            0 8px 30px rgba(15, 23, 42, 0.035);
    }

    .hero-title {
        color: #0F172A;

        font-size: 40px;
        font-weight: 800;

        line-height: 1.15;

        letter-spacing: -1.3px;

        margin-bottom: 12px;
    }

    .hero-title span {
        color: #2563EB;
    }

    .hero-description {
        max-width: 700px;

        color: #64748B;

        font-size: 15px;

        line-height: 1.65;
    }


    /* =====================================================
       INPUT
       ===================================================== */

    div[data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;

        color: #0F172A !important;

        border: 1px solid #CBD5E1 !important;

        border-radius: 12px !important;

        font-size: 15px !important;

        padding: 15px !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #2563EB !important;

        box-shadow:
            0 0 0 3px rgba(37, 99, 235, 0.10) !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 10px !important;

        min-height: 42px !important;

        font-weight: 600 !important;
    }

    .stButton > button:hover {
        border-color: #2563EB !important;

        color: #2563EB !important;

        background-color: #EFF6FF !important;
    }


    /* =====================================================
       ANSWER CARD
       ===================================================== */

    .answer-card {
        background-color: #FFFFFF;

        border: 1px solid #E2E8F0;

        border-left: 4px solid #2563EB;

        border-radius: 13px;

        padding: 23px 25px;

        margin-top: 24px;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.035);
    }

    .answer-heading {
        color: #0F172A;

        font-size: 16px;
        font-weight: 750;

        margin-bottom: 12px;
    }

    .answer-content {
        color: #334155;

        font-size: 15px;

        line-height: 1.75;
    }


    /* =====================================================
       SOURCE CARD
       ===================================================== */

    .source-card {
        background-color: #FFFFFF;

        border: 1px solid #E2E8F0;

        border-radius: 11px;

        padding: 17px 19px;

        margin-bottom: 10px;
    }

    .source-title {
        color: #0F172A;

        font-size: 14px;
        font-weight: 700;
    }

    .source-meta {
        color: #64748B;

        font-size: 11px;

        margin-top: 6px;
    }

    .source-snippet {
        color: #475569;

        font-size: 13px;

        line-height: 1.55;

        margin-top: 10px;

        padding-top: 10px;

        border-top: 1px solid #F1F5F9;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    .metric-card {
        background-color: #FFFFFF;

        border: 1px solid #E2E8F0;

        border-radius: 11px;

        padding: 14px 16px;
    }

    .metric-label {
        color: #64748B;

        font-size: 10px;

        text-transform: uppercase;

        letter-spacing: 0.07em;
    }

    .metric-value {
        color: #0F172A;

        font-size: 18px;

        font-weight: 750;

        margin-top: 3px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;

        color: #94A3B8;

        font-size: 11px;

        margin-top: 50px;

        padding-top: 20px;

        border-top: 1px solid #E2E8F0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():
    """
    Initialize the RAG pipeline once.

    Returns:
        pipeline
        error
    """

    try:
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=TOP_K
        )

        return pipeline, None

    except Exception as error:
        return None, str(error)


rag_pipeline, rag_error = initialize_rag()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Logo
    st.markdown(
        '<div class="sidebar-logo">P</div>',
        unsafe_allow_html=True,
    )

    # Brand
    st.markdown(
        '<div class="sidebar-brand-name">PolicyCopilot</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-brand-description">'
        'Enterprise Policy Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    # Technology
    st.markdown(
        '<div class="sidebar-heading">Technology</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-line">
            <strong>RAG</strong><br>
            Custom Retrieval-Augmented Generation
        </div>

        <div class="sidebar-line">
            <strong>Vector database</strong><br>
            ChromaDB
        </div>

        <div class="sidebar-line">
            <strong>Embeddings</strong><br>
            Sentence Transformers
        </div>

        <div class="sidebar-line">
            <strong>Language model</strong><br>
            OpenRouter
        </div>

        <div class="sidebar-line">
            <strong>Interface</strong><br>
            Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieval
    st.markdown(
        '<div class="sidebar-heading">Retrieval</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-line">
            <strong>Top-K</strong><br>
            {TOP_K} relevant policy chunks
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Guardrails
    st.markdown(
        '<div class="sidebar-heading">AI guardrails</div>',
        unsafe_allow_html=True,
    )

    for guardrail in [
        "Grounded responses",
        "Source citations",
        "No policy invention",
        "Out-of-scope refusal",
    ]:

        st.markdown(
            f"""
            <div class="guardrail">
                <span>✓</span>{guardrail}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN BRAND
# ============================================================

st.markdown(
    """
    <div class="brand-wrapper">

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
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            Ask about your <span>company policies</span>
        </div>

        <div class="hero-description">
            Get accurate answers grounded in your organization's
            policy knowledge base, supported by transparent
            source citations.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown(
    "#### Popular questions"
)

example_columns = st.columns(4)

examples = [
    "How many vacation days do employees receive?",
    "Can unused vacation days be carried over?",
    "What is the remote work policy?",
    "How are business expenses reimbursed?",
]

for index, example in enumerate(examples):

    with example_columns[index]:

        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):
            st.session_state["question"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Ask a policy question",
    value=st.session_state.get(
        "question",
        "",
    ),
    height=125,
    placeholder=(
        "Ask a question about vacation, "
        "remote work, expenses, security, "
        "benefits, travel..."
    ),
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
# EXECUTE RAG
# ============================================================

if ask:

    question = question.strip()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not question:

        st.warning(
            "Please enter a policy question."
        )

        st.stop()

    if len(question) > 1000:

        st.warning(
            "Please keep your question under 1,000 characters."
        )

        st.stop()

    # --------------------------------------------------------
    # Pipeline check
    # --------------------------------------------------------

    if rag_pipeline is None:

        st.error(
            "PolicyCopilot is temporarily unavailable. "
            "Please try again later."
        )

        # Technical information is only available to the
        # developer during testing.

        with st.expander(
            "Developer diagnostics"
        ):

            st.code(
                rag_error
                or "Unknown RAG initialization error."
            )

        st.stop()

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:

        with st.spinner(
            "Searching policies and generating your answer..."
        ):

            result = rag_pipeline.answer(
                question
            )

    except Exception as error:

        st.error(
            "The policy assistant could not process "
            "your question."
        )

        with st.expander(
            "Developer diagnostics"
        ):

            st.code(str(error))

        st.stop()

    latency = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Read result
    # --------------------------------------------------------

    if not isinstance(result, dict):

        st.error(
            "The RAG pipeline returned an invalid response."
        )

        st.stop()

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

    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="answer-heading">'
        'PolicyCopilot Answer'
        '</div>',
        unsafe_allow_html=True,
    )

    # IMPORTANT:
    # st.write() is used for the actual LLM answer.
    # This prevents raw HTML from the LLM response
    # from being rendered as interface markup.

    st.write(answer)

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # METRICS
    # ========================================================

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
            <div class="metric-card">

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
            "### Sources & Evidence"
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

            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{index}. {title}**"
                )

                st.caption(
                    f"{document_id}  •  {section}"
                )

                if snippet:

                    if len(snippet) > 600:
                        snippet = (
                            snippet[:600]
                            + "..."
                        )

                    st.write(
                        snippet
                    )

    elif citations:

        st.markdown(
            "### Citations"
        )

        for citation in citations:

            title = citation.get(
                "title",
                "Policy document",
            )

            section = citation.get(
                "section",
                "Policy section",
            )

            st.info(
                f"{title} • {section}"
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
        Policy-grounded AI assistance
    </div>
    """,
    unsafe_allow_html=True,
)
