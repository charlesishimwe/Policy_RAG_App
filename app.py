import logging
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "PolicyCopilot"
APP_TAGLINE = "Enterprise Policy Intelligence"

TOP_K = int(os.getenv("TOP_K", "5"))


# ============================================================
# 2. LOGGING
# ============================================================
# Technical errors are logged server-side.
# They are intentionally NOT displayed to end users.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("policycopilot")


# ============================================================
# 3. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 4. ENTERPRISE DESIGN SYSTEM
# ============================================================
#
# Brand:
#   Blue       #155EEF
#   Deep Blue  #0B3B91
#   Green      #16A34A
#   Light Blue #EAF4FF
#   White      #FFFFFF
#   Black      #0B1220
#
# Neutral:
#   Background #F5F7FA
#   Border     #D9E2EC
#   Gray       #64748B
#
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APPLICATION
    ======================================================== */

    .stApp {
        background-color: #F5F7FA;
        color: #0B1220;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 24px;
        padding-bottom: 55px;
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


    /* ========================================================
       TYPOGRAPHY
    ======================================================== */

    html,
    body,
    [class*="css"] {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Helvetica,
            Arial,
            sans-serif;
    }

    h1 {
        color: #0B1220 !important;
        font-size: 38px !important;
        font-weight: 800 !important;
        letter-spacing: -1.1px !important;
        line-height: 1.15 !important;
    }

    h2 {
        color: #0B1220 !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #0B1220 !important;
        font-weight: 700 !important;
    }

    p {
        color: #475569;
        line-height: 1.6;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #D9E2EC;

        min-width: 280px !important;
        max-width: 280px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 22px 18px 28px 18px;
    }

    section[data-testid="stSidebar"] h2 {
        color: #0B1220 !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        margin-bottom: 0 !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: #0B1220 !important;
        font-size: 14px !important;
        font-weight: 750 !important;
    }

    section[data-testid="stSidebar"] p {
        color: #475569 !important;
        font-size: 12.5px !important;
        line-height: 1.55 !important;
    }


    /* ========================================================
       TEXT AREA
    ======================================================== */

    div[data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #0B1220 !important;

        border: 1px solid #B8C6D6 !important;

        border-radius: 9px !important;

        font-size: 15px !important;

        line-height: 1.6 !important;

        padding: 16px !important;

        box-shadow:
            0 1px 3px rgba(11, 18, 32, 0.03)
            !important;
    }

    div[data-testid="stTextArea"] textarea:hover {
        border-color: #7E9CC2 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #155EEF !important;

        box-shadow:
            0 0 0 3px rgba(21, 94, 239, 0.10)
            !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #7A8797 !important;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {
        min-height: 43px !important;

        border-radius: 8px !important;

        border: 1px solid #C7D3E0 !important;

        background-color: #FFFFFF !important;

        color: #0B1220 !important;

        font-weight: 600 !important;

        font-size: 13px !important;

        transition:
            all 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #EAF4FF !important;

        border-color: #155EEF !important;

        color: #155EEF !important;
    }


    /* ========================================================
       PRIMARY ACTION
    ======================================================== */

    button[kind="primary"] {
        background-color: #155EEF !important;

        border-color: #155EEF !important;

        color: #FFFFFF !important;

        font-weight: 700 !important;

        box-shadow:
            0 3px 8px rgba(21, 94, 239, 0.18)
            !important;
    }

    button[kind="primary"]:hover {
        background-color: #0B4CC4 !important;

        border-color: #0B4CC4 !important;

        color: #FFFFFF !important;
    }


    /* ========================================================
       METRICS
    ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;

        border: 1px solid #D9E2EC;

        border-radius: 9px;

        padding: 15px 17px;

        box-shadow:
            0 1px 3px rgba(11, 18, 32, 0.025);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 11px !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0B1220 !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       CONTAINERS / CARDS
    ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;

        border: 1px solid #D9E2EC !important;

        border-radius: 9px !important;

        box-shadow:
            0 1px 3px rgba(11, 18, 32, 0.025);
    }


    /* ========================================================
       ALERTS
    ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }


    /* ========================================================
       DIVIDERS
    ======================================================== */

    hr {
        border-color: #D9E2EC !important;
    }


    /* ========================================================
       EXPANDERS
    ======================================================== */

    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;

        border: 1px solid #D9E2EC !important;

        border-radius: 8px !important;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .enterprise-footer {
        color: #64748B;
        font-size: 11px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. RAG PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():
    """
    Initialize the RAG pipeline once per application process.

    Backend implementation:
        rag.pipeline.RAGPipeline

    UI deliberately does not expose technical exceptions.
    """

    try:

        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=TOP_K
        )

        logger.info(
            "RAG pipeline initialized successfully."
        )

        return pipeline

    except Exception as exc:

        logger.exception(
            "RAG initialization failed: %s",
            exc,
        )

        return None


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.markdown("## 🔵 PolicyCopilot")

    st.caption(
        "Enterprise Policy Intelligence"
    )

    st.divider()


    # --------------------------------------------------------
    # WORKSPACE
    # --------------------------------------------------------

    st.markdown("### Workspace")

    st.markdown(
        "**Policy Assistant**"
    )

    st.caption(
        "Search and understand organizational policies."
    )


    # --------------------------------------------------------
    # PLATFORM
    # --------------------------------------------------------

    st.markdown("### Platform")

    st.markdown(
        "**AI Engine**"
    )

    st.caption(
        "Retrieval-Augmented Generation"
    )

    st.markdown(
        "**Knowledge Base**"
    )

    st.caption(
        "Company policies and procedures"
    )

    st.markdown(
        "**Vector Search**"
    )

    st.caption(
        "ChromaDB"
    )


    st.divider()


    # --------------------------------------------------------
    # AI CONTROLS
    # --------------------------------------------------------

    st.markdown("### AI Controls")

    st.markdown(
        "🟢 **Grounded responses**"
    )

    st.markdown(
        "🟢 **Source citations**"
    )

    st.markdown(
        "🟢 **No policy invention**"
    )

    st.markdown(
        "🟢 **Out-of-scope protection**"
    )


    st.divider()


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    st.markdown("### Retrieval")

    st.markdown(
        f"**Top-K**  \n{TOP_K} relevant policy chunks"
    )

    st.caption(
        "Retrieval depth is configurable through "
        "the application environment."
    )


    st.divider()


    # --------------------------------------------------------
    # PRODUCT FOOTER
    # --------------------------------------------------------

    st.caption(
        "PolicyCopilot"
    )

    st.caption(
        "Enterprise AI Assistant"
    )


# ============================================================
# 7. MAIN HEADER
# ============================================================

header_left, header_center, header_right = st.columns(
    [0.65, 7.85, 1.5]
)


with header_left:

    st.markdown("## 🔵")


with header_center:

    st.markdown(
        "## PolicyCopilot"
    )

    st.caption(
        APP_TAGLINE
    )


with header_right:

    st.markdown("")

    st.success(
        "Secure"
    )


st.divider()


# ============================================================
# 8. HERO
# ============================================================

st.markdown(
    "# Ask about your company policies"
)

st.write(
    "Get clear, policy-grounded answers from your "
    "organization's knowledge base, with transparent "
    "source references."
)


# ============================================================
# 9. TRUST CARDS
# ============================================================

trust_1, trust_2, trust_3 = st.columns(3)


with trust_1:

    with st.container(border=True):

        st.markdown(
            "### 🟢 Grounded"
        )

        st.caption(
            "Responses are based on retrieved policy content."
        )


with trust_2:

    with st.container(border=True):

        st.markdown(
            "### 🔵 Transparent"
        )

        st.caption(
            "Supporting documents and evidence are shown."
        )


with trust_3:

    with st.container(border=True):

        st.markdown(
            "### 🩵 Controlled"
        )

        st.caption(
            "The assistant is designed to avoid unsupported claims."
        )


st.markdown("")


# ============================================================
# 10. POPULAR QUESTIONS
# ============================================================

st.markdown(
    "### Popular questions"
)

st.caption(
    "Choose a common question or enter your own."
)


popular_questions = [
    "How many vacation days do employees receive?",
    "Can unused vacation days be carried over?",
    "What is the remote work policy?",
    "How are business expenses reimbursed?",
]


def select_question(question_text):

    st.session_state[
        "policy_question"
    ] = question_text


question_columns = st.columns(4)


for index, question_example in enumerate(
    popular_questions
):

    with question_columns[index]:

        st.button(
            question_example,
            key=f"popular_{index}",
            use_container_width=True,
            on_click=select_question,
            args=(question_example,),
        )


# ============================================================
# 11. QUESTION INPUT
# ============================================================

st.markdown(
    "### Ask a policy question"
)

question = st.text_area(
    "Policy question",
    key="policy_question",
    label_visibility="collapsed",
    height=130,
    placeholder=(
        "Example: How many vacation days can "
        "employees take per year?"
    ),
)


# ============================================================
# 12. PRIMARY ACTION
# ============================================================

ask = st.button(
    "Ask PolicyCopilot",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 13. QUERY PROCESSING
# ============================================================

if ask:

    question = question.strip()


    # --------------------------------------------------------
    # INPUT VALIDATION
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
    # RAG INITIALIZATION
    # --------------------------------------------------------

    rag_pipeline = initialize_rag()


    if rag_pipeline is None:

        st.error(
            "The policy knowledge service is temporarily "
            "unavailable. Please try again."
        )

        st.stop()


    # --------------------------------------------------------
    # RAG QUERY
    # --------------------------------------------------------

    start_time = time.perf_counter()


    try:

        with st.spinner(
            "Searching the policy knowledge base..."
        ):

            result = rag_pipeline.answer(
                question
            )

    except Exception as exc:

        logger.exception(
            "Question processing failed: %s",
            exc,
        )

        st.error(
            "We couldn't process your question right now. "
            "Please try again."
        )

        st.stop()


    latency = (
        time.perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # RESPONSE VALIDATION
    # --------------------------------------------------------

    if not isinstance(result, dict):

        logger.error(
            "RAG pipeline returned invalid response type."
        )

        st.error(
            "The policy assistant returned an unexpected "
            "response. Please try again."
        )

        st.stop()


    answer = result.get(
        "answer",
        "No answer was generated."
    )

    sources = result.get(
        "sources",
        []
    )

    citations = result.get(
        "citations",
        []
    )


    # ========================================================
    # 14. ANSWER
    # ========================================================

    st.divider()

    st.markdown(
        "### PolicyCopilot Answer"
    )


    with st.container(border=True):

        st.markdown(
            "🟢 **Policy-grounded response**"
        )

        st.write(
            answer
        )


    # ========================================================
    # 15. RESPONSE METRICS
    # ========================================================

    st.markdown("")

    metric_1, metric_2, metric_3 = st.columns(3)


    with metric_1:

        st.metric(
            "Response time",
            f"{latency:.2f}s",
        )


    with metric_2:

        st.metric(
            "Sources retrieved",
            len(sources),
        )


    with metric_3:

        st.metric(
            "Retrieval depth",
            f"Top {TOP_K}",
        )


    # ========================================================
    # 16. SOURCES
    # ========================================================

    if sources:

        st.divider()

        st.markdown(
            "### Sources & Evidence"
        )

        st.caption(
            "Retrieved policy passages supporting the response."
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


            # Keep source cards concise.
            if snippet and len(snippet) > 700:

                snippet = (
                    snippet[:700]
                    + "..."
                )


            with st.container(
                border=True
            ):

                source_left, source_right = st.columns(
                    [8, 2]
                )


                with source_left:

                    st.markdown(
                        f"**{index}. {title}**"
                    )


                with source_right:

                    st.markdown(
                        "🟢 Source"
                    )


                if document_id:

                    st.caption(
                        f"{document_id} • {section}"
                    )

                else:

                    st.caption(
                        section
                    )


                if snippet:

                    st.write(
                        snippet
                    )


    # ========================================================
    # 17. CITATION FALLBACK
    # ========================================================

    elif citations:

        st.divider()

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


            with st.container(
                border=True
            ):

                st.markdown(
                    f"🔵 **{title}**"
                )

                st.caption(
                    section
                )


    # ========================================================
    # 18. NO SUPPORTING EVIDENCE
    # ========================================================

    else:

        st.info(
            "No supporting policy sources were returned "
            "for this question."
        )


# ============================================================
# 19. ENTERPRISE FOOTER
# ============================================================

st.divider()

st.caption(
    "PolicyCopilot · Enterprise Policy Intelligence"
)

st.caption(
    "Secure • Grounded • Transparent • Enterprise-ready"
)
