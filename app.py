import os
import time
import logging
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
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("policycopilot")


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
# ENTERPRISE BANKING-STYLE DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       COLOR SYSTEM
       
       Blue      #155EEF
       Dark Blue #0B3B91
       Green     #16A34A
       Light Blue#EAF4FF
       White     #FFFFFF
       Black     #0B1220
       Gray      #64748B
       Border    #D9E2EC
       Background#F4F7FB
    ====================================================== */


    /* ======================================================
       GLOBAL APPLICATION
    ====================================================== */

    .stApp {
        background: #F4F7FB;
        color: #0B1220;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 28px;
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


    /* ======================================================
       TYPOGRAPHY
    ====================================================== */

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
        letter-spacing: -1px !important;
        line-height: 1.2 !important;
    }

    h2 {
        color: #0B1220 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #0B1220 !important;
        font-weight: 700 !important;
    }

    p {
        color: #475569;
        line-height: 1.6;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #D9E2EC;
        min-width: 275px !important;
        max-width: 275px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 24px 18px 28px 18px;
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
        font-size: 12.5px;
    }


    /* ======================================================
       SIDEBAR BRAND
    ====================================================== */

    section[data-testid="stSidebar"] .stCaption {
        color: #64748B !important;
    }


    /* ======================================================
       TEXT AREA
    ====================================================== */

    div[data-testid="stTextArea"] textarea {
        background: #FFFFFF !important;
        color: #0B1220 !important;

        border: 1px solid #B8C6D6 !important;

        border-radius: 9px !important;

        font-size: 15px !important;

        line-height: 1.6 !important;

        padding: 16px !important;

        box-shadow:
            0 1px 2px rgba(11, 18, 32, 0.03)
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


    /* ======================================================
       STANDARD BUTTONS
    ====================================================== */

    .stButton > button {
        min-height: 43px !important;

        border-radius: 8px !important;

        border: 1px solid #C7D3E0 !important;

        background: #FFFFFF !important;

        color: #0B1220 !important;

        font-weight: 600 !important;

        font-size: 13px !important;

        transition:
            background 0.15s ease,
            border 0.15s ease,
            color 0.15s ease;
    }

    .stButton > button:hover {
        background: #EAF4FF !important;

        border-color: #155EEF !important;

        color: #155EEF !important;
    }


    /* ======================================================
       PRIMARY BLUE BUTTON
    ====================================================== */

    button[kind="primary"] {
        background: #155EEF !important;

        border-color: #155EEF !important;

        color: #FFFFFF !important;

        font-weight: 700 !important;

        box-shadow:
            0 2px 5px rgba(21, 94, 239, 0.18)
            !important;
    }

    button[kind="primary"]:hover {
        background: #0B4CC4 !important;

        border-color: #0B4CC4 !important;

        color: #FFFFFF !important;
    }


    /* ======================================================
       METRICS
    ====================================================== */

    div[data-testid="stMetric"] {
        background: #FFFFFF;

        border: 1px solid #D9E2EC;

        border-radius: 9px;

        padding: 15px 17px;

        box-shadow:
            0 1px 3px rgba(11, 18, 32, 0.025);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;

        font-size: 11px !important;

        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0B1220 !important;

        font-size: 22px !important;

        font-weight: 800 !important;
    }


    /* ======================================================
       CONTAINERS / CARDS
    ====================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;

        border-color: #D9E2EC !important;

        border-radius: 9px !important;
    }


    /* ======================================================
       ALERTS
    ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }


    /* ======================================================
       SUCCESS
    ====================================================== */

    div[data-testid="stAlert"][kind="success"] {
        border-left-color: #16A34A !important;
    }


    /* ======================================================
       DIVIDERS
    ====================================================== */

    hr {
        border-color: #D9E2EC !important;
    }


    /* ======================================================
       EXPANDERS
    ====================================================== */

    div[data-testid="stExpander"] {
        background: #FFFFFF !important;

        border: 1px solid #D9E2EC !important;

        border-radius: 8px !important;
    }


    /* ======================================================
       LINKS
    ====================================================== */

    a {
        color: #155EEF !important;
    }


    /* ======================================================
       CHECKBOXES / RADIO
    ====================================================== */

    div[data-testid="stCheckbox"] label {
        color: #0B1220 !important;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer-text {
        color: #64748B;
        font-size: 11px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RAG INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    try:

        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(
            top_k=TOP_K
        )

        logger.info(
            "PolicyCopilot RAG initialized successfully."
        )

        return pipeline

    except Exception as error:

        logger.exception(
            "RAG initialization failed: %s",
            error
        )

        return None


# ============================================================
# SIDEBAR
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
    # NAVIGATION / PRODUCT
    # --------------------------------------------------------

    st.markdown("### Workspace")

    st.markdown(
        "**Policy Assistant**"
    )

    st.caption(
        "Search and understand company policies."
    )


    st.divider()


    # --------------------------------------------------------
    # SYSTEM
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
        "Company policies & procedures"
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
    # RETRIEVAL SETTINGS
    # --------------------------------------------------------

    st.markdown("### Retrieval")

    st.markdown(
        f"**Top-K**  \n{TOP_K} relevant policy passages"
    )

    st.caption(
        "Relevant policy passages are retrieved "
        "before generating the response."
    )


    st.divider()


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.caption(
        "PolicyCopilot"
    )

    st.caption(
        "Enterprise AI Assistant"
    )


# ============================================================
# MAIN HEADER
# ============================================================

header_left, header_center, header_right = st.columns(
    [0.7, 7.8, 1.5]
)


with header_left:

    st.markdown("## 🔵")


with header_center:

    st.markdown(
        "## PolicyCopilot"
    )

    st.caption(
        "Enterprise Policy Intelligence"
    )


with header_right:

    st.markdown("")

    st.success(
        "Secure"
    )


st.divider()


# ============================================================
# HERO
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
# TRUST / CAPABILITY CARDS
# ============================================================

card1, card2, card3 = st.columns(3)


with card1:

    with st.container(border=True):

        st.markdown(
            "### 🟢 Grounded"
        )

        st.caption(
            "Answers are generated from the approved "
            "policy knowledge base."
        )


with card2:

    with st.container(border=True):

        st.markdown(
            "### 🔵 Transparent"
        )

        st.caption(
            "Supporting policy documents and evidence "
            "are provided with responses."
        )


with card3:

    with st.container(border=True):

        st.markdown(
            "### 🩵 Controlled"
        )

        st.caption(
            "The assistant is designed to avoid "
            "unsupported policy claims."
        )


st.markdown("")


# ============================================================
# POPULAR QUESTIONS
# ============================================================

st.markdown(
    "### Popular questions"
)

st.caption(
    "Select a question or write your own."
)


examples = [
    "How many vacation days do employees receive?",
    "Can unused vacation days be carried over?",
    "What is the remote work policy?",
    "How are business expenses reimbursed?",
]


def set_question(question_text):

    st.session_state[
        "policy_question"
    ] = question_text


example_columns = st.columns(4)


for index, example in enumerate(examples):

    with example_columns[index]:

        st.button(
            example,
            key=f"example_question_{index}",
            use_container_width=True,
            on_click=set_question,
            args=(example,),
        )


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown(
    "### Ask a policy question"
)

question = st.text_area(
    label="Policy question",
    label_visibility="collapsed",
    key="policy_question",
    height=130,
    placeholder=(
        "Example: How many vacation days can "
        "employees take per year?"
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
# PROCESS REQUEST
# ============================================================

if ask:

    question = question.strip()


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not question:

        st.warning(
            "Please enter a policy question."
        )

        st.stop()


    if len(question) > 1000:

        st.warning(
            "Please keep your question under "
            "1,000 characters."
        )

        st.stop()


    # --------------------------------------------------------
    # INITIALIZE RAG
    # --------------------------------------------------------

    rag_pipeline = initialize_rag()


    if rag_pipeline is None:

        st.error(
            "The policy knowledge service is temporarily "
            "unavailable. Please try again."
        )

        st.stop()


    # --------------------------------------------------------
    # EXECUTE RAG
    # --------------------------------------------------------

    start_time = time.perf_counter()


    try:

        with st.spinner(
            "Searching the policy knowledge base..."
        ):

            result = rag_pipeline.answer(
                question
            )

    except Exception:

        logger.exception(
            "Policy question processing failed."
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
    # VALIDATE RESULT
    # --------------------------------------------------------

    if not isinstance(result, dict):

        logger.error(
            "Invalid RAG response."
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
    # ANSWER SECTION
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
    # PERFORMANCE METRICS
    # ========================================================

    st.markdown("")

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Response time",
            f"{latency:.2f}s"
        )


    with metric2:

        st.metric(
            "Sources retrieved",
            len(sources)
        )


    with metric3:

        st.metric(
            "Retrieval depth",
            f"Top {TOP_K}"
        )


    # ========================================================
    # SOURCES
    # ========================================================

    if sources:

        st.divider()

        st.markdown(
            "### Sources & Evidence"
        )

        st.caption(
            "Policy passages retrieved by the system "
            "to support this answer."
        )


        for index, source in enumerate(
            sources,
            start=1
        ):

            title = source.get(
                "title",
                source.get(
                    "source",
                    "Policy document"
                )
            )

            document_id = source.get(
                "document_id",
                ""
            )

            section = source.get(
                "section",
                "Policy section"
            )

            snippet = source.get(
                "snippet",
                ""
            )


            if snippet and len(snippet) > 700:

                snippet = (
                    snippet[:700]
                    + "..."
                )


            with st.container(
                border=True
            ):

                source_header_left, source_header_right = st.columns(
                    [8, 2]
                )


                with source_header_left:

                    st.markdown(
                        f"**{index}. {title}**"
                    )


                with source_header_right:

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
    # CITATION FALLBACK
    # ========================================================

    elif citations:

        st.divider()

        st.markdown(
            "### Citations"
        )


        for citation in citations:

            title = citation.get(
                "title",
                "Policy document"
            )

            section = citation.get(
                "section",
                "Policy section"
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
    # NO SOURCES
    # ========================================================

    else:

        st.info(
            "No supporting policy sources were returned "
            "for this question."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PolicyCopilot · Enterprise Policy Intelligence"
)

st.caption(
    "Secure • Grounded • Transparent • Enterprise-ready"
)
