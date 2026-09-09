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
# PROFESSIONAL BANKING-STYLE UI
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background-color: #F5F7FA;
        color: #0B1220;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 25px;
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

    h1,
    h2,
    h3,
    h4 {
        color: #0B1220 !important;
    }

    p {
        color: #475569;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #D9E1EA;
    }

    section[data-testid="stSidebar"] > div {
        padding: 28px 22px;
    }

    section[data-testid="stSidebar"] h2 {
        color: #0B1220 !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] h3 {
        color: #0B1220 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] p {
        color: #475569 !important;
        font-size: 13px;
        line-height: 1.6;
    }


    /* ======================================================
       TEXT AREA
    ====================================================== */

    div[data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #0B1220 !important;

        border: 1px solid #B8C4D1 !important;

        border-radius: 8px !important;

        font-size: 15px !important;

        line-height: 1.6 !important;

        padding: 16px !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #7A8797 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #155EEF !important;

        box-shadow:
            0 0 0 2px rgba(21, 94, 239, 0.12)
            !important;
    }


    /* ======================================================
       BUTTONS
    ====================================================== */

    .stButton > button {
        border-radius: 8px !important;

        min-height: 44px !important;

        font-weight: 600 !important;

        border: 1px solid #C7D0DB !important;

        background-color: #FFFFFF !important;

        color: #0B1220 !important;

        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #155EEF !important;

        color: #155EEF !important;

        background-color: #F5F8FF !important;
    }


    /* ======================================================
       PRIMARY BUTTON
    ====================================================== */

    button[kind="primary"] {
        background-color: #155EEF !important;

        border-color: #155EEF !important;

        color: #FFFFFF !important;

        font-weight: 700 !important;
    }

    button[kind="primary"]:hover {
        background-color: #0F4BC4 !important;

        border-color: #0F4BC4 !important;

        color: #FFFFFF !important;
    }


    /* ======================================================
       METRICS
    ====================================================== */

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;

        border: 1px solid #D9E1EA;

        border-radius: 8px;

        padding: 15px 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0B1220 !important;
        font-weight: 750 !important;
    }


    /* ======================================================
       ALERTS
    ====================================================== */

    div[data-testid="stAlert"] {
        border-radius: 8px !important;
    }


    /* ======================================================
       EXPANDERS
    ====================================================== */

    div[data-testid="stExpander"] {
        border: 1px solid #D9E1EA !important;

        border-radius: 8px !important;

        background-color: #FFFFFF !important;
    }


    /* ======================================================
       DIVIDERS
    ====================================================== */

    hr {
        border-color: #D9E1EA !important;
    }


    /* ======================================================
       LINK / ACCENT
    ====================================================== */

    a {
        color: #155EEF !important;
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

    st.markdown("## 🔵 PolicyCopilot")

    st.caption(
        "Enterprise Policy Intelligence"
    )

    st.divider()

    st.markdown("### Platform")

    st.markdown(
        "**Policy Intelligence**  \n"
        "AI-powered policy search and question answering."
    )

    st.markdown("### Technology")

    st.markdown(
        "**Retrieval-Augmented Generation**  \n"
        "Grounded responses using the company policy corpus."
    )

    st.markdown(
        "**Vector Database**  \n"
        "ChromaDB"
    )

    st.markdown(
        "**Embeddings**  \n"
        "Sentence Transformers"
    )

    st.markdown(
        "**Language Model**  \n"
        "OpenRouter"
    )

    st.markdown(
        "**Application**  \n"
        "Streamlit"
    )

    st.divider()

    st.markdown("### Retrieval")

    st.markdown(
        f"**Top-K:** {TOP_K}"
    )

    st.caption(
        "The system retrieves the most relevant "
        "policy passages before generating an answer."
    )

    st.divider()

    st.markdown("### AI Controls")

    st.markdown("✓ Policy-grounded answers")
    st.markdown("✓ Source citations")
    st.markdown("✓ No policy invention")
    st.markdown("✓ Out-of-scope protection")

    st.divider()

    st.caption(
        "Internal Enterprise AI Assistant"
    )


# ============================================================
# MAIN HEADER
# ============================================================

header_left, header_right = st.columns(
    [0.8, 9]
)

with header_left:

    st.markdown("## 🔵")


with header_right:

    st.markdown(
        "## PolicyCopilot"
    )

    st.caption(
        "Enterprise Policy Intelligence"
    )


st.divider()


# ============================================================
# HERO
# ============================================================

st.markdown(
    "# Ask about your company policies"
)

st.markdown(
    "Get clear, policy-grounded answers from your "
    "organization's knowledge base, with transparent "
    "source references."
)


# ============================================================
# SECURITY / TRUST MESSAGE
# ============================================================

trust_col1, trust_col2, trust_col3 = st.columns(3)

with trust_col1:

    st.markdown(
        "**🔒 Policy grounded**"
    )

    st.caption(
        "Answers are based on the approved policy corpus."
    )


with trust_col2:

    st.markdown(
        "**✓ Transparent sources**"
    )

    st.caption(
        "Supporting policy documents are shown with answers."
    )


with trust_col3:

    st.markdown(
        "**◉ Controlled AI**"
    )

    st.caption(
        "The assistant is designed to avoid unsupported claims."
    )


st.divider()


# ============================================================
# POPULAR QUESTIONS
# ============================================================

st.markdown("### Popular questions")

example_columns = st.columns(4)

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
# QUESTION AREA
# ============================================================

st.markdown("### Ask a policy question")

question = st.text_area(
    label="Policy question",
    label_visibility="collapsed",
    key="policy_question",
    height=130,
    placeholder=(
        "Example: How many vacation days can I take "
        "per year?"
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
# ANSWER PROCESSING
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
    # INITIALIZE RAG ONLY WHEN NEEDED
    #
    # This avoids displaying a startup error to the user.
    # --------------------------------------------------------

    rag_pipeline = initialize_rag()


    if rag_pipeline is None:

        st.error(
            "We couldn't connect to the policy knowledge "
            "service right now. Please try again."
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
            "RAG pipeline returned an invalid response."
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
    # ANSWER
    # ========================================================

    st.divider()

    st.markdown(
        "### PolicyCopilot Answer"
    )

    st.container(
        border=True
    )

    with st.container(
        border=True
    ):

        st.write(answer)


    # ========================================================
    # RESPONSE INFORMATION
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
            "Retrieval",
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
            "The following policy passages were retrieved "
            "to support the response."
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


            if snippet:

                if len(snippet) > 700:

                    snippet = (
                        snippet[:700]
                        + "..."
                    )


            with st.container(
                border=True
            ):

                st.markdown(
                    f"**{index}. {title}**"
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
    # CITATIONS FALLBACK
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
                    f"**{title}**"
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
    "Policy-grounded AI assistance"
)
