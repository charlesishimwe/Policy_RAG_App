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

# Logging is kept in the server/terminal, not displayed to users.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("policycopilot")


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS ONLY
#
# IMPORTANT:
# This is the ONLY HTML/CSS block in the application.
# It is used only for styling and is never displayed as text.
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background-color: #F7F9FC;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 30px;
        padding-bottom: 50px;
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


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 25px;
    }

    section[data-testid="stSidebar"] h2 {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] p {
        color: #475569;
    }


    /* --------------------------------------------------------
       TEXT AREA
    -------------------------------------------------------- */

    div[data-testid="stTextArea"] textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        padding: 14px !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10) !important;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px !important;
        min-height: 42px !important;
        font-weight: 600 !important;
        border: 1px solid #CBD5E1 !important;
    }

    .stButton > button:hover {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background-color: #EFF6FF !important;
    }


    /* --------------------------------------------------------
       PRIMARY BUTTON
    -------------------------------------------------------- */

    button[kind="primary"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: white !important;
    }

    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        color: white !important;
    }


    /* --------------------------------------------------------
       HEADINGS
    -------------------------------------------------------- */

    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.8px;
    }

    h2, h3, h4 {
        color: #0F172A !important;
    }


    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {
        border-color: #E2E8F0 !important;
    }


    /* --------------------------------------------------------
       INFO / SUCCESS / WARNING
    -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE RAG
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():
    """
    Initialize the RAG pipeline once and cache it.

    The actual exception is logged to the server instead of
    displaying Python code or traceback information to users.
    """

    try:
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(top_k=TOP_K)

        logger.info("RAG pipeline initialized successfully.")

        return pipeline

    except Exception:
        logger.exception("Failed to initialize RAG pipeline.")

        return None


rag_pipeline = initialize_rag()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Brand
    st.markdown("## 🔵 PolicyCopilot")
    st.caption(APP_SUBTITLE)

    st.divider()

    # Technology
    st.markdown("### Technology")

    st.markdown(
        """
        **RAG**  
        Retrieval-Augmented Generation
        """
    )

    st.markdown(
        """
        **Vector Database**  
        ChromaDB
        """
    )

    st.markdown(
        """
        **Embeddings**  
        Sentence Transformers
        """
    )

    st.markdown(
        """
        **Language Model**  
        OpenRouter
        """
    )

    st.markdown(
        """
        **Interface**  
        Streamlit
        """
    )

    st.divider()

    # Retrieval
    st.markdown("### Retrieval")

    st.markdown(
        f"""
        **Top-K:** {TOP_K}

        The system retrieves the {TOP_K} most relevant policy
        chunks before generating an answer.
        """
    )

    st.divider()

    # Guardrails
    st.markdown("### AI Guardrails")

    st.markdown("✓ Grounded responses")
    st.markdown("✓ Source citations")
    st.markdown("✓ No policy invention")
    st.markdown("✓ Out-of-scope refusal")

    st.divider()

    st.caption("PolicyCopilot")
    st.caption("Enterprise Policy Intelligence")


# ============================================================
# MAIN BRAND
# ============================================================

brand_left, brand_right = st.columns([0.7, 9])

with brand_left:
    st.markdown("## 🔵")

with brand_right:
    st.markdown("## PolicyCopilot")
    st.caption(APP_SUBTITLE)


# ============================================================
# HERO SECTION
# ============================================================

st.title("Ask about your company policies")

st.write(
    "Get accurate answers grounded in your organization's "
    "policy knowledge base, supported by transparent source citations."
)


# ============================================================
# APPLICATION STATUS
# ============================================================

if rag_pipeline is None:

    st.error(
        "PolicyCopilot is currently unavailable. "
        "Please check the application configuration."
    )

else:

    st.success("Policy knowledge base ready")


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
    st.session_state["policy_question"] = question_text


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

st.markdown("### Ask a policy question")

question = st.text_area(
    label="",
    key="policy_question",
    height=125,
    placeholder=(
        "Ask a question about vacation, remote work, "
        "expenses, security, benefits, travel..."
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
# PROCESS QUESTION
# ============================================================

if ask:

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = question.strip()

    if not question:

        st.warning("Please enter a policy question.")

        st.stop()

    if len(question) > 1000:

        st.warning(
            "Please keep your question under 1,000 characters."
        )

        st.stop()


    # --------------------------------------------------------
    # Validate RAG
    # --------------------------------------------------------

    if rag_pipeline is None:

        st.error(
            "The policy assistant is temporarily unavailable. "
            "Please try again later."
        )

        st.stop()


    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    start_time = time.perf_counter()

    try:

        with st.spinner(
            "Searching policies and generating your answer..."
        ):

            result = rag_pipeline.answer(question)

    except Exception:

        logger.exception(
            "Error while processing policy question."
        )

        st.error(
            "The policy assistant could not process your question. "
            "Please try again."
        )

        st.stop()


    latency = time.perf_counter() - start_time


    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if not isinstance(result, dict):

        st.error(
            "The policy assistant returned an invalid response."
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

    st.markdown("### PolicyCopilot Answer")

    # IMPORTANT:
    # Use st.write() rather than injecting generated text
    # into HTML. This prevents raw HTML/code from appearing
    # incorrectly in the interface.

    st.write(answer)


    # ========================================================
    # METRICS
    # ========================================================

    st.divider()

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            label="Response time",
            value=f"{latency:.2f}s",
        )


    with metric2:

        st.metric(
            label="Sources retrieved",
            value=len(sources),
        )


    with metric3:

        st.metric(
            label="Retrieval depth",
            value=f"Top {TOP_K}",
        )


    # ========================================================
    # SOURCES & EVIDENCE
    # ========================================================

    if sources:

        st.divider()

        st.markdown("### Sources & Evidence")

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


            # Limit very long snippets
            if snippet and len(snippet) > 700:

                snippet = (
                    snippet[:700]
                    + "..."
                )


            with st.container(border=True):

                st.markdown(
                    f"**{index}. {title}**"
                )

                if document_id:

                    st.caption(
                        f"{document_id} • {section}"
                    )

                else:

                    st.caption(section)


                if snippet:

                    st.write(snippet)


    # ========================================================
    # FALLBACK CITATIONS
    # ========================================================

    elif citations:

        st.divider()

        st.markdown("### Citations")

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


    # ========================================================
    # NO SOURCES
    # ========================================================

    else:

        st.info(
            "No supporting policy sources were returned."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PolicyCopilot · Enterprise Policy Intelligence · "
    "Policy-grounded AI assistance"
)
