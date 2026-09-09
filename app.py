import os
import time
import logging
from pathlib import Path
from datetime import datetime

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("policycopilot")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="PolicyCopilot",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ENTERPRISE DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       COLOR PALETTE
       ========================================================

       Primary Blue:      #155EEF
       Deep Blue:         #0B3B91
       Light Blue:        #EAF4FF
       Soft Blue:         #F4F8FF
       Green:             #16A34A
       Light Green:       #ECFDF3
       White:             #FFFFFF
       Black:             #0B1220
       Gray:              #64748B
       Border:            #D9E2EC
       Background:        #F4F7FB

    ======================================================== */


    /* ========================================================
       APPLICATION
    ======================================================== */

    .stApp {
        background-color: #F4F7FB;
        color: #0B1220;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 20px;
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
        font-size: 36px !important;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        line-height: 1.2 !important;
    }

    h2 {
        color: #0B1220 !important;
        font-weight: 800 !important;
    }

    h3 {
        color: #0B1220 !important;
        font-weight: 750 !important;
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

        border-right:
            1px solid #D9E2EC;

        min-width: 285px !important;
        max-width: 285px !important;
    }

    section[data-testid="stSidebar"] > div {

        padding:
            20px 18px 25px 18px;
    }

    section[data-testid="stSidebar"] h2 {

        color: #0B1220 !important;

        font-size: 20px !important;

        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] h3 {

        color: #0B1220 !important;

        font-size: 14px !important;

        font-weight: 750 !important;

        margin-top: 8px !important;
    }

    section[data-testid="stSidebar"] p {

        color: #475569 !important;

        font-size: 12.5px !important;

        line-height: 1.5 !important;
    }


    /* ========================================================
       TEXT INPUT
    ======================================================== */

    div[data-testid="stTextArea"] textarea {

        background-color: #FFFFFF !important;

        color: #0B1220 !important;

        border:
            1px solid #B8C6D6 !important;

        border-radius:
            10px !important;

        font-size:
            15px !important;

        line-height:
            1.6 !important;

        padding:
            16px !important;

        box-shadow:
            0 1px 3px
            rgba(11,18,32,0.04) !important;
    }

    div[data-testid="stTextArea"] textarea:hover {

        border-color:
            #7E9CC2 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {

        border-color:
            #155EEF !important;

        box-shadow:
            0 0 0 3px
            rgba(21,94,239,0.10) !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {

        color:
            #7A8797 !important;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {

        min-height:
            42px !important;

        border-radius:
            8px !important;

        border:
            1px solid #C7D3E0 !important;

        background-color:
            #FFFFFF !important;

        color:
            #0B1220 !important;

        font-weight:
            600 !important;

        font-size:
            13px !important;

        transition:
            all 0.15s ease;
    }

    .stButton > button:hover {

        background-color:
            #EAF4FF !important;

        border-color:
            #155EEF !important;

        color:
            #155EEF !important;
    }


    /* ========================================================
       PRIMARY BUTTON
    ======================================================== */

    button[kind="primary"] {

        background-color:
            #155EEF !important;

        border-color:
            #155EEF !important;

        color:
            #FFFFFF !important;

        font-weight:
            700 !important;

        box-shadow:
            0 3px 8px
            rgba(21,94,239,0.18) !important;
    }

    button[kind="primary"]:hover {

        background-color:
            #0B4CC4 !important;

        border-color:
            #0B4CC4 !important;

        color:
            #FFFFFF !important;
    }


    /* ========================================================
       METRICS
    ======================================================== */

    div[data-testid="stMetric"] {

        background-color:
            #FFFFFF;

        border:
            1px solid #D9E2EC;

        border-radius:
            9px;

        padding:
            14px 16px;

        box-shadow:
            0 1px 3px
            rgba(11,18,32,0.025);
    }

    div[data-testid="stMetricLabel"] {

        color:
            #64748B !important;

        font-size:
            11px !important;
    }

    div[data-testid="stMetricValue"] {

        color:
            #0B1220 !important;

        font-size:
            21px !important;

        font-weight:
            800 !important;
    }


    /* ========================================================
       CARDS
    ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        background-color:
            #FFFFFF !important;

        border:
            1px solid #D9E2EC !important;

        border-radius:
            10px !important;

        box-shadow:
            0 1px 4px
            rgba(11,18,32,0.025);
    }


    /* ========================================================
       ALERTS
    ======================================================== */

    div[data-testid="stAlert"] {

        border-radius:
            8px !important;
    }


    /* ========================================================
       DIVIDERS
    ======================================================== */

    hr {

        border-color:
            #D9E2EC !important;
    }


    /* ========================================================
       EXPANDERS
    ======================================================== */

    div[data-testid="stExpander"] {

        background-color:
            #FFFFFF !important;

        border:
            1px solid #D9E2EC !important;

        border-radius:
            8px !important;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .enterprise-footer {

        color:
            #64748B;

        font-size:
            11px;

        text-align:
            center;
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
            "RAG pipeline initialized."
        )

        return pipeline

    except Exception as exc:

        logger.exception(
            "RAG initialization failed: %s",
            exc,
        )

        return None


# ============================================================
# SESSION STATE
# ============================================================

if "policy_question" not in st.session_state:

    st.session_state.policy_question = ""


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "last_result" not in st.session_state:

    st.session_state.last_result = None


# ============================================================
# QUESTION SELECTOR
# ============================================================

def set_question(question_text):

    st.session_state.policy_question = question_text


# ============================================================
# NEW CHAT
# ============================================================

def new_chat():

    st.session_state.policy_question = ""

    st.session_state.chat_history = []

    st.session_state.last_result = None


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
    # NEW CHAT
    # --------------------------------------------------------

    st.button(
        "＋ New Chat",
        use_container_width=True,
        on_click=new_chat,
    )


    st.divider()


    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    st.markdown("### Workspace")

    st.button(
        "💬 Policy Assistant",
        use_container_width=True,
    )

    st.button(
        "📚 Knowledge Base",
        use_container_width=True,
    )

    st.button(
        "📊 Analytics",
        use_container_width=True,
    )


    # --------------------------------------------------------
    # RECENT QUESTIONS
    # --------------------------------------------------------

    if st.session_state.chat_history:

        st.markdown("### Recent Questions")

        recent_questions = (
            st.session_state.chat_history[-4:]
        )

        for index, item in enumerate(
            reversed(recent_questions)
        ):

            question_text = item.get(
                "question",
                ""
            )

            if len(question_text) > 35:

                question_text = (
                    question_text[:35]
                    + "..."
                )

            st.button(
                f"• {question_text}",
                key=f"history_{index}",
                use_container_width=True,
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
        f"**Top-K:** {TOP_K}"
    )

    st.caption(
        "Relevant policy chunks are retrieved "
        "before answer generation."
    )


    st.divider()


    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.markdown("### System")

    st.markdown(
        "🟢 **Knowledge service**"
    )

    st.caption(
        "Policy-grounded AI"
    )


    st.divider()


    st.caption(
        "PolicyCopilot"
    )

    st.caption(
        "Enterprise AI Assistant"
    )


# ============================================================
# TOP NAVIGATION BAR
# ============================================================

top_left, top_center, top_right = st.columns(
    [0.6, 7.4, 2]
)


with top_left:

    st.markdown(
        "## 🔵"
    )


with top_center:

    st.markdown(
        "### PolicyCopilot"
    )

    st.caption(
        "Enterprise Policy Intelligence"
    )


with top_right:

    st.markdown("")

    st.success(
        "Secure"
    )


st.divider()


# ============================================================
# WELCOME HEADER
# ============================================================

st.markdown(
    "# How can I help with your policies?"
)

st.write(
    "Ask questions about company policies and procedures. "
    "PolicyCopilot retrieves relevant evidence and provides "
    "a grounded response with supporting sources."
)


# ============================================================
# TRUST CARDS
# ============================================================

card1, card2, card3, card4 = st.columns(4)


with card1:

    with st.container(border=True):

        st.markdown(
            "### 🟢 Grounded"
        )

        st.caption(
            "Answers are grounded in the policy corpus."
        )


with card2:

    with st.container(border=True):

        st.markdown(
            "### 🔵 Cited"
        )

        st.caption(
            "Supporting policy sources are displayed."
        )


with card3:

    with st.container(border=True):

        st.markdown(
            "### 🩵 Secure"
        )

        st.caption(
            "Designed for controlled enterprise use."
        )


with card4:

    with st.container(border=True):

        st.markdown(
            "### ⚫ Focused"
        )

        st.caption(
            "Concise answers without unnecessary detail."
        )


st.markdown("")


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown(
    "### Suggested questions"
)

suggestions = [
    "How many vacation days do employees receive?",
    "Can unused vacation days be carried over?",
    "What is the remote work policy?",
    "How are business expenses reimbursed?",
]


suggestion_columns = st.columns(4)


for index, suggestion in enumerate(
    suggestions
):

    with suggestion_columns[index]:

        st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
            on_click=set_question,
            args=(suggestion,),
        )


# ============================================================
# CHAT / MAIN CONTENT FRAME
# ============================================================

st.markdown(
    "### Policy Assistant"
)

with st.container(border=True):

    # --------------------------------------------------------
    # EXISTING CHAT HISTORY
    # --------------------------------------------------------

    if st.session_state.chat_history:

        for message in st.session_state.chat_history:

            role = message.get(
                "role"
            )

            content = message.get(
                "content",
                ""
            )

            timestamp = message.get(
                "timestamp",
                ""
            )


            if role == "user":

                st.markdown(
                    "🔵 **You**"
                )

                st.write(
                    content
                )

                if timestamp:

                    st.caption(
                        timestamp
                    )


            else:

                st.markdown(
                    "🟢 **PolicyCopilot**"
                )

                st.write(
                    content
                )

                if timestamp:

                    st.caption(
                        timestamp
                    )


            st.divider()


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    st.markdown(
        "**Ask a policy question**"
    )

    question = st.text_area(
        "Policy question",
        key="policy_question",
        label_visibility="collapsed",
        height=120,
        placeholder=(
            "Ask about vacation, remote work, "
            "expenses, travel, security, benefits..."
        ),
    )


    # --------------------------------------------------------
    # INPUT ACTIONS
    # --------------------------------------------------------

    action_left, action_middle, action_right = st.columns(
        [1.2, 1.2, 7.6]
    )


    with action_left:

        attach = st.button(
            "📎 Attach",
            use_container_width=True,
        )


    with action_middle:

        clear = st.button(
            "Clear",
            use_container_width=True,
        )


    with action_right:

        ask = st.button(
            "Send  →",
            type="primary",
            use_container_width=True,
        )


    if clear:

        st.session_state.policy_question = ""

        st.rerun()


    if attach:

        st.info(
            "Document attachment can be enabled through "
            "the ingestion pipeline."
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
            "Please keep your question under 1,000 characters."
        )

        st.stop()


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    rag_pipeline = initialize_rag()


    if rag_pipeline is None:

        st.error(
            "The policy service is temporarily unavailable. "
            "Please try again."
        )

        st.stop()


    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    start_time = time.perf_counter()


    try:

        with st.spinner(
            "PolicyCopilot is analyzing the policy knowledge base..."
        ):

            result = rag_pipeline.answer(
                question
            )

    except Exception as exc:

        logger.exception(
            "RAG request failed: %s",
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
            "Invalid RAG response."
        )

        st.error(
            "The assistant returned an unexpected response."
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
    # SAVE CHAT
    # ========================================================

    timestamp = datetime.now().strftime(
        "%H:%M"
    )


    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
            "timestamp": timestamp,
        }
    )


    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "timestamp": timestamp,
        }
    )


    st.session_state.last_result = result


    # ========================================================
    # ANSWER FRAME
    # ========================================================

    st.divider()

    st.markdown(
        "### 🟢 PolicyCopilot Response"
    )


    with st.container(border=True):

        st.markdown(
            "**Policy-grounded answer**"
        )

        st.write(
            answer
        )


    # ========================================================
    # ACTIONS
    # ========================================================

    action1, action2, action3, action4 = st.columns(4)


    with action1:

        st.button(
            "↻ Regenerate",
            use_container_width=True,
            disabled=True,
        )


    with action2:

        st.button(
            "📋 Copy",
            use_container_width=True,
            disabled=True,
        )


    with action3:

        st.button(
            "↗ Export",
            use_container_width=True,
            disabled=True,
        )


    with action4:

        if sources:

            st.button(
                "📚 View Sources",
                use_container_width=True,
            )

        else:

            st.button(
                "No Sources",
                use_container_width=True,
                disabled=True,
            )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.markdown("")

    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "Response time",
            f"{latency:.2f}s",
        )


    with metric2:

        st.metric(
            "Sources",
            len(sources),
        )


    with metric3:

        st.metric(
            "Retrieval",
            f"Top {TOP_K}",
        )


    with metric4:

        st.metric(
            "Grounding",
            "Policy",
        )


    # ========================================================
    # SOURCES
    # ========================================================

    if sources:

        st.divider()

        st.markdown(
            "### 📚 Sources & Evidence"
        )

        st.caption(
            "Policy passages retrieved to support the answer."
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
                        "🟢 Verified"
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
            "### 📚 Citations"
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
    # NO SOURCES
    # ========================================================

    else:

        st.info(
            "No supporting policy sources were returned."
        )


# ============================================================
# EMPTY STATE
# ============================================================

if (
    not ask
    and not st.session_state.chat_history
):

    st.markdown("")

    with st.container(border=True):

        st.markdown(
            "### 🩵 Start a conversation"
        )

        st.write(
            "Ask PolicyCopilot about a company policy. "
            "The assistant will retrieve relevant policy "
            "evidence before generating its response."
        )

        st.caption(
            "Examples: vacation • remote work • expenses • "
            "security • benefits • travel"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PolicyCopilot · Enterprise Policy Intelligence"
)

st.caption(
    "🔵 Secure  •  🟢 Grounded  •  🩵 Transparent"
)
