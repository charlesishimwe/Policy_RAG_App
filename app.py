from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
import os
import time

load_dotenv()

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "PolicyCopilot"
APP_VERSION = "1.0.0"

# ============================================================
# TEMPORARY POLICY DATA
# ============================================================
# This is intentionally simple for Step 1.
# Later, we will replace this with ChromaDB + embeddings + RAG.

POLICIES = [
    {
        "document_id": "PTO-001",
        "title": "Paid Time Off Policy",
        "section": "Vacation Entitlement",
        "content": (
            "Employees receive 20 vacation days per calendar year. "
            "Vacation requests should normally be submitted at least "
            "10 business days before the requested start date. "
            "Unused vacation may carry over up to 5 days into the "
            "following calendar year."
        ),
    },
    {
        "document_id": "REMOTE-001",
        "title": "Remote Work Policy",
        "section": "Remote Work Eligibility",
        "content": (
            "Eligible employees may work remotely up to three days per week. "
            "Remote work requires manager approval and employees must maintain "
            "appropriate security controls when accessing company systems."
        ),
    },
    {
        "document_id": "SEC-001",
        "title": "Information Security Policy",
        "section": "Multi-Factor Authentication",
        "content": (
            "Employees must use multi-factor authentication when accessing "
            "company systems that support MFA. Employees must report suspected "
            "security incidents or phishing attempts to the security team."
        ),
    },
    {
        "document_id": "EXP-001",
        "title": "Expense Policy",
        "section": "Expense Reimbursement",
        "content": (
            "Business expenses must be supported by receipts when required. "
            "Employees should submit expense claims within 30 days of the "
            "expense date and obtain the required manager approval."
        ),
    },
]


# ============================================================
# SIMPLE RETRIEVAL
# ============================================================

def retrieve_policies(question, top_k=3):
    """
    Very simple keyword-based retrieval for the first working version.

    In the next step, this will be replaced with:
        Documents
        -> Chunking
        -> Embeddings
        -> ChromaDB
        -> Top-K semantic retrieval
    """

    question_words = set(
        word.lower().strip(".,?!")
        for word in question.split()
        if len(word) > 2
    )

    scored_documents = []

    for policy in POLICIES:
        text = (
            policy["title"]
            + " "
            + policy["section"]
            + " "
            + policy["content"]
        ).lower()

        score = sum(1 for word in question_words if word in text)

        scored_documents.append((score, policy))

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        policy
        for score, policy in scored_documents[:top_k]
        if score > 0
    ]


# ============================================================
# GUARDRAIL
# ============================================================

def is_out_of_scope(question, retrieved_documents):
    """
    If no policy document appears relevant, refuse the question.
    """

    if not question or not question.strip():
        return True

    return len(retrieved_documents) == 0


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(question, retrieved_documents):
    """
    Temporary deterministic answer generator.

    Later this function will call the LLM through OpenRouter.
    """

    if is_out_of_scope(question, retrieved_documents):
        return (
            "I can only answer questions covered by the "
            "company policy corpus."
        )

    # For this first version, return the strongest matching policy.
    best_policy = retrieved_documents[0]

    return best_policy["content"]


# ============================================================
# HTML / CSS UI
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>{{ app_name }}</title>

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                Roboto,
                Arial,
                sans-serif;

            background: #f4f8fc;
            color: #172033;
            min-height: 100vh;
        }

        .navbar {
            height: 70px;
            background: #ffffff;
            border-bottom: 1px solid #dbe7f3;

            display: flex;
            align-items: center;
            justify-content: space-between;

            padding: 0 6%;

            position: sticky;
            top: 0;
            z-index: 10;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;

            font-size: 21px;
            font-weight: 700;

            color: #0759b8;
        }

        .logo {
            width: 40px;
            height: 40px;

            border-radius: 10px;

            background: #0b66c3;
            color: white;

            display: flex;
            align-items: center;
            justify-content: center;

            font-size: 19px;
            font-weight: 800;

            box-shadow: 0 5px 15px rgba(11, 102, 195, 0.20);
        }

        .status {
            display: flex;
            align-items: center;
            gap: 8px;

            color: #55708f;
            font-size: 13px;
        }

        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;

            background: #16a34a;
        }

        .hero {
            max-width: 1050px;
            margin: 0 auto;

            padding: 70px 20px 30px;

            text-align: center;
        }

        .hero-badge {
            display: inline-block;

            padding: 7px 14px;

            background: #e7f2ff;
            color: #0759b8;

            border-radius: 999px;

            font-size: 13px;
            font-weight: 700;

            margin-bottom: 20px;
        }

        .hero h1 {
            font-size: 48px;
            line-height: 1.1;

            color: #092f57;

            margin-bottom: 18px;
        }

        .hero h1 span {
            color: #0b66c3;
        }

        .hero p {
            max-width: 700px;
            margin: 0 auto;

            color: #60758d;

            font-size: 17px;
            line-height: 1.7;
        }

        .chat-container {
            max-width: 900px;
            margin: 25px auto 70px;

            padding: 0 20px;
        }

        .chat-card {
            background: white;

            border: 1px solid #dce8f5;

            border-radius: 18px;

            box-shadow:
                0 15px 45px rgba(25, 76, 125, 0.10);

            overflow: hidden;
        }

        .chat-header {
            padding: 20px 24px;

            border-bottom: 1px solid #e5edf6;

            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-title {
            font-size: 16px;
            font-weight: 700;
            color: #183b60;
        }

        .chat-subtitle {
            font-size: 13px;
            color: #8094a9;
            margin-top: 3px;
        }

        .chat-body {
            padding: 25px;
        }

        textarea {
            width: 100%;
            min-height: 130px;

            resize: vertical;

            border: 1px solid #cbdbea;

            border-radius: 12px;

            padding: 16px;

            font-size: 16px;

            color: #172033;

            outline: none;

            transition: 0.2s;
        }

        textarea:focus {
            border-color: #0b66c3;

            box-shadow:
                0 0 0 4px rgba(11, 102, 195, 0.10);
        }

        textarea::placeholder {
            color: #9aaabd;
        }

        .button-row {
            margin-top: 15px;

            display: flex;
            justify-content: flex-end;
        }

        button {
            border: none;

            background: #0b66c3;
            color: white;

            padding: 13px 25px;

            border-radius: 10px;

            font-size: 15px;
            font-weight: 700;

            cursor: pointer;

            transition: 0.2s;

            box-shadow:
                0 6px 16px rgba(11, 102, 195, 0.20);
        }

        button:hover {
            background: #084f98;
            transform: translateY(-1px);
        }

        button:disabled {
            background: #9bb9d6;
            cursor: not-allowed;
            transform: none;
        }

        .answer-card {
            margin-top: 25px;

            background: #f8fbff;

            border: 1px solid #dceaf7;

            border-radius: 14px;

            padding: 22px;

            display: none;
        }

        .answer-header {
            color: #0759b8;

            font-size: 14px;
            font-weight: 800;

            text-transform: uppercase;

            letter-spacing: 0.5px;

            margin-bottom: 12px;
        }

        .answer-text {
            color: #263b51;

            font-size: 16px;

            line-height: 1.7;
        }

        .sources {
            margin-top: 22px;

            border-top: 1px solid #dce8f5;

            padding-top: 18px;
        }

        .sources-title {
            color: #183b60;

            font-size: 14px;
            font-weight: 700;

            margin-bottom: 12px;
        }

        .source {
            background: white;

            border: 1px solid #dbe7f3;

            border-radius: 10px;

            padding: 13px;

            margin-top: 9px;
        }

        .source-title {
            color: #0759b8;

            font-weight: 700;

            font-size: 14px;
        }

        .source-section {
            color: #7b8ea3;

            font-size: 12px;

            margin-top: 4px;
        }

        .source-snippet {
            color: #50657b;

            font-size: 13px;

            line-height: 1.5;

            margin-top: 8px;
        }

        .error {
            color: #b42318;
        }

        .examples {
            max-width: 900px;

            margin: 0 auto 60px;

            padding: 0 20px;
        }

        .examples h2 {
            color: #183b60;

            font-size: 20px;

            margin-bottom: 15px;
        }

        .example-grid {
            display: grid;

            grid-template-columns:
                repeat(3, 1fr);

            gap: 12px;
        }

        .example {
            background: white;

            border: 1px solid #dce8f5;

            border-radius: 12px;

            padding: 17px;

            color: #526b84;

            font-size: 14px;

            cursor: pointer;

            transition: 0.2s;
        }

        .example:hover {
            border-color: #0b66c3;

            color: #0759b8;

            transform: translateY(-2px);
        }

        footer {
            text-align: center;

            padding: 25px;

            color: #8295a9;

            font-size: 13px;

            border-top: 1px solid #dce8f5;

            background: white;
        }

        @media (max-width: 700px) {

            .hero h1 {
                font-size: 36px;
            }

            .example-grid {
                grid-template-columns: 1fr;
            }

            .navbar {
                padding: 0 20px;
            }

            .status {
                display: none;
            }

        }

    </style>
</head>

<body>

    <nav class="navbar">

        <div class="brand">

            <div class="logo">
                P
            </div>

            PolicyCopilot

        </div>

        <div class="status">

            <div class="status-dot"></div>

            System Online

        </div>

    </nav>


    <section class="hero">

        <div class="hero-badge">
            ENTERPRISE AI POLICY ASSISTANT
        </div>

        <h1>
            Ask your policies.<br>
            <span>Get grounded answers.</span>
        </h1>

        <p>
            PolicyCopilot helps employees find answers from
            company policies and procedures using
            retrieval-augmented generation.
        </p>

    </section>


    <main class="chat-container">

        <div class="chat-card">

            <div class="chat-header">

                <div>

                    <div class="chat-title">
                        Policy Assistant
                    </div>

                    <div class="chat-subtitle">
                        Answers are grounded in the policy corpus
                    </div>

                </div>

            </div>


            <div class="chat-body">

                <textarea
                    id="question"
                    placeholder="Ask a question about company policies..."
                ></textarea>


                <div class="button-row">

                    <button
                        id="askButton"
                        onclick="askQuestion()"
                    >
                        Ask PolicyCopilot
                    </button>

                </div>


                <div
                    id="answerCard"
                    class="answer-card"
                >

                    <div class="answer-header">
                        Answer
                    </div>

                    <div
                        id="answer"
                        class="answer-text"
                    ></div>


                    <div
                        id="sources"
                        class="sources"
                    >

                        <div class="sources-title">
                            Sources
                        </div>

                        <div id="sourceList"></div>

                    </div>

                </div>

            </div>

        </div>

    </main>


    <section class="examples">

        <h2>
            Try an example
        </h2>

        <div class="example-grid">

            <div
                class="example"
                onclick="useExample(
                    'How many vacation days do employees receive?'
                )"
            >
                How many vacation days do employees receive?
            </div>

            <div
                class="example"
                onclick="useExample(
                    'How many vacation days can be carried over?'
                )"
            >
                How many vacation days can be carried over?
            </div>

            <div
                class="example"
                onclick="useExample(
                    'What are the remote work requirements?'
                )"
            >
                What are the remote work requirements?
            </div>

        </div>

    </section>


    <footer>

        PolicyCopilot v{{ version }}
        &nbsp;•&nbsp;
        Enterprise RAG Policy Assistant

    </footer>


    <script>

        function useExample(question) {

            document.getElementById("question").value = question;

            document.getElementById("question").focus();

        }


        async function askQuestion() {

            const question =
                document
                    .getElementById("question")
                    .value
                    .trim();

            const button =
                document.getElementById("askButton");

            const answerCard =
                document.getElementById("answerCard");

            const answer =
                document.getElementById("answer");

            const sourceList =
                document.getElementById("sourceList");


            if (!question) {

                answerCard.style.display = "block";

                answer.innerHTML =
                    '<span class="error">' +
                    'Please enter a question.' +
                    '</span>';

                sourceList.innerHTML = "";

                return;

            }


            button.disabled = true;

            button.innerText = "Searching policies...";


            answerCard.style.display = "block";

            answer.innerText = "Searching the policy corpus...";

            sourceList.innerHTML = "";


            try {

                const response = await fetch(
                    "/chat",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            question: question
                        })
                    }
                );


                const data = await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.error ||
                        "Request failed."
                    );

                }


                answer.innerText =
                    data.answer || "No answer available.";


                if (
                    data.citations &&
                    data.citations.length > 0
                ) {

                    sourceList.innerHTML =
                        data.citations.map(
                            source => `

                                <div class="source">

                                    <div class="source-title">
                                        ${escapeHtml(
                                            source.title
                                        )}
                                    </div>

                                    <div class="source-section">
                                        ${escapeHtml(
                                            source.document_id
                                        )}
                                        •
                                        ${escapeHtml(
                                            source.section
                                        )}
                                    </div>

                                    <div class="source-snippet">
                                        ${escapeHtml(
                                            source.snippet
                                        )}
                                    </div>

                                </div>

                            `
                        ).join("");

                } else {

                    sourceList.innerHTML =
                        "<div>No supporting sources found.</div>";

                }

            } catch (error) {

                answer.innerHTML =
                    '<span class="error">' +
                    escapeHtml(error.message) +
                    '</span>';

                sourceList.innerHTML = "";

            } finally {

                button.disabled = false;

                button.innerText =
                    "Ask PolicyCopilot";

            }

        }


        function escapeHtml(value) {

            return String(value)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#039;");

        }


        document
            .getElementById("question")
            .addEventListener(
                "keydown",
                function(event) {

                    if (
                        event.key === "Enter" &&
                        !event.shiftKey
                    ) {

                        event.preventDefault();

                        askQuestion();

                    }

                }
            );

    </script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_PAGE,
        app_name=APP_NAME,
        version=APP_VERSION
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": APP_NAME,
            "version": APP_VERSION
        }
    )


@app.route("/chat", methods=["POST"])
def chat():

    start_time = time.perf_counter()

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify(
                {
                    "error": "Request body must be JSON."
                }
            ), 400

        question = data.get("question", "")

        if not isinstance(question, str):
            return jsonify(
                {
                    "error": "Question must be a string."
                }
            ), 400

        question = question.strip()

        if not question:
            return jsonify(
                {
                    "error": "Question cannot be empty."
                }
            ), 400

        # Retrieve relevant policies
        retrieved_documents = retrieve_policies(
            question,
            top_k=3
        )

        # Generate answer
        answer = generate_answer(
            question,
            retrieved_documents
        )

        # Build citations
        citations = []

        for policy in retrieved_documents:

            citations.append(
                {
                    "document_id":
                        policy["document_id"],

                    "title":
                        policy["title"],

                    "section":
                        policy["section"],

                    "snippet":
                        policy["content"]
                }
            )

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        return jsonify(
            {
                "answer": answer,

                "citations": citations,

                "retrieved_documents":
                    len(retrieved_documents),

                "latency_ms":
                    latency_ms
            }
        )

    except Exception as error:

        app.logger.exception(
            "Error processing /chat request"
        )

        return jsonify(
            {
                "error":
                    "An internal error occurred.",
                "details":
                    str(error)
            }
        ), 500


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="127.0.0.1",
        port=port,
        debug=True
    )
