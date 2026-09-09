import os
import time
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ============================================================
# Configuration
# ============================================================

APP_NAME = "PolicyCopilot"
APP_VERSION = "1.0.0"

# Import the RAG pipeline only if it exists.
# This keeps the application easy to start while we build
# the RAG components step by step.
try:
    from rag.pipeline import answer_question
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


# ============================================================
# Home page
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        app_name=APP_NAME,
        version=APP_VERSION
    )


# ============================================================
# Health check
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "rag_available": RAG_AVAILABLE
    })


# ============================================================
# Chat API
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    start_time = time.perf_counter()

    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:
        return jsonify({
            "error": "Question is required."
        }), 400

    # --------------------------------------------------------
    # Basic output guardrail
    # --------------------------------------------------------

    if len(question) > 1000:
        return jsonify({
            "error": "Question is too long. Please keep it under 1000 characters."
        }), 400

    # --------------------------------------------------------
    # Out-of-scope guardrail
    # --------------------------------------------------------

    if not RAG_AVAILABLE:

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        return jsonify({
            "answer": (
                "The RAG system is not initialized yet. "
                "Please run the ingestion pipeline first."
            ),
            "citations": [],
            "sources": [],
            "latency_ms": latency_ms
        })

    # --------------------------------------------------------
    # Run RAG pipeline
    # --------------------------------------------------------

    try:

        result = answer_question(question)

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        # Make sure the response is always a dictionary
        if not isinstance(result, dict):
            result = {
                "answer": str(result),
                "citations": [],
                "sources": []
            }

        result["latency_ms"] = latency_ms

        return jsonify(result)

    except Exception as exc:

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        app.logger.exception("RAG request failed")

        return jsonify({
            "answer": (
                "I’m sorry, but I could not process your question. "
                "Please try again."
            ),
            "citations": [],
            "sources": [],
            "latency_ms": latency_ms,
            "error": str(exc)
        }), 500


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":

    port = int(os.getenv("PORT", "5000"))

    app.run(
        host="127.0.0.1",
        port=port,
        debug=True
    )
