
"""
Acme Corp Policy Assistant — Flask Web Application

Endpoints:
  GET  /                 — Chat UI
  POST /chat             — RAG query API (returns JSON)
  GET  /health           — Health check (returns JSON)
  GET  /policies/<path>  — Serve policy documents
"""

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, abort
from dotenv import load_dotenv

from rag import answer_question, get_collection, get_embed_model

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY", os.urandom(24).hex()
)


@app.route("/", methods=["GET"])
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Process a RAG question and return a JSON response."""
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "question" not in data:
        return jsonify({
            "error": "Request body must contain a 'question' field."
        }), 400

    question = str(data["question"]).strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if len(question) > 2000:
        return jsonify({
            "error": "Question exceeds maximum length of 2000 characters."
        }), 400

    try:
        k = int(data.get("k", os.getenv("TOP_K", "5")))
        k = max(1, min(k, 10))
    except (ValueError, TypeError):
        k = 5

    try:
        result = answer_question(question, k=k)
        return jsonify(result)

    except Exception:
        app.logger.exception("Error answering policy question")
        return jsonify({
            "error": (
                "An error occurred while processing your question. "
                "Please try again."
            )
        }), 500


@app.route("/policies/<path:filename>", methods=["GET"])
def serve_policy(filename):
    """Serve a policy document for inline viewing."""
    policies_dir = Path(
        os.getenv("POLICIES_DIR", "./policies")
    ).resolve()

    file_path = (policies_dir / filename).resolve()

    # Prevent access to files outside the policies directory.
    try:
        file_path.relative_to(policies_dir)
    except ValueError:
        abort(403)

    if not file_path.is_file():
        abort(404)

    return send_file(
        file_path,
        mimetype="text/plain",
        as_attachment=False,
    )


@app.route("/health", methods=["GET"])
def health():
    """Return application and RAG index health status."""
    try:
        collection = get_collection()
        doc_count = collection.count()
        status = "healthy"
        indexed = True

    except Exception:
        app.logger.exception("Health check: ChromaDB unavailable")
        doc_count = 0
        status = "degraded"
        indexed = False

    api_key_env = os.getenv(
        "LLM_API_KEY_ENV", "OPENROUTER_API_KEY"
    )
    llm_configured = bool(os.getenv(api_key_env, ""))

    return jsonify({
        "status": status,
        "indexed_chunks": doc_count,
        "index_ready": indexed,
        "llm_configured": llm_configured,
        "model": os.getenv(
            "LLM_MODEL",
            "meta-llama/llama-3.1-8b-instruct:free",
        ),
        "embed_model": "all-MiniLM-L6-v2",
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = (
        os.getenv("FLASK_DEBUG", "false").lower() == "true"
    )

    print("Pre-loading embedding model...")

    try:
        get_embed_model()
        print("Embedding model loaded successfully.")

        collection = get_collection()
        count = collection.count()

        if count == 0:
            print(
                "WARNING: ChromaDB collection is empty. "
                "Run 'python ingest.py' to index your policies."
            )
        else:
            print(
                f"ChromaDB ready: {count} chunks indexed."
            )

    except Exception:
        app.logger.exception(
            "Startup check failed; verify the RAG configuration."
        )

    print(f"Starting Flask app on port {port}...")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
    )
