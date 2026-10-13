"""
Acme Corp Policy Assistant — Flask Web Application

Endpoints:
  GET  /        — Chat UI
  POST /chat    — RAG query API (returns JSON)
  GET  /health  — Health check (returns JSON)
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, abort
from dotenv import load_dotenv
from rag import answer_question, get_collection, get_embed_model

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())


@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST /chat
    Request body (JSON): { "question": "<user question>" }
    Response (JSON):
      {
        "answer": "...",
        "sources": [{"title": "...", "file": "..."}],
        "snippets": [{"source_title": "...", "text": "...", "similarity": 0.85}],
        "latency_ms": 1234,
        "retrieved_chunks": 5
      }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Request body must contain a 'question' field."}), 400

    question = str(data["question"]).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if len(question) > 2000:
        return jsonify({"error": "Question exceeds maximum length of 2000 characters."}), 400

    try:
        k = int(data.get("k", os.getenv("TOP_K", 5)))
        k = max(1, min(k, 10))
    except (ValueError, TypeError):
        k = 5

    try:
        result = answer_question(question, k=k)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error answering question: {e}")
        return jsonify({
            "error": "An error occurred while processing your question. Please try again.",
        }), 500


@app.route("/policies/<path:filename>")
def serve_policy(filename):
    """Serve a policy document file for inline viewing."""
    policies_dir = Path(os.getenv("POLICIES_DIR", "./policies")).resolve()
    file_path = (policies_dir / filename).resolve()
    if not str(file_path).startswith(str(policies_dir)):
        abort(403)
    if not file_path.exists():
        abort(404)
    return send_file(file_path, mimetype="text/plain; charset=utf-8")


@app.route("/health")
def health():
    """GET /health — Returns application health status."""
    try:
        collection = get_collection()
        doc_count = collection.count()
        status = "healthy"
        indexed = True
    except Exception:
        doc_count = 0
        status = "degraded"
        indexed = False

    api_key_env = os.getenv("LLM_API_KEY_ENV", "OPENROUTER_API_KEY")
    llm_configured = bool(os.getenv(api_key_env, ""))

    return jsonify({
        "status": status,
        "indexed_chunks": doc_count,
        "index_ready": indexed,
        "llm_configured": llm_configured,
        "model": os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
        "embed_model": "all-MiniLM-L6-v2",
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print("Pre-loading embedding model...")
    get_embed_model()

    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            print("⚠️  Warning: ChromaDB collection is empty. Run 'python ingest.py' first.")
        else:
            print(f"✓ ChromaDB ready: {count} chunks indexed.")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to ChromaDB: {e}")

    print(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)  # nosec B104
