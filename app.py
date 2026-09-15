"""Flask entry point."""

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)

from rag.rag_pipeline import (
    RAGPipeline,
)

app = Flask(__name__)

rag = RAGPipeline()

@app.route("/health")

def health():

    return jsonify(

        {

            "status": "healthy"

        }

    )

@app.route("/")

def index():

    return render_template(

        "index.html"

    )


@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = request.get_json()

    question = (
        data.get("question", "")
    )

    if not question:

        return jsonify(
            {
                "error":
                    "Question is required"
            }
        ), 400

    result = rag.ask(
        question
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
