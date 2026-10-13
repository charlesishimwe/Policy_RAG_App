
# PolicyCopilot — Enterprise Policy Intelligence

## Overview
PolicyCopilot is a retrieval-augmented generation (RAG) application
that answers questions using company policy documents and provides
source citations.

The included Acme Corporation policies are synthetic demonstration
documents, not actual legal or employment advice.

## Architecture
1. Markdown policy documents are stored in `policies/`.
2. `ingest.py` cleans and chunks documents.
3. Sentence Transformers creates document embeddings.
4. ChromaDB stores the chunks and metadata.
5. The application retrieves relevant passages for a question.
6. OpenRouter generates a grounded answer with source citations.

## Requirements
- Python 3.11 recommended
- OpenRouter API key
- Internet access for the LLM service and initial model download

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file:
```env
OPENROUTER_API_KEY=your_actual_key
OPENROUTER_MODEL=openai/gpt-4o-mini
CHROMA_DB_PATH=Vectordb
```

Never commit the real API key.

## Build the vector database
```bash
python ingest.py
```

## Run the Streamlit interface
```bash
python -m streamlit run app.py
```

## Run the API
In another terminal:
```bash
python -m uvicorn api:app --reload --port 8000
```

- Health: `http://127.0.0.1:8000/health`
- API documentation: `http://127.0.0.1:8000/docs`
- Chat endpoint: `POST http://127.0.0.1:8000/chat`

## Testing
```bash
python test_retrieval.py
```

Run the endpoint and evaluation tests described in
`design-and-evaluation.md` before submitting.

## Limitations
Answers depend on the coverage and quality of the policy corpus,
retrieval quality, LLM availability, and API account access.
The system can make mistakes; verify important decisions against
the original policy documents.
