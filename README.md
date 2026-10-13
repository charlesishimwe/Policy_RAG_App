
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


# Design and Evaluation

## Problem
Employees need a way to locate and understand internal policy information.

## Architecture
Policy Markdown files -> cleaning and chunking -> embeddings ->
ChromaDB retrieval -> OpenRouter LLM -> cited answer.

## Components
- Ingestion: `ingest.py`
- Vector database: ChromaDB
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM API: OpenRouter
- Web interface: Streamlit (`app.py`)
- REST API: FastAPI (`api.py`)

## Retrieval and grounding
The system retrieves up to five passages. The LLM is instructed
to answer from those passages, cite source filenames, and acknowledge
when evidence is insufficient.

## Evaluation methodology
Create 15–30 questions covering:
- Questions directly answered by a policy
- Questions requiring multiple passages
- Questions with no answer in the corpus
- Similar or potentially confusing policy topics

For every question, record:
1. Expected answer and source
2. Retrieved sources
3. Whether the answer is supported
4. Whether citations are correct
5. Response latency

## Metrics
Groundedness (%) =
supported answers / evaluated answers * 100

Citation accuracy (%) =
correctly cited answers / answers requiring citations * 100

Latency:
Report median (p50) and 95th percentile (p95) response time
in seconds. State the number of runs and test environment.

## Results
Populate this section with actual measurements after testing.
Do not invent or estimate test results as if they were measured.

| Metric | Result |
|---|---|
| Questions evaluated | Pending |
| Groundedness | Pending |
| Citation accuracy | Pending |
| Latency p50 | Pending |
| Latency p95 | Pending |

## Limitations and future improvements
- Expand and validate corpus coverage.
- Improve retrieval using measured test results.
- Add automated regression tests and monitoring.
- Verify access controls before using confidential documents.
