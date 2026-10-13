
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
