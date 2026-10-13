
import os
from pathlib import Path

import chromadb
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DB_VALUE = os.getenv("CHROMA_DB_PATH", "Vectordb")
DB_PATH = Path(DB_VALUE)
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH

COLLECTION_NAME = "company_policies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="PolicyCopilot API", version="1.0.0")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """Check whether the API and policy index are available."""
    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        names = [c.name for c in client.list_collections()]

        if COLLECTION_NAME not in names:
            return {
                "status": "degraded",
                "database": "connected",
                "collection": "missing",
            }

        collection = client.get_collection(COLLECTION_NAME)

        return {
            "status": "ok" if collection.count() > 0 else "degraded",
            "database": "connected",
            "collection": COLLECTION_NAME,
            "indexed_chunks": collection.count(),
        }
    except Exception:
        return {
            "status": "degraded",
            "database": "unavailable",
            "collection": "unknown",
        }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenRouter API key is not configured.",
        )

    try:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)

        model = SentenceTransformer(MODEL_NAME)
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(5, collection.count()),
            include=["documents", "metadatas"],
        )

        docs = result["documents"][0] or []
        metadata = result["metadatas"][0] or []

        if not docs:
            return ChatResponse(
                answer="I could not find relevant policy evidence.",
                sources=[],
            )

        sources = []
        evidence_parts = []

        for i, doc in enumerate(docs):
            item = metadata[i] or {} if i < len(metadata) else {}
            filename = item.get("source", "Unknown policy")
            sources.append(filename)
            evidence_parts.append(f"[Source: {filename}]\n{doc}")

        evidence = "\n\n".join(evidence_parts)

        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "temperature": 0.1,
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied company policy "
                            "evidence. Cite the exact source filenames. "
                            "Do not invent rules. If evidence is insufficient, "
                            "say that the available policies do not establish "
                            "an answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nEvidence:\n{evidence}"
                        ),
                    },
                ],
            },
            timeout=60,
        )

        if response.status_code == 401:
            raise HTTPException(
                status_code=502,
                detail="OpenRouter rejected the API key. Check your .env.",
            )

        if response.status_code == 402:
            raise HTTPException(
                status_code=502,
                detail="OpenRouter credits or billing need attention.",
            )

        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]

        return ChatResponse(answer=answer, sources=sorted(set(sources)))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to process the question: {str(exc)[:250]}",
        ) from exc
