import os
from typing import Any, Dict, List

from dotenv import load_dotenv

from rag.embeddings import EmbeddingModel
from rag.vectorstore import VectorStore
from rag.generator import LLMGenerator


load_dotenv()


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Flow:
        User question
            ↓
        Embedding
            ↓
        ChromaDB retrieval
            ↓
        Context construction
            ↓
        OpenRouter LLM
            ↓
        Answer + citations
    """

    def __init__(self, top_k: int = 5):
        self.top_k = int(os.getenv("TOP_K", top_k))

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore(
            collection_name="policy_documents"
        )

        self.generator = LLMGenerator()

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant policy chunks.
        """

        if not question or not question.strip():
            return []

        query_embedding = self.embedding_model.embed_query(
            question.strip()
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        return results

    def build_context(
        self,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Build the context sent to the LLM.
        """

        if not documents:
            return ""

        context_parts = []

        for index, doc in enumerate(documents, start=1):

            metadata = doc.get("metadata", {})

            title = metadata.get(
                "title",
                metadata.get("source", "Unknown Policy")
            )

            section = metadata.get(
                "section",
                "Unknown Section"
            )

            snippet = doc.get(
                "document",
                doc.get("text", "")
            )

            context_parts.append(
                f"""
SOURCE {index}
Document: {title}
Section: {section}

Content:
{snippet}
""".strip()
            )

        return "\n\n---\n\n".join(context_parts)

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Execute the complete RAG workflow.
        """

        question = question.strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "citations": [],
                "sources": []
            }

        # --------------------------------------------------
        # 1. RETRIEVAL
        # --------------------------------------------------

        documents = self.retrieve(question)

        # --------------------------------------------------
        # 2. NO CONTEXT → REFUSE
        # --------------------------------------------------

        if not documents:

            return {
                "answer": (
                    "I can only answer questions covered by "
                    "the company policy corpus. "
                    "I could not find relevant policy information."
                ),
                "citations": [],
                "sources": []
            }

        # --------------------------------------------------
        # 3. BUILD CONTEXT
        # --------------------------------------------------

        context = self.build_context(documents)

        # --------------------------------------------------
        # 4. GENERATE ANSWER
        # --------------------------------------------------

        answer = self.generator.generate(
            question=question,
            context=context
        )

        # --------------------------------------------------
        # 5. BUILD CITATIONS
        # --------------------------------------------------

        citations = []

        for doc in documents:

            metadata = doc.get("metadata", {})

            title = metadata.get(
                "title",
                metadata.get("source", "Unknown Policy")
            )

            section = metadata.get(
                "section",
                "Unknown Section"
            )

            document_id = metadata.get(
                "document_id",
                metadata.get("source", "unknown")
            )

            citations.append({
                "document_id": document_id,
                "title": title,
                "section": section
            })

        # --------------------------------------------------
        # 6. BUILD SOURCES FOR UI
        # --------------------------------------------------

        sources = []

        for doc in documents:

            metadata = doc.get("metadata", {})

            source = metadata.get(
                "source",
                metadata.get("title", "Unknown")
            )

            title = metadata.get(
                "title",
                source
            )

            section = metadata.get(
                "section",
                "Unknown Section"
            )

            document_id = metadata.get(
                "document_id",
                source
            )

            snippet = doc.get(
                "document",
                doc.get("text", "")
            )

            sources.append({
                "title": title,
                "document_id": document_id,
                "section": section,
                "source": source,
                "snippet": snippet
            })

        return {
            "answer": answer,
            "citations": citations,
            "sources": sources
        }
