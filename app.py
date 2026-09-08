"""
Policy RAG Application - Streamlit Web Interface
Main application for querying company policies using RAG
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListCompressor
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenRouter, Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .source-box {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)


class PolicyRAGEngine:
    """Main RAG Engine for Policy Question Answering"""
    
    def __init__(self):
        """Initialize RAG engine with embeddings and vector store"""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize vector store
        self.chroma_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(self.chroma_settings)
        self.collection = self.client.get_or_create_collection(
            name="policies",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        logger.info("PolicyRAGEngine initialized successfully")
    
    def _initialize_llm(self):
        """Initialize LLM with fallback options"""
        api_key = os.getenv("GROQ_API_KEY")
        
        if api_key:
            try:
                return Groq(
                    groq_api_key=api_key,
                    model_name="mixtral-8x7b-32768",
                    temperature=0.7,
                    max_tokens=1024
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        # Fallback to OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            return OpenRouter(
                openrouter_api_key=openrouter_key,
                model="mistralai/mistral-7b-instruct"
            )
        
        raise ValueError("No LLM API key configured. Set GROQ_API_KEY or OPENROUTER_API_KEY")
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """Retrieve relevant documents from vector store"""
        try:
            results = self.collection.query(
                query_embeddings=[self.embedding_model.embed_query(query)],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return []
            
            retrieved = []
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                source = metadata.get("source", "Unknown")
                similarity = 1 - distance  # Convert distance to similarity
                retrieved.append((doc, similarity, source))
            
            return retrieved
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer(
        self, 
        query: str, 
        retrieved_docs: List[Tuple[str, float, str]],
        k: int = 5
    ) -> Tuple[str, List[Dict], float]:
        """Generate answer using retrieved documents"""
        if not retrieved_docs:
            return (
                "I cannot find relevant information in the policy documents to answer your question. "
                "Please try asking about: PTO, security, remote work, expenses, holidays, or code of conduct.",
                [],
                0.0
            )
        
        # Prepare context
        context_text = "\n\n".join([
            f"[Source: {source}]\n{doc}"
            for doc, _, source in retrieved_docs[:k]
        ])
        
        # Create prompt
        prompt = PromptTemplate(
            template="""You are a helpful policy assistant. Answer questions based ONLY on the provided policy documents.

Policy Documents:
{context}

Question: {query}

Important guidelines:
1. Only use information from the provided documents
2. If the answer is not in the documents, say "I don't have information about this"
3. Always cite which policy document your answer comes from
4. Be concise and clear
5. If there are multiple relevant policies, mention all of them

Answer:""",
            input_variables=["context", "query"]
        )
        
        try:
            full_prompt = prompt.format(context=context_text, query=query)
            
            start_time = time.time()
            response = self.llm.invoke(full_prompt)
            latency = time.time() - start_time
            
            # Extract citations
            sources = list(set([source for _, _, source in retrieved_docs[:k]]))
            citations = [
                {
                    "source": source,
                    "document": doc[:500],  # First 500 chars
                    "relevance": float(score)
                }
                for doc, score, source in retrieved_docs[:k]
            ]
            
            return response, citations, latency
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}", [], 0.0


@st.cache_resource
def initialize_rag():
    """Initialize RAG engine (cached)"""
    return PolicyRAGEngine()


def format_citations(citations: List[Dict]) -> str:
    """Format citations for display"""
    if not citations:
        return ""
    
    formatted = "### 📚 Sources\n"
    for i, citation in enumerate(citations, 1):
        formatted += f"\n**{i}. {citation['source']}** (Relevance: {citation['relevance']:.2%})\n"
        formatted += f"```\n{citation['document']}...\n```\n"
    
    return formatted


def main():
    """Main Streamlit application"""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">📋 Policy Assistant</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.clear()
            st.rerun()
    
    st.markdown('<div class="info-box">Ask questions about company policies and procedures. I\'ll search our policy documents and provide accurate, cited answers.</div>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    try:
        rag_engine = initialize_rag()
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.info("Please set GROQ_API_KEY or OPENROUTER_API_KEY in your environment variables")
        return
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = []
    
    # Sidebar - Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        k = st.slider(
            "Number of documents to retrieve (k):",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher k retrieves more documents but may include less relevant results"
        )
        
        temperature = st.slider(
            "Model Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )
        
        st.divider()
        st.subheader("📊 Metrics")
        if st.session_state.metrics:
            avg_latency = np.mean([m["latency"] for m in st.session_state.metrics])
            st.metric("Avg Response Time", f"{avg_latency:.2f}s")
            st.metric("Total Queries", len(st.session_state.metrics))
    
    # Main chat area
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "citations" in message:
                with st.expander("📚 View Sources"):
                    st.markdown(format_citations(message["citations"]))
    
    # Input area
    user_input = st.chat_input("Ask a question about our policies...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.spinner("🔍 Searching policies..."):
            try:
                # Retrieve documents
                retrieved_docs = rag_engine.retrieve_documents(user_input, k=k)
                
                if not retrieved_docs:
                    response = "I couldn't find relevant information in the policy documents. Please try rephrasing your question."
                    citations = []
                    latency = 0.0
                else:
                    # Generate answer
                    response, citations, latency = rag_engine.generate_answer(
                        user_input,
                        retrieved_docs,
                        k=k
                    )
                
                # Record metrics
                st.session_state.metrics.append({
                    "query": user_input,
                    "latency": latency,
                    "k": k,
                    "num_citations": len(citations)
                })
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                response = f"Error processing your query: {str(e)}"
                citations = []
                latency = 0.0
        
        # Display response
        with st.chat_message("assistant"):
            st.write(response)
            
            if citations:
                with st.expander(f"📚 View {len(citations)} Source(s)"):
                    st.markdown(format_citations(citations))
            
            # Display latency
            if latency > 0:
                st.caption(f"⏱️ Response time: {latency:.2f}s")
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "citations": citations,
            "latency": latency
        })
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Policy Assistant | Powered by RAG and LLMs | Questions? Check the documentation</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def health_check() -> Dict:
    """Health check endpoint info"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    main()
