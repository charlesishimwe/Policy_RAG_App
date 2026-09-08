"""
Policy RAG Application
A Retrieval-Augmented Generation system for answering company policy questions.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq, openrouter, openai
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Free, fast, high quality
    LLM_MODEL = "mixtral-8x7b-32768"  # Groq free tier (128k context)
    
    # RAG Configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    RETRIEVAL_K = 5
    TEMPERATURE = 0.3  # Lower = more deterministic
    MAX_TOKENS = 1024
    
    # Vector Store
    CHROMA_DB_PATH = "./chroma_db"
    POLICIES_PATH = "./policies"
    
    # Evaluation
    EVALUATION_SET_PATH = "./data/evaluation_set.json"

config = Config()

# ============================================================================
# VECTOR STORE INITIALIZATION
# ============================================================================

class VectorStore:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, db_path: str, embedding_model_name: str):
        """Initialize vector store"""
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB
        settings = Settings(
            chroma_db_impl="duckdb",
            persist_directory=db_path,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        self.collection = None
        
    def get_or_create_collection(self, name: str = "policies"):
        """Get or create collection"""
        try:
            self.collection = self.client.get_collection(name=name)
            logger.info(f"Loaded existing collection: {name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {name}")
        return self.collection
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict],
                     ids: List[str]):
        """Add documents to vector store"""
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents"""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                retrieved.append({
                    "content": doc,
                    "source": metadata.get("source", "Unknown"),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "relevance_score": 1 - distance  # Convert distance to similarity
                })
        
        return retrieved

# ============================================================================
# RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline for policy QA"""
    
    def __init__(self, vector_store: VectorStore, config: Config):
        """Initialize RAG pipeline"""
        self.vector_store = vector_store
        self.config = config
        self.llm_endpoint = self._get_llm_endpoint()
        
    def _get_llm_endpoint(self) -> Tuple[str, str, str]:
        """Get LLM API endpoint and headers"""
        if config.LLM_PROVIDER == "groq":
            return (
                "https://api.groq.com/openai/v1/chat/completions",
                config.GROQ_API_KEY,
                "groq"
            )
        elif config.LLM_PROVIDER == "openrouter":
            return (
                "https://openrouter.ai/api/v1/chat/completions",
                config.OPENROUTER_API_KEY,
                "openrouter"
            )
        elif config.LLM_PROVIDER == "openai":
            return (
                "https://api.openai.com/v1/chat/completions",
                config.OPENAI_API_KEY,
                "openai"
            )
        else:
            raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant documents"""
        return self.vector_store.retrieve(query, k=self.config.RETRIEVAL_K)
    
    def generate_answer(self, 
                       query: str, 
                       context: List[Dict]) -> Tuple[str, List[Dict]]:
        """Generate answer using LLM"""
        
        if not context:
            return (
                "I cannot find relevant information in the company policies to answer your question. "
                "Please rephrase your question or contact HR for assistance.",
                []
            )
        
        # Build context string
        context_text = "\n\n".join([
            f"[Source: {c['source']}]\n{c['content']}"
            for c in context
        ])
        
        # Build prompt
        system_prompt = """You are a helpful company policy assistant. Your role is to:
1. Answer questions about company policies based ONLY on provided context
2. Always cite the source document for your answer
3. Be accurate and never make up policy information
4. If the answer is not in the context, clearly state this
5. Keep answers concise and professional

Important: Only answer based on the provided context. Do not use external knowledge about policies."""
        
        user_prompt = f"""Based on the following company policy documents, answer this question:

Question: {query}

Context from policies:
{context_text}

Please provide a clear, accurate answer with proper citations."""
        
        # Call LLM
        endpoint, api_key, provider = self.llm_endpoint
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.config.TEMPERATURE,
            "max_tokens": self.config.MAX_TOKENS
        }
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            return answer, context
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API error: {e}")
            return f"Error generating answer: {str(e)}", context
    
    def answer_question(self, query: str) -> Tuple[str, List[Dict], float]:
        """Answer a question using RAG pipeline"""
        start_time = time.time()
        
        # Retrieve
        context = self.retrieve(query)
        
        # Generate
        answer, retrieved = self.generate_answer(query, context)
        
        latency = time.time() - start_time
        
        return answer, retrieved, latency

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def init_session_state():
    """Initialize session state"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore(
            config.CHROMA_DB_PATH,
            config.EMBEDDING_MODEL
        )
        st.session_state.vector_store.get_or_create_collection()
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            st.session_state.vector_store,
            config
        )
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def render_header():
    """Render application header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏢 Policy Assistant")
        st.markdown("Ask questions about company policies and get instant, source-cited answers.")
    with col2:
        st.metric("LLM", config.LLM_PROVIDER.upper())

def render_chat_interface():
    """Render chat interface"""
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")
        
        for i, (question, answer, sources, latency) in enumerate(st.session_state.chat_history):
            with st.container():
                col1, col2 = st.columns([1, 10])
                with col1:
                    st.markdown("❓")
                with col2:
                    st.markdown(f"**Q:** {question}")
                
                st.markdown(f"**A:** {answer}")
                
                if sources:
                    with st.expander("📎 Sources"):
                        for source in sources:
                            st.markdown(f"""
- **Source:** {source['source']}
- **Relevance:** {source['relevance_score']:.2%}
- **Excerpt:** {source['content'][:200]}...
""")
                
                st.caption(f"⏱️ Latency: {latency:.2f}s")
                st.markdown("---")
    
    # Input section
    st.markdown("---")
    st.subheader("Ask a Question")
    
    question = st.text_input(
        "Enter your question about company policies:",
        placeholder="e.g., 'What is the PTO policy?' or 'How many vacation days do I get?'"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        submit_btn = st.button("📤 Submit", use_container_width=True)
    
    with col2:
        clear_btn = st.button("🗑️ Clear History", use_container_width=True)
    
    with col3:
        health_btn = st.button("💚 Health Check", use_container_width=True)
    
    # Process question
    if submit_btn and question:
        with st.spinner("🔍 Searching policies and generating answer..."):
            try:
                answer, sources, latency = st.session_state.rag_pipeline.answer_question(question)
                
                # Add to history
                st.session_state.chat_history.append((question, answer, sources, latency))
                
                st.success("✅ Answer generated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if clear_btn:
        st.session_state.chat_history = []
        st.info("✨ Chat history cleared!")
        st.rerun()
    
    if health_btn:
        st.info("✅ Application is healthy and running!")

def render_sidebar():
    """Render sidebar information"""
    with st.sidebar:
        st.markdown("## 📊 Configuration")
        
        st.markdown(f"""
- **LLM Provider:** {config.LLM_PROVIDER}
- **Model:** {config.LLM_MODEL}
- **Embedding Model:** {config.EMBEDDING_MODEL}
- **Vector DB:** ChromaDB
- **Chunk Size:** {config.CHUNK_SIZE}
- **Retrieval K:** {config.RETRIEVAL_K}
- **Temperature:** {config.TEMPERATURE}
        """)
        
        st.markdown("---")
        st.markdown("## 📚 About")
        st.markdown("""
This is a Retrieval-Augmented Generation (RAG) application that:
1. Ingests company policy documents
2. Converts them into embeddings
3. Stores them in a vector database
4. Retrieves relevant information for user queries
5. Generates grounded answers with citations

**Project:** Quantic AI Engineering Program
**Author:** Charles Ishimwe
        """)
        
        st.markdown("---")
        st.markdown("## 🔗 Links")
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("GitHub", "https://github.com/charlesishimwe/policy_rag_app")
        with col2:
            st.link_button("Documentation", "https://github.com/charlesishimwe/policy_rag_app#readme")

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Policy Assistant",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render components
    render_header()
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()
