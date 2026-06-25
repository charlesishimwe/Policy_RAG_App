import os
import streamlit as st
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"   # put PDFs here
DB_DIR = BASE_DIR / "chroma_db"

st.set_page_config(page_title="Policy RAG Chat App", layout="wide")


# =========================
# SAFE INGESTION (FIXED)
# =========================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # If DB exists → load it
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    # Otherwise create it safely
    docs = []

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if len(pdf_files) == 0:
        st.warning("⚠️ No PDFs found in /data folder. Please add policy documents.")
        return None

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR)
    )

    vectordb.persist()
    return vectordb


# =========================
# LLM (FREE OPTION SAFE)
# =========================
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )


# =========================
# QA PIPELINE
# =========================
def get_qa_chain(vectordb):
    llm = get_llm()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa


# =========================
# UI
# =========================
def main():

    st.title("📄 Policy RAG Chat App")
    st.caption("Ask anything about company policies")

    vectordb = load_vectorstore()

    if vectordb is None:
        st.stop()

    qa_chain = get_qa_chain(vectordb)

    query = st.text_input("💬 Ask your question:")

    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})

            answer = result["result"]
            sources = result.get("source_documents", [])

        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📌 Sources")
        for i, doc in enumerate(sources):
            st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'unknown')}")
            st.text(doc.page_content[:300])


# =========================
# HEALTH CHECK (STREAMLIT STYLE)
# =========================
if __name__ == "__main__":
    main()
