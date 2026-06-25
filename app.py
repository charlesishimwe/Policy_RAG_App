import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI   # you can replace with Groq/OpenRouter if needed


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Policy RAG App", layout="wide")

PDF_PATH = "policies.pdf"   # put your PDF in project root
CHROMA_DIR = "chroma_db"


# ----------------------------
# LLM (FREE OPTION FRIENDLY)
# ----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ----------------------------
# EMBEDDINGS
# ----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------------
# LOAD + SPLIT DOCUMENTS
# ----------------------------
@st.cache_resource
def load_docs():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(docs)


# ----------------------------
# VECTOR DB
# ----------------------------
@st.cache_resource
def get_vectorstore():
    docs = load_docs()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectorstore


vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ----------------------------
# PROMPT (IMPORTANT FOR RAG + CITATIONS)
# ----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a policy assistant.

Answer ONLY using the context below.
If you don't know, say: "I can only answer based on the policy documents."

Always include:
- bullet point answer
- source reference if available

Context:
{context}

Question:
{input}
""")


# ----------------------------
# RAG CHAIN (NEW LANGCHAIN WAY)
# ----------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📄 Company Policy RAG Chatbot")

query = st.text_input("Ask a question about company policies:")

if st.button("Ask"):
    if query:
        result = rag_chain.invoke({"input": query})

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for doc in result["context"]:
            st.write("📌", doc.metadata.get("source", "Unknown"))
            st.write(doc.page_content[:300])
            st.markdown("---")
