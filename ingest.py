mport os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Clean old DB if exists
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Load all .md files manually
policy_dir = Path("./data/policies")
docs = []
for md_file in policy_dir.glob("*.md"):
    loader = TextLoader(str(md_file), encoding='utf-8')
    docs.extend(loader.load())

print(f"Loaded {len(docs)} documents.")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Success! Ingested {len(chunks)} chunks into ChromaDB.")
