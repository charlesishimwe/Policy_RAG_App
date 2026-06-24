
# Policy RAG Application

## Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about company policies and procedures and receive accurate, source-cited answers.

The system ingests policy documents, converts them into embeddings, stores them in a vector database, retrieves relevant information based on user questions, and generates responses using a Large Language Model (LLM).

This project was developed as part of the Quantic AI Engineering Project.

---

## Features

- Document ingestion and indexing
- PDF, TXT, Markdown document support
- Text chunking with overlap
- Vector embeddings using Sentence Transformers
- ChromaDB vector database
- Retrieval-Augmented Generation (RAG)
- Source citations included in responses
- Streamlit web interface
- Health check endpoint
- GitHub Actions CI/CD workflow
- Deployable to Render or Railway

---

## Project Architecture

```text
policy_rag_app/
│
├── app.py
├── ingest.py
├── requirements.txt
├── README.md
├── data/
│   └── policies/
│
├── chroma_db/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── utils/
```

---

## Technology Stack

### Backend

- Python 3.11+
- Streamlit
- LangChain
- ChromaDB

### Embeddings

- Sentence Transformers
- all-MiniLM-L6-v2

### Vector Database

- ChromaDB

### LLM

- OpenRouter
- Groq
- OpenAI Compatible APIs

### Deployment

- Render
- Railway

### CI/CD

- GitHub Actions

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/charlesishimwe/policy_rag_app.git
cd policy_rag_app
```

### 2. Create Virtual Environment

Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key
GROQ_API_KEY=your_api_key
```

---

## Document Ingestion

Place policy documents inside:

```text
data/policies/
```

Run ingestion:

```bash
python ingest.py
```

This process:

- Reads documents
- Splits text into chunks
- Generates embeddings
- Stores vectors in ChromaDB

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Application URL:

```text
http://localhost:8501
```

---

## Example Questions

Users can ask questions such as:

- What is the PTO policy?
- How many vacation days do employees receive?
- What is the remote work policy?
- What are the password security requirements?
- What expenses are reimbursable?

---

## Retrieval-Augmented Generation Workflow

1. User submits a question
2. Question is embedded
3. Relevant document chunks are retrieved from ChromaDB
4. Retrieved context is sent to the LLM
5. LLM generates a grounded answer
6. Sources and citations are displayed

---

## CI/CD

GitHub Actions automatically runs on:

- Push
- Pull Request

Workflow tasks:

```bash
pip install -r requirements.txt
python -c "import app"
```

Workflow file:

```text
.github/workflows/ci.yml
```

---

## Deployment

### Render

1. Push repository to GitHub
2. Create a new Render Web Service
3. Connect GitHub repository
4. Add environment variables
5. Deploy

### Railway

1. Create Railway project
2. Connect GitHub repository
3. Add environment variables
4. Deploy

---

## Evaluation Metrics

The application is evaluated using:

### Answer Quality

- Groundedness
- Citation Accuracy
- Answer Relevance

### System Metrics

- Response Latency
- Retrieval Speed
- End-to-End Response Time

---

## Design Decisions

| Component | Choice |
|------------|---------|
| Embeddings | all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Frontend | Streamlit |
| Deployment | Render / Railway |
| CI/CD | GitHub Actions |

---

## Future Improvements

- Hybrid search
- Re-ranking
- Conversation memory
- User authentication
- Multi-document source highlighting
- Advanced evaluation dashboards

---

## Author

**Charles Ishimwe Hagenimana**

AI Engineer | Data Scientist | Software Engineer

GitHub:
https://github.com/charlesishimwe

LinkedIn:
Add your LinkedIn profile here

---

## License

This project is for educational purposes as part of the Quantic AI Engineering Program.
