# RAG-based Q&A System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions 
from PDF documents using LangChain, ChromaDB, and LLMs (OpenAI & Groq).

## 🚀 Features
- Load and process PDF documents automatically
- Split documents into optimized chunks for retrieval
- Generate vector embeddings using SentenceTransformers
- Store and retrieve vectors using ChromaDB
- Answer questions using OpenAI GPT or Groq (Qwen-32B)
- Cosine similarity-based document ranking

## 🛠️ Tech Stack
- **LangChain** — document loading, chunking, LLM integration
- **ChromaDB** — vector storage and retrieval
- **SentenceTransformers** — embedding generation (all-MiniLM-L6-v2)
- **OpenAI API** — GPT-based answer generation
- **Groq API** — Qwen-32B based answer generation
- **PyPDF** — PDF parsing
- **Python** — core language

## 📁 Project Structure
RAG-based-QA-System/
│
├── data/
│   ├── pdfs/              # Place your PDF files here
│   │   ├── research1.pdf
│   │   └── research2.pdf
│   └── vector_store/      # ChromaDB persisted vector store
│
├── RAG-based QA System.ipynb   # Main notebook
└── README.md

## ⚙️ Installation

```bash
pip install langchain langchain-core langchain-community
pip install pypdf pymupdf
pip install sentence-transformers
pip install chromadb
pip install langchain-openai
pip install langchain-groq
```

## 🔑 API Keys Setup

In the notebook, replace with your actual API keys:

```python
# OpenAI
API_KEY_OPENAI = "your-openai-api-key"

# Groq
API_Key_GROQ = "your-groq-api-key"
```

## 🔄 Pipeline Overview

PDFs → Document Loading → Chunking → Embedding → ChromaDB
↓
User Query → Embedding → Semantic Search → Top-K Chunks
↓
LLM (OpenAI / Groq) → Answer

## 📊 Pipeline Details

| Step | Tool | Config |
|------|------|--------|
| Document Loading | PyPDFLoader | 2 PDFs, 32 pages |
| Chunking | RecursiveCharacterTextSplitter | size=500, overlap=50 |
| Embedding | all-MiniLM-L6-v2 | 384 dimensions |
| Vector Store | ChromaDB | 320 chunks indexed |
| Retrieval | Cosine Similarity | top-k=5 |
| LLM | OpenAI GPT / Groq Qwen-32B | temp=0.1 |

## 💬 Example Usage

```python
# Ask a question
answer = generate_output("What is encoder-decoder?", rag_retriever, llm)
print(answer)
```

**Sample Output:**
An encoder-decoder is a neural network architecture with two main parts:
  -Encoder: processes the input and converts it into internal representations
  -Decoder: takes the encoder output and generates the target output

## 📚 Research Papers Indexed
- Attention is All You Need — Vaswani et al. (2017)
- RAG Survey Paper

## 🔮 Future Improvements
- Add Streamlit UI for easy interaction
- Support more file formats (DOCX, TXT)
- Deploy on AWS EC2 with FastAPI
- Add conversation memory for multi-turn Q&A

## 👤 Author
**Biswanath Bhuyan**  
AI/ML Engineer | [LinkedIn](https://www.linkedin.com/in/biswanath-bhuyan/) 
