# Automotive Financial Analyst - RAG Application

A Retrieval-Augmented Generation (RAG) application for analyzing automotive company annual reports (BMW, Tesla, Ford) using **LLM APIs**.

## Project Overview

Built for Consigli Technical Interview - A production-ready RAG system that demonstrates:
- Advanced document processing and retrieval
- Integration with free, high-performance LLM APIs
- Professional Streamlit interface for presentations
- Docker containerization
- Cost-effective architecture decisions

## Features

-  **Interactive Chat Interface**: Natural conversation with follow-up questions
-  **Multi-Company Analysis**: Compare BMW, Tesla, and Ford
-  **Time-Series Data**: Analyze trends from 2021-2023
-  **Source Citations**: Transparent answers with document references
-  **Fast Responses**: Sub-2 second query processing with Groq
-  **Zero Cost**: Uses free API tiers (Groq/HuggingFace)
-  **Presentation-Ready**: Clean, professional Streamlit UI

## Architecture

```
User Query
    ↓
[Streamlit UI]
    ↓
[RAG Engine]
    ↓
┌─────────────┴──────────────┐
│                            │
[Retrieval]              [Generation]
    ↓                        ↓
[FAISS Search]         [Groq/HF API]
    ↓                        
[sentence-transformers]      
    ↓
[Document Chunks]
```

### Tech Stack:

**LLM Providers:**
- **Groq API** (Recommended): Llama 3.1 70B - 300-600 tokens/sec
- **HuggingFace API** (Backup): Llama 3.1 8B - Reliable fallback

**Local Components:**
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (CPU-optimized)
- **Framework**: LangChain
- **Interface**: Streamlit
- **Processing**: PyPDF2

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- API key (Groq or HuggingFace)

### 1. Get API Key (30 seconds)

**Option A: Groq**
```bash
# Visit: https://console.groq.com/
# Sign up → API Keys → Create Key
# Copy key (starts with gsk_)
```

**Option B: HuggingFace (Alternative)**
```bash
# Visit: https://huggingface.co/settings/tokens
# Create token → Copy (starts with hf_)
```

### 2. Setup Project

```bash
# Navigate to project
cd C:\Users\Jem\Documents\GitHub\Consigli_3

# Activate virtual environment
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
# Create .env file
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your key:
GROQ_API_KEY=gsk_your_actual_key_here
# OR
HF_API_KEY=hf_your_actual_key_here
```

### 4. Run Application

```bash
streamlit run main.py
```

Navigate to `http://localhost:8501`

## Project Structure

```
Consigli_1/
├── main.py                      # Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF processing & chunking
│   ├── vector_store.py          # FAISS vector store
│   └── rag_engine.py            # RAG orchestration
├── data/                        # Annual reports
│   ├── BMW/
│   │   ├── BMW_Annual_Report_2021.pdf
│   │   ├── BMW_Annual_Report_2022.pdf
│   │   └── BMW_Annual_Report_2023.pdf
│   ├── Ford/
│   │   ├── Ford_Annual_Report_2021.pdf
│   │   ├── Ford_Annual_Report_2022.pdf
│   │   └── Ford_Annual_Report_2023.pdf
│   └── Tesla/
│       ├── Tesla_Annual_Report_2022.pdf
│       └── Tesla_Annual_Report_2023.pdf
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
├── .gitignore
├── README.md
├── FREE_MODELS_GUIDE.md        # Detailed free models setup
├── SETUP_GUIDE.md              # Complete setup instructions
└── TECHNICAL_DOCUMENTATION.md  # Architecture details
```

## Sample Questions

Try these:

**Basic Queries:**
- "What was BMW's total revenue in 2023?"
- "How much revenue did Tesla generate in 2023?"

**Comparisons:**
- "Between Tesla and Ford, which company achieved higher profits in 2022?"
- "Compare the revenue of all three companies in 2023"

**Analysis:**
- "What key economic factors influenced Ford's performance in 2021?"
- "Provide a summary of revenue figures for Tesla, BMW, and Ford over the past three years"

**Follow-ups:**
- Ask a question, then: "How does that compare to 2022?"
- Or: "What about the other companies?"

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# Or build manually
docker build -t automotive-rag .
docker run -p 8501:8501 --env-file .env automotive-rag

# Access at http://localhost:8501
```
## Troubleshooting

**Application won't start:**
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**API errors:**
```bash
# Verify API key in .env
cat .env  # Linux/Mac
type .env  # Windows

# Test API connection
python -c "from groq import Groq; print('✓ Groq works')"
```

**Slow responses:**
```bash
# Switch to faster model in rag_engine.py:
model="llama-3.1-8b-instant"

# Reduce retrieved chunks:
search_kwargs={"k": 3}
```

## Acknowledgments

- LangChain for RAG framework
- Groq for fast, LLM inference
- HuggingFace for model hosting
- Streamlit for beautiful UI
- FAISS for efficient vector search

---

## Quick Commands Reference

```bash
# Setup
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# Run
streamlit run main.py

# Docker
docker-compose up --build

# Test
python -c "import streamlit; import langchain; print('Ready')"
```
