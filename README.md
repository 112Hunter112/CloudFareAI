# Appliance RAG - Intelligent Document Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system for answering questions about appliance manuals using advanced natural language processing and large language models.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Future Roadmap](#future-roadmap)
- [Troubleshooting](#troubleshooting)
- [Technical Stack](#technical-stack)

---

## Overview

This RAG system processes appliance owner manuals in PDF format and provides accurate, context-aware answers to user queries. The system leverages semantic search, vector embeddings, and large language models to deliver precise information extraction from technical documentation.

**Current Implementation:** Refrigerator LED replacement query system

---

## Features

### Core Capabilities
- PDF document processing and text extraction
- Intelligent text chunking with semantic awareness
- Vector embeddings using state-of-the-art transformer models
- High-performance similarity search via FAISS
- Local LLM inference using Ollama
- Persistent embedding storage for efficient reuse

### Technical Highlights
- Dual virtual environment architecture to resolve dependency conflicts
- Modular pipeline design for easy customization
- Scalable vector database integration
- Context-aware answer generation

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────┐
│  Stage 1: Document Processing (venv_langchain)  │
│  ├── PDF Loading & Text Extraction             │
│  ├── Semantic Text Chunking (LangChain)        │
│  ├── Embedding Generation (Sentence-BERT)      │
│  ├── FAISS Index Construction                  │
│  └── Query-based Chunk Retrieval               │
│                                                 │
│  Output: retrieved_chunks.json                 │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Answer Generation (venv_ollama)      │
│  ├── Context Loading                           │
│  ├── Prompt Engineering                        │
│  ├── LLM Inference (Ollama)                    │
│  └── Response Generation                       │
│                                                 │
│  Output: answer.txt                            │
└─────────────────────────────────────────────────┘
```

### Dependency Management

The system employs a dual virtual environment strategy:

- **venv_langchain**: Utilizes Pydantic v1.x (compatible with LangChain 0.1.x)
- **venv_ollama**: Utilizes Pydantic v2.x (required by Ollama)

This architecture eliminates version conflicts while maintaining full functionality of both components.

---

## Installation

### Prerequisites

- Python 3.12 or higher
- Ollama installed locally ([Download](https://ollama.com))
- Git (for version control)
- Windows OS (batch scripts provided)

### System Setup

**Step 1: Clone Repository**
```bash
git clone <repository-url>
cd Appliance_RAG
```

**Step 2: Install Ollama Model**
```bash
ollama pull gemma3:12b
```

**Step 3: Initialize Virtual Environments**
```powershell
.\setup_environments.bat
```

This script will:
- Create `venv_langchain` with appropriate dependencies
- Create `venv_ollama` with Ollama requirements
- Install all necessary packages in isolated environments

---

## Usage

### Quick Start

Execute the complete pipeline:
```powershell
.\run_pipeline.bat
```

### Step-by-Step Execution

**Stage 1: Document Processing**
```powershell
venv_langchain\Scripts\activate.bat
python fridge_rag.py
deactivate
```

**Stage 2: Answer Generation**
```powershell
venv_ollama\Scripts\activate.bat
python step2_ollama_answer.py
deactivate
```

### Output

Results are saved to `answer.txt` in the project root directory.

---

## Project Structure

```
Appliance_RAG/
│
├── Core Scripts
│   ├── fridge_rag.py              # Document processing pipeline
│   └── step2_ollama_answer.py     # Answer generation module
│
├── Setup & Execution
│   ├── setup_environments.bat     # Environment initialization
│   └── run_pipeline.bat           # Complete pipeline execution
│
├── Configuration Files
│   ├── requirements.txt           # Legacy single-environment deps
│   ├── requirements_langchain.txt # Stage 1 dependencies
│   └── requirements_ollama.txt    # Stage 2 dependencies
│
├── Documentation
│   ├── README.md                  # Main documentation (this file)
│   ├── README_SIMPLE.md           # Quick start guide
│   ├── README_DUAL_ENV.md         # Technical deep-dive
│   └── SETUP_GUIDE.md             # Environment setup guide
│
├── Source Documents
│   └── fridge-owners-manual.pdf   # Input document
│
├── Virtual Environments (auto-generated)
│   ├── venv_langchain/            # Stage 1 environment
│   └── venv_ollama/               # Stage 2 environment
│
└── Generated Artifacts (excluded from git)
    ├── chunks.json                # Processed text chunks
    ├── embeddings.npy             # Vector representations
    ├── faiss_index.bin            # Search index
    ├── retrieved_chunks.json      # Retrieved context
    └── answer.txt                 # Final output
```

---

## Configuration

### Query Modification

Edit `fridge_rag.py` at line 83:
```python
query = "How do I replace the LED lights?"
```

### LLM Model Selection

Edit `step2_ollama_answer.py` at line 35:
```python
answer = ask_ollama(query, chunks, model_name="gemma3:12b")
```

Available models: Run `ollama list` to view installed models

### Chunking Parameters

Edit `fridge_rag.py` at line 68:
```python
chunks = pdf_chunker(
    "fridge-owners-manual.pdf",
    chunk_size=1000,    # Characters per chunk
    overlap=500         # Overlap between chunks
)
```

### Embedding Model

Edit `fridge_rag.py` at line 42:
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
```

Alternative models:
- `all-mpnet-base-v2` (higher quality, slower)
- `paraphrase-MiniLM-L3-v2` (faster, lower quality)

---

## Future Roadmap

### Phase 1: Core System Enhancements
- Multi-document indexing and retrieval
- Query history persistence and analytics
- Interactive CLI with conversation context
- Confidence scoring for retrieved chunks
- Source attribution with page references

### Phase 2: Interface Development
- RESTful API with FastAPI
- Web-based user interface
- Conversational chat interface
- Integrated PDF viewer with highlighting
- Multi-format export (Markdown, JSON, PDF)

### Phase 3: Advanced NLP Features
- Multi-modal document processing (images, diagrams)
- Speech-to-text query interface
- Multi-LLM provider support (OpenAI, Anthropic, Cohere)
- Hybrid search (semantic + keyword)
- Cross-encoder re-ranking
- Real-time streaming responses

### Phase 4: Production Deployment
- Containerized deployment (Docker)
- Cloud infrastructure (AWS/GCP/Azure)
- Vector database migration (Pinecone/Weaviate/Qdrant)
- Multi-tenant authentication system
- API rate limiting and quotas
- Monitoring, logging, and analytics
- Automated document update pipeline

### Phase 5: AI/ML Improvements
- Domain-specific fine-tuning
- Advanced prompt engineering and optimization
- Multi-agent orchestration system
- Automated fact-checking and verification
- Explainable AI for answer reasoning
- Active learning from user feedback

### Phase 6: Extended Ecosystem
- Intelligent troubleshooting wizard
- Computer vision for parts identification
- Automated maintenance scheduling
- Integration with video tutorial platforms
- Community knowledge base
- Internationalization and localization

---

## Troubleshooting

### Environment Setup

**Issue: Virtual environment creation fails**
```bash
# Verify Python installation
python --version

# Ensure pip is updated
python -m pip install --upgrade pip
```

**Issue: Package installation errors**
```bash
# Clear pip cache
pip cache purge

# Reinstall with verbose logging
pip install -r requirements_langchain.txt --verbose
```

### Runtime Errors

**Issue: Ollama connection failure**
```bash
# Start Ollama server
ollama serve

# Verify model availability
ollama list
ollama pull gemma3:12b
```

**Issue: Sentence-transformers model download fails**
- Ensure stable internet connection (required for initial download)
- Model cache location: `~/.cache/huggingface/`
- Verify firewall/proxy settings

**Issue: FAISS installation fails (Windows)**
```bash
# Install CPU-optimized version
pip install faiss-cpu
```

### Performance Optimization

**Issue: Slow embedding generation**
- Reduce chunk_size parameter
- Use lighter embedding model
- Process fewer chunks in parallel

**Issue: High LLM latency**
- Switch to smaller model (`gemma:2b`)
- Reduce retrieved chunk count (top_k parameter)
- Optimize prompt length

---

## Technical Stack

### Core Libraries
- **LangChain 0.1.20** - Document processing and text chunking
- **Sentence-Transformers** - Semantic embedding generation
- **FAISS** - High-performance vector similarity search
- **Ollama** - Local large language model inference
- **PyPDF** - PDF parsing and text extraction

### Supporting Technologies
- **NumPy** - Numerical computing
- **Pydantic** - Data validation
- **Tiktoken** - Tokenization for chunk splitting

---

## Additional Resources

- [Setup Guide](SETUP_GUIDE.md) - Detailed environment setup instructions
- [Quick Start](README_SIMPLE.md) - Beginner-friendly guide
- [Technical Documentation](README_DUAL_ENV.md) - Architecture deep-dive

---

## License

[Specify License]

---

## Acknowledgments

This project builds upon the following open-source technologies:
- LangChain Framework
- Hugging Face Transformers
- Facebook AI Similarity Search (FAISS)
- Ollama Project
- PyPDF Library

---

**Last Updated:** October 2025
