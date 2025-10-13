# Dual Environment Setup for RAG Pipeline

This setup solves the Pydantic version conflict between LangChain (requires v1) and Ollama (requires v2) by using two separate virtual environments.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Step 1: venv_langchain (Pydantic v1)          │
│  - PDF processing                               │
│  - Text chunking with LangChain                 │
│  - Embeddings creation                          │
│  - FAISS indexing                               │
│  - Chunk retrieval                              │
│  Output: retrieved_chunks.json                  │
└─────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  Step 2: venv_ollama (Pydantic v2)             │
│  - Load retrieved chunks                        │
│  - Generate answer with Ollama LLM              │
│  Output: answer.txt                             │
└─────────────────────────────────────────────────┘
```

## Files

### Main Scripts
- `step1_process_chunks.py` - PDF processing, chunking, embeddings (runs in venv_langchain)
- `step2_ollama_answer.py` - Ollama answer generation (runs in venv_ollama)

### Requirements
- `requirements_langchain.txt` - Dependencies for venv_langchain
- `requirements_ollama.txt` - Dependencies for venv_ollama

### Setup Scripts
- `setup_environments.bat` - Creates both virtual environments
- `run_pipeline.bat` - Runs the complete pipeline

### Intermediate Files (auto-generated)
- `chunks.json` - All text chunks from PDF
- `embeddings.npy` - Embeddings array
- `faiss_index.bin` - FAISS index for vector search
- `retrieved_chunks.json` - Retrieved chunks passed to Step 2
- `answer.txt` - Final answer from Ollama

## Setup Instructions

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
setup_environments.bat
```

### Option 2: Manual Setup
```bash
# Create venv_langchain
python -m venv venv_langchain
venv_langchain\Scripts\activate.bat
pip install -r requirements_langchain.txt
deactivate

# Create venv_ollama
python -m venv venv_ollama
venv_ollama\Scripts\activate.bat
pip install -r requirements_ollama.txt
deactivate
```

## Usage

### Option 1: Run Complete Pipeline
```bash
run_pipeline.bat
```

### Option 2: Run Steps Manually

**Step 1: Process PDF and create embeddings**
```bash
venv_langchain\Scripts\activate.bat
python step1_process_chunks.py
deactivate
```

**Step 2: Generate answer with Ollama**
```bash
venv_ollama\Scripts\activate.bat
python step2_ollama_answer.py
deactivate
```

## Environment Details

### venv_langchain (Pydantic v1)
- sentence-transformers
- faiss-cpu
- pypdf
- langchain==0.1.20
- pydantic<2.0
- tiktoken
- numpy<2

### venv_ollama (Pydantic v2)
- ollama
- pydantic>=2.9

## Benefits of This Approach

✅ **No Version Conflicts** - Each environment has compatible dependencies
✅ **Modular Design** - Easy to modify or replace components
✅ **Reusable Embeddings** - Step 1 output can be reused for multiple queries
✅ **Clean Separation** - PDF processing and LLM inference are isolated
✅ **Easy Debugging** - Test each step independently

## Customization

### Change the Query
Edit `step1_process_chunks.py` and modify:
```python
query = "Your question here"
```

### Change Ollama Model
Edit `step2_ollama_answer.py` and modify:
```python
answer = ask_ollama(query, chunks, model_name="your-model-name")
```

### Adjust Chunking Parameters
Edit `step1_process_chunks.py`:
```python
chunks = pdf_chunker("fridge-owners-manual.pdf", chunk_size=1000, overlap=500)
```

## Troubleshooting

**Issue: sentence-transformers model download fails**
- Ensure you have internet connection
- Model will be cached after first download

**Issue: Ollama connection error**
- Make sure Ollama is running locally
- Check if the model is installed: `ollama list`
- Pull the model if needed: `ollama pull gemma3:12b`

**Issue: Virtual environment activation fails**
- Ensure Python is installed and in PATH
- Try using absolute paths to activate scripts
