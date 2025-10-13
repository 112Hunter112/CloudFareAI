# Fridge RAG - Simple Setup Guide

## Quick Start (3 Steps)

### Step 1: Setup Virtual Environments (One-time)
```bash
setup_environments.bat
```

### Step 2: Run the Complete Pipeline
```bash
run_pipeline.bat
```

### Step 3: Check the Answer
Open `answer.txt` to see the AI-generated answer!

---

## What Does It Do?

This RAG (Retrieval-Augmented Generation) system:
1. **Reads** your fridge manual PDF
2. **Chunks** the text into smaller pieces
3. **Creates embeddings** (vector representations)
4. **Searches** for relevant chunks based on your query
5. **Generates an answer** using Ollama AI

---

## File Structure

### Main Files
- `fridge_rag.py` - PDF processing, chunking, embeddings
- `step2_ollama_answer.py` - AI answer generation
- `fridge-owners-manual.pdf` - Your manual

### Setup Files
- `setup_environments.bat` - Creates virtual environments
- `run_pipeline.bat` - Runs the complete pipeline
- `requirements_langchain.txt` - Dependencies for Step 1
- `requirements_ollama.txt` - Dependencies for Step 2

### Output Files (auto-generated)
- `chunks.json` - All text chunks
- `embeddings.npy` - Vector embeddings
- `faiss_index.bin` - Search index
- `retrieved_chunks.json` - Retrieved relevant chunks
- `answer.txt` - **Final AI answer**

---

## Manual Running (Alternative)

If you prefer to run each step manually:

### Step 1: Process PDF & Create Embeddings
```bash
venv_langchain\Scripts\activate.bat
python fridge_rag.py
deactivate
```

### Step 2: Generate Answer with Ollama
```bash
venv_ollama\Scripts\activate.bat
python step2_ollama_answer.py
deactivate
```

---

## Customization

### Change the Question
Edit `fridge_rag.py` line 87:
```python
query = "Your question here"
```

### Change Ollama Model
Edit `step2_ollama_answer.py` line 35:
```python
answer = ask_ollama(query, chunks, model_name="your-model")
```

### Adjust Chunk Size
Edit `fridge_rag.py` line 68:
```python
chunks = pdf_chunker("fridge-owners-manual.pdf", chunk_size=1000, overlap=500)
```

---

## Troubleshooting

**Problem: Setup fails**
- Make sure Python is installed
- Run as Administrator if needed

**Problem: Ollama not working**
- Install Ollama: https://ollama.com
- Pull the model: `ollama pull gemma3:12b`
- Make sure Ollama is running

**Problem: Internet connection error**
- sentence-transformers needs to download model on first run
- After first download, it works offline

---

## Why Two Virtual Environments?

- `venv_langchain` uses Pydantic v1 (for LangChain)
- `venv_ollama` uses Pydantic v2 (for Ollama)
- This avoids version conflicts!

---

## Need Help?

Check `README_DUAL_ENV.md` for detailed documentation.
