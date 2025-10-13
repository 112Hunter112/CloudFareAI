# Environment Setup Guide

Complete guide for setting up the Appliance RAG system with dual virtual environments.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Setup Scripts Overview](#setup-scripts-overview)
4. [Manual Setup Instructions](#manual-setup-instructions)
5. [Automated Setup](#automated-setup)
6. [Verification](#verification)
7. [Common Issues](#common-issues)

---

## Prerequisites

### Required Software

1. **Python 3.12+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked during installation

2. **Ollama**
   - Download from [ollama.com](https://ollama.com)
   - Install the application
   - Pull required model: `ollama pull gemma3:12b`

3. **Git** (Optional, for version control)
   - Download from [git-scm.com](https://git-scm.com/)

### System Requirements

- **OS**: Windows 10/11 (batch scripts provided)
- **RAM**: Minimum 8GB (16GB recommended for larger models)
- **Storage**: 5GB free space for models and dependencies
- **Network**: Internet connection for initial package downloads

---

## Setup Scripts Overview

### setup_environments.bat

**Purpose**: Creates and configures both virtual environments

**What it does**:
1. Creates `venv_langchain` virtual environment
2. Activates `venv_langchain`
3. Installs packages from `requirements_langchain.txt`
4. Deactivates `venv_langchain`
5. Creates `venv_ollama` virtual environment
6. Activates `venv_ollama`
7. Installs packages from `requirements_ollama.txt`
8. Deactivates `venv_ollama`

**Usage**:
```powershell
.\setup_environments.bat
```

### run_pipeline.bat

**Purpose**: Executes the complete RAG pipeline

**What it does**:
1. Activates `venv_langchain`
2. Runs `fridge_rag.py` (PDF processing and embedding)
3. Deactivates `venv_langchain`
4. Activates `venv_ollama`
5. Runs `step2_ollama_answer.py` (Answer generation)
6. Deactivates `venv_ollama`

**Usage**:
```powershell
.\run_pipeline.bat
```

---

## Manual Setup Instructions

If you prefer manual setup or the batch scripts fail, follow these steps:

### Step 1: Create venv_langchain

```bash
# Create virtual environment
python -m venv venv_langchain

# Activate environment
venv_langchain\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements_langchain.txt

# Verify installation
python -c "from langchain.text_splitter import RecursiveCharacterTextSplitter; print('Success')"

# Deactivate
deactivate
```

### Step 2: Create venv_ollama

```bash
# Create virtual environment
python -m venv venv_ollama

# Activate environment
venv_ollama\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements_ollama.txt

# Verify installation
python -c "import ollama; print('Success')"

# Deactivate
deactivate
```

### Step 3: Verify Ollama Setup

```bash
# Check Ollama is running
ollama list

# Pull required model if not present
ollama pull gemma3:12b

# Verify model works
ollama run gemma3:12b "Hello"
```

---

## Automated Setup

### Using PowerShell

```powershell
# Navigate to project directory
cd C:\Users\YourName\PycharmProjects\Appliance_RAG

# Run setup (requires '.\'  prefix in PowerShell)
.\setup_environments.bat

# Wait for completion (may take 5-10 minutes)
```

### Using Command Prompt (CMD)

```cmd
# Navigate to project directory
cd C:\Users\YourName\PycharmProjects\Appliance_RAG

# Run setup (no prefix needed in CMD)
setup_environments.bat

# Wait for completion
```

---

## Verification

### Verify venv_langchain

```powershell
# Activate environment
venv_langchain\Scripts\activate.bat

# Check installed packages
pip list

# Test imports
python -c "from pypdf import PdfReader; from sentence_transformers import SentenceTransformer; import faiss; from langchain.text_splitter import RecursiveCharacterTextSplitter; print('All imports successful')"

# Deactivate
deactivate
```

**Expected packages**:
- sentence-transformers
- faiss-cpu
- pypdf
- langchain==0.1.20
- pydantic<2.0
- tiktoken
- numpy<2

### Verify venv_ollama

```powershell
# Activate environment
venv_ollama\Scripts\activate.bat

# Check installed packages
pip list

# Test imports
python -c "import ollama; print('Ollama import successful')"

# Test Ollama connection
python -c "import ollama; print(ollama.list())"

# Deactivate
deactivate
```

**Expected packages**:
- ollama
- pydantic>=2.9

### Test Complete Pipeline

```powershell
# Run complete pipeline
.\run_pipeline.bat

# Check for output files
dir chunks.json
dir embeddings.npy
dir faiss_index.bin
dir retrieved_chunks.json
dir answer.txt

# View answer
type answer.txt
```

---

## Common Issues

### Issue: "Python not recognized"

**Cause**: Python not in system PATH

**Solution**:
```powershell
# Check Python installation
where python

# If not found, add Python to PATH manually:
# 1. Search "Environment Variables" in Windows
# 2. Edit System Environment Variables
# 3. Add Python installation directory to PATH
# 4. Restart terminal
```

### Issue: "pip install fails with SSL error"

**Cause**: Network/firewall restrictions

**Solution**:
```bash
# Use trusted host flag
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements_langchain.txt

# Or configure proxy if needed
pip install --proxy http://proxy:port -r requirements_langchain.txt
```

### Issue: "Virtual environment activation fails"

**Cause**: PowerShell execution policy

**Solution**:
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy to allow scripts (run as Administrator)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Retry activation
venv_langchain\Scripts\activate.bat
```

### Issue: "FAISS installation fails"

**Cause**: Missing C++ compiler on Windows

**Solution**:
```bash
# Install CPU version (no compilation needed)
pip install faiss-cpu

# Or install Visual C++ Build Tools:
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Issue: "Ollama connection refused"

**Cause**: Ollama service not running

**Solution**:
```bash
# Start Ollama service
ollama serve

# In new terminal, verify
ollama list

# If model missing
ollama pull gemma3:12b
```

### Issue: "sentence-transformers model download hangs"

**Cause**: Network timeout or slow connection

**Solution**:
```bash
# Set longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Or manually download model:
# 1. Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# 2. Download files to: ~/.cache/huggingface/
```

### Issue: "Import error after installation"

**Cause**: Wrong environment activated

**Solution**:
```bash
# Verify which environment is active
python -c "import sys; print(sys.prefix)"

# Should show path to venv_langchain or venv_ollama
# If not, deactivate and reactivate correct environment

deactivate
venv_langchain\Scripts\activate.bat
```

---

## Environment Variables (Optional)

Create `.env` file in project root for customization:

```bash
# .env file (copy from .env.example)

# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# Embedding settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=500

# FAISS settings
FAISS_INDEX_PATH=faiss_index.bin
VECTOR_DIMENSION=384

# LangChain settings
LANGCHAIN_TRACING=false
```

---

## Maintenance

### Updating Dependencies

```bash
# Update venv_langchain packages
venv_langchain\Scripts\activate.bat
pip install --upgrade -r requirements_langchain.txt
deactivate

# Update venv_ollama packages
venv_ollama\Scripts\activate.bat
pip install --upgrade -r requirements_ollama.txt
deactivate
```

### Rebuilding Environments

```bash
# Remove old environments
rmdir /s /q venv_langchain
rmdir /s /q venv_ollama

# Recreate
.\setup_environments.bat
```

### Cleaning Generated Files

```bash
# Remove generated artifacts
del chunks.json
del embeddings.npy
del faiss_index.bin
del retrieved_chunks.json
del answer.txt

# Regenerate by running pipeline
.\run_pipeline.bat
```

---

## Next Steps

After successful setup:

1. **Test the system**: Run `.\run_pipeline.bat`
2. **Check output**: View `answer.txt`
3. **Customize query**: Edit `fridge_rag.py` line 83
4. **Explore configuration**: See [Configuration](README.md#configuration)
5. **Read documentation**: Review [README.md](README.md)

---

## Support

For additional help:
- Review main [README.md](README.md)
- Check [Troubleshooting](README.md#troubleshooting) section
- Consult [Technical Documentation](README_DUAL_ENV.md)

---

**Setup Guide Version:** 1.0
**Last Updated:** October 2025
