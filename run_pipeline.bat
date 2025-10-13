@echo off
echo ========================================
echo Running RAG Pipeline with Separate Environments
echo ========================================

echo.
echo Step 1: Processing PDF and creating embeddings (venv_langchain)...
call venv_langchain\Scripts\activate.bat
python fridge_rag.py
call deactivate

echo.
echo Step 2: Generating answer with Ollama (venv_ollama)...
call venv_ollama\Scripts\activate.bat
python step2_ollama_answer.py
call deactivate

echo.
echo ========================================
echo Pipeline complete! Check answer.txt for results.
echo ========================================
pause
