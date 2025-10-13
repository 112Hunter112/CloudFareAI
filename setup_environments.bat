@echo off
echo ========================================
echo Setting up separate virtual environments
echo ========================================

echo.
echo Creating venv_langchain...
python -m venv venv_langchain
call venv_langchain\Scripts\activate.bat
pip install -r requirements_langchain.txt
call deactivate

echo.
echo Creating venv_ollama...
python -m venv venv_ollama
call venv_ollama\Scripts\activate.bat
pip install -r requirements_ollama.txt
call deactivate

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To use:
echo 1. Run Step 1 (PDF processing): venv_langchain\Scripts\activate.bat ^&^& python fridge_rag.py
echo 2. Run Step 2 (Ollama answer): venv_ollama\Scripts\activate.bat ^&^& python step2_ollama_answer.py
echo.
echo Or run the complete pipeline: run_pipeline.bat
echo.
pause
