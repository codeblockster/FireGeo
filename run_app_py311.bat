@echo off
echo Activating Python 3.11 environment...
call venv_py311\Scripts\activate.bat

echo Starting Streamlit App...
streamlit run frontend/app.py
pause
