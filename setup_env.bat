@echo off
echo Creating a new environment with Python 3.11 for Wildfire App...

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% equ 0 (
    echo Conda detected. Creating 'wildfire_env' with Python 3.11...
    call conda create -n wildfire_env python=3.11 -y
    echo.
    echo Environment created. To activate and install dependencies, run:
    echo conda activate wildfire_env
    echo pip install -r requirements.txt
    echo.
    echo Then run the app with:
    echo streamlit run frontend/app.py
    goto :eof
)

REM Check if py launcher is available
where py >nul 2>nul
if %errorlevel% equ 0 (
    echo Python Launcher (py) detected. Creating venv with Python 3.11...
    py -3.11 -m venv venv_py311
    if %errorlevel% neq 0 (
        echo.
        echo Error: Could not create venv with Python 3.11. 
        echo Please ensure Python 3.11 is installed from python.org.
        echo Download: https://www.python.org/downloads/release/python-3119/
    ) else (
        echo.
        echo Environment created in 'venv_py311'.
        echo To use it, run:
        echo .\venv_py311\Scripts\activate
        echo pip install -r requirements.txt
    )
    goto :eof
)

echo.
echo No Conda or Python Launcher found.
echo Please manually install Python 3.11 from python.org and try again.
pause
