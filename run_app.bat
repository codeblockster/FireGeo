@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   FIRE ALERT - Wildfire Detection System
echo ========================================
echo.

REM Get the project root directory (where this batch file is located)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=!PROJECT_ROOT:~0,-1!"

echo [1/5] Setting up Python environment...
echo.

REM First check for venv in local project folder, then F:\v4 cleanup
set "VENV_PATH="

if exist "%PROJECT_ROOT%\venv_py311\Scripts\activate.bat" (
    set "VENV_PATH=%PROJECT_ROOT%\venv_py311"
    echo Found: %PROJECT_ROOT%\venv_py311
) else if exist "F:\v4 cleanup\venv_py311\Scripts\activate.bat" (
    set "VENV_PATH=F:\v4 cleanup\venv_py311"
    echo Found: F:\v4 cleanup\venv_py311
) else (
    echo ERROR: venv_py311 not found!
    echo Please ensure the virtual environment exists.
    pause
    exit /b 1
)

echo Using venv: %VENV_PATH%
echo.

echo [2/5] Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"
echo Virtual environment activated.
echo.

echo [3/5] Starting Backend Server (main.py) on port 8000...
echo.

REM Change to backend directory and start the backend
cd /d "%PROJECT_ROOT%\backend"
start "Backend Server" cmd /k "cd /d "%PROJECT_ROOT%\backend" && python main.py"

timeout /t 6 /nobreak >nul

echo [4/5] Starting Frontend Server on port 5173...
echo.

REM Start frontend in a new window
start "Frontend Server" cmd /k "cd /d "%PROJECT_ROOT%\frontend" && npm run dev"

timeout /t 5 /nobreak >nul

echo [5/5] All servers started!
echo.
echo ========================================
echo   SYSTEM READY - All Services Running
echo ========================================
echo.
echo   Backend:    http://localhost:8000       ^(FastAPI API^)
echo   Health:     http://localhost:8000/health
echo   API Docs:   http://localhost:8000/docs   ^(Swagger UI^)
echo   Frontend:   http://localhost:5173       ^(Web UI^)
echo.
echo ========================================
echo   Servers started in separate windows
echo   Close each window to stop that server
echo ========================================
echo.
echo Press any key to exit...
pause >nul
