@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title IHS Chatbot

echo ============================================================
echo                IHS CHATBOT - STARTUP
echo ============================================================
echo.

set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo ERROR: Python 3.10 or newer was not found.
    echo.
    echo Install Python 3.10+ from python.org and make sure
    echo "Add Python to PATH" is enabled, then run this file again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating isolated Python environment...
    %PY_CMD% -m venv ".venv"
    if errorlevel 1 goto :venv_error
) else (
    echo [1/5] Python environment already exists.
)

set "PYTHON=%CD%\.venv\Scripts\python.exe"

echo [2/5] Updating pip installer...
"%PYTHON%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if errorlevel 1 goto :install_error

echo [3/5] Installing / checking required packages...
"%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :install_error

echo [4/5] Verifying required Python modules...
"%PYTHON%" -c "import fastapi, uvicorn, pydantic, dotenv, pandas, requests, rapidfuzz, gspread, fitz; import langchain_core, langchain_openai, langchain_groq, langchain_ollama, langchain_google_genai, langchain_mcp_adapters, langgraph, fastmcp; print('Dependency check passed.')"
if errorlevel 1 goto :verify_error

echo [5/5] Starting interface at http://127.0.0.1:8000
echo.
echo Keep this window open while using the chatbot.
echo Press CTRL+C here when you want to stop the server.
echo.

start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "$url='http://127.0.0.1:8000/'; for($i=0;$i -lt 90;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1; if($r.StatusCode -ge 200){ Start-Process $url; exit 0 } } catch{}; Start-Sleep -Seconds 1 }; exit 1"

"%PYTHON%" -m uvicorn app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
exit /b 0

:venv_error
echo.
echo ERROR: Could not create the Python virtual environment.
echo Make sure your Python installation includes the venv module.
pause
exit /b 1

:install_error
echo.
echo ERROR: One or more required packages could not be installed.
echo Check your internet connection, then run run_App.bat again.
pause
exit /b 1

:verify_error
echo.
echo ERROR: Package verification failed after installation.
echo Review the messages above, then run run_App.bat again.
pause
exit /b 1
