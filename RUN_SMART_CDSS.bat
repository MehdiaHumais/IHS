@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title SMART Clinic Launcher

set "SMART_PORT=8503"
set "DIAG_PORT=8600"
set "SMART_APP=%CD%\app_patient_diagnostics_fixed.py"
set "DIAG_DIR=%CD%\general_diagnostics"
set "DIAG_APP=%DIAG_DIR%\app.py"
set "SMART_URL=http://127.0.0.1:%SMART_PORT%"
set "VENV_DIR=%CD%\.venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "INSTALL_MARKER=%VENV_DIR%\.smart_clinic_packages_installed_v3"
set "LOG_DIR=%CD%\logs"
set "SMART_LOG=%LOG_DIR%\smart_clinic.log"

echo ============================================================
echo                    SMART CLINIC
echo ============================================================
echo.
echo Launcher folder: %CD%
echo Main interface:  %SMART_URL%
echo.

if not exist "%SMART_APP%" (
    echo ERROR: Main application file is missing:
    echo   %SMART_APP%
    goto :fatal
)

if not exist "%CD%\requirements.txt" (
    echo ERROR: requirements.txt is missing.
    goto :fatal
)

if not exist "%DIAG_APP%" (
    echo ERROR: Full Diagnostics app is missing:
    echo   %DIAG_APP%
    goto :fatal
)

if not exist "%DIAG_DIR%\requirements.txt" (
    echo ERROR: Full Diagnostics requirements.txt is missing.
    goto :fatal
)

rem ------------------------------------------------------------
rem Find a usable system Python. Prefer the Windows py launcher.
rem ------------------------------------------------------------
set "SYSTEM_PYTHON="
where py.exe >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "SYSTEM_PYTHON=py -3"
)

if not defined SYSTEM_PYTHON (
    where python.exe >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set "SYSTEM_PYTHON=python"
    )
)

if not defined SYSTEM_PYTHON (
    echo ERROR: Python 3.10 or newer was not found.
    echo.
    echo Install Python from python.org and enable "Add python.exe to PATH",
    echo then double-click this BAT file again.
    goto :fatal
)

rem ------------------------------------------------------------
rem Never trust a copied/pre-packaged virtual environment.
rem Verify it actually runs on this PC; recreate it if it does not.
rem ------------------------------------------------------------
if exist "%PYTHON%" (
    "%PYTHON%" -c "import sys; print(sys.executable)" >nul 2>nul
    if errorlevel 1 (
        echo Existing .venv belongs to another Python installation.
        echo Rebuilding the local environment...
        rmdir /s /q "%VENV_DIR%" >nul 2>nul
    )
)

if not exist "%PYTHON%" (
    echo [1/4] Creating a clean local Python environment...
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        goto :fatal
    )
)

rem ------------------------------------------------------------
rem Verify essential imports every launch. A marker alone is not enough.
rem If anything is missing, install/repair requirements.
rem ------------------------------------------------------------
set "NEEDS_INSTALL=0"
if not exist "%INSTALL_MARKER%" set "NEEDS_INSTALL=1"

if "!NEEDS_INSTALL!"=="0" (
    "%PYTHON%" -c "import streamlit,pandas,gspread,sklearn,joblib,plotly,reportlab,numpy,PIL,cv2,skimage,fastapi,uvicorn,dotenv,requests,fitz" >nul 2>nul
    if errorlevel 1 set "NEEDS_INSTALL=1"
)

if "!NEEDS_INSTALL!"=="0" (
    pushd "%DIAG_DIR%"
    "%PYTHON%" -c "import streamlit,pandas,numpy,PyPDF2,docx,pydicom,plotly,matplotlib,requests; from agents.readmission_predictor import ReadmissionPredictor" >nul 2>nul
    if errorlevel 1 set "NEEDS_INSTALL=1"
    popd
)

if "!NEEDS_INSTALL!"=="1" (
    echo [2/4] Installing or repairing required packages...
    "%PYTHON%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    if errorlevel 1 goto :install_error

    echo.
    echo Installing SMART Clinic packages...
    "%PYTHON%" -m pip install --disable-pip-version-check -r "%CD%\requirements.txt"
    if errorlevel 1 goto :install_error

    echo.
    echo Installing Full Diagnostics packages...
    "%PYTHON%" -m pip install --disable-pip-version-check -r "%DIAG_DIR%\requirements.txt"
    if errorlevel 1 goto :install_error

    echo.
    echo Verifying package consistency...
    "%PYTHON%" -m pip check
    if errorlevel 1 goto :install_error

    >"%INSTALL_MARKER%" echo Packages installed and verified on %DATE% %TIME%
) else (
    echo [2/4] Required packages are already installed and verified.
)

rem Final import test before starting the UI.
echo [3/4] Checking the application environment...
"%PYTHON%" -c "import streamlit,pandas,sklearn,gspread; print('Environment OK')" >nul 2>nul
if errorlevel 1 goto :install_error

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
>"%SMART_LOG%" echo SMART Clinic startup log - %DATE% %TIME%

rem Close old listeners from a previous crashed copy.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ports=%SMART_PORT%,%DIAG_PORT%; foreach($port in $ports){ Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { if($_ -and $_ -ne $PID){ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } } }" >nul 2>nul

set "AGENTIC_DIAGNOSTIC_DIR=%DIAG_DIR%"
set "AGENTIC_DIAGNOSTIC_PORT=%DIAG_PORT%"

echo [4/4] Starting SMART Clinic interface...
start "SMART Clinic Server" /min cmd /c ""%PYTHON%" -m streamlit run "%SMART_APP%" --server.address 127.0.0.1 --server.port %SMART_PORT% --server.headless true --browser.gatherUsageStats false >>"%SMART_LOG%" 2>&1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url='%SMART_URL%/_stcore/health'; for($i=0;$i -lt 120;$i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 1; if($r.StatusCode -eq 200){ exit 0 } } catch{}; Start-Sleep -Seconds 1 }; exit 1"

if errorlevel 1 goto :startup_error

start "" "%SMART_URL%"
echo.
echo ============================================================
echo SMART Clinic started successfully.
echo Browser: %SMART_URL%
echo Log:     %SMART_LOG%
echo ============================================================
timeout /t 4 /nobreak >nul
exit /b 0

:install_error
echo.
echo ============================================================
echo ERROR: Required Python packages could not be installed.
echo ============================================================
echo.
echo Check your internet connection and the messages above.
echo You can safely run this BAT again; it will repair incomplete setup.
goto :fatal

:startup_error
echo.
echo ============================================================
echo ERROR: The interface did not start successfully.
echo ============================================================
echo.
echo Last startup messages:
echo ------------------------------------------------------------
powershell -NoProfile -Command "if(Test-Path '%SMART_LOG%'){Get-Content '%SMART_LOG%' -Tail 60}"
echo ------------------------------------------------------------
echo Full log:
echo   %SMART_LOG%
goto :fatal

:fatal
echo.
echo Press any key to close this window.
pause >nul
exit /b 1
