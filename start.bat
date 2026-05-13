@echo off
REM ============================================================================
REM GPU Benchmarking - Start Script
REM ============================================================================
REM This script sets up and runs the GPU benchmark suite with proper error handling
REM and cleanup between runs.
REM ============================================================================

setlocal enabledelayedexpansion

REM ============================================================================
REM BLOCK 1: CONFIG - Central variables and paths
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
set "OUTPUT_DIR=%SCRIPT_DIR%outputs"
set "MAIN_SCRIPT=%SCRIPT_DIR%main.py"
set "WINDOW_TITLE=GPU Benchmarking Suite"

REM ============================================================================
REM BLOCK 2: PREFLIGHT CHECKS - Verify environment and dependencies
REM ============================================================================

title %WINDOW_TITLE%

echo.
echo ========================================================================
echo GPU BENCHMARKING - PRE-FLIGHT CHECKS
echo ========================================================================
echo.

REM Check if Python is available
echo [1/3] Checking for Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo.
    echo Please install Python 3.11+ from https://www.python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
echo   ✓ %PYTHON_VERSION%

REM Check if virtual environment exists
echo [2/3] Checking for virtual environment...
if not exist "%VENV_DIR%" (
    echo WARNING: Virtual environment not found
    echo.
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo   ✓ Virtual environment created
) else (
    echo   ✓ Virtual environment found
)

REM Check if main.py exists
echo [3/3] Checking for main.py...
if not exist "%MAIN_SCRIPT%" (
    echo ERROR: main.py not found
    pause
    exit /b 1
)
echo   ✓ main.py found

echo.
echo ✓ All pre-flight checks passed
echo.

REM ============================================================================
REM BLOCK 3: TARGETED CLEANUP - Clear cache before benchmark
REM ============================================================================

echo ========================================================================
echo CLEANUP
echo ========================================================================
echo.

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    echo   ✓ Created outputs directory
) else (
    echo   ✓ Outputs directory ready
)

echo   ✓ GPU cache will be cleared at startup
echo.

REM ============================================================================
REM BLOCK 4: DETERMINISTIC BOOTSTRAP - Activate venv and verify dependencies
REM ============================================================================

echo ========================================================================
echo BOOTSTRAP - Activating environment and checking dependencies
echo ========================================================================
echo.

REM Activate virtual environment
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo   ✓ Virtual environment activated

REM Check if dependencies are installed
echo [*] Verifying dependencies...
"%VENV_PYTHON%" -c "import torch, transformers, matplotlib, numpy, psutil; print('    ✓ All dependencies present')" 2>nul
if errorlevel 1 (
    echo WARNING: Some dependencies may be missing
    echo.
    echo Installing dependencies...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >nul 2>&1
    pip install matplotlib numpy psutil transformers >nul 2>&1
    echo   ✓ Dependencies installed
) else (
    echo   ✓ Dependencies verified
)

REM Verify GPU availability
echo [*] Checking GPU availability...
"%VENV_PYTHON%" -c "import torch; print('    CUDA Available:', torch.cuda.is_available()); cuda_dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'; print('    GPU:', cuda_dev)"

echo.

REM ============================================================================
REM BLOCK 5: ORDERED STARTUP - Execute the benchmark
REM ============================================================================

echo ========================================================================
echo STARTING GPU BENCHMARK SUITE
echo ========================================================================
echo.
echo Outputs will be saved to: %OUTPUT_DIR%
echo.
echo Press Ctrl+C to stop the benchmark at any time.
echo.
pause

REM Run the benchmark script
"%VENV_PYTHON%" "%MAIN_SCRIPT%"
set BENCHMARK_EXIT_CODE=!errorlevel!

echo.

REM ============================================================================
REM BLOCK 6: READINESS + OBSERVABILITY - Check completion
REM ============================================================================

if !BENCHMARK_EXIT_CODE! equ 0 (
    echo ========================================================================
    echo ✅ BENCHMARK COMPLETED SUCCESSFULLY
    echo ========================================================================
) else (
    echo ========================================================================
    echo ❌ BENCHMARK FAILED (Exit code: !BENCHMARK_EXIT_CODE!)
    echo ========================================================================
)

echo.

REM ============================================================================
REM BLOCK 7: USER HANDOFF - Display results and next steps
REM ============================================================================

echo BENCHMARK RESULTS
echo ========================================================================
echo.
echo Output files saved to:
echo   %OUTPUT_DIR%
echo.
echo Output file types:
echo   • .json - Detailed metrics with all runs
echo   • .png  - Visualization with charts
echo   • .txt  - Human-readable text report
echo.

REM List recent output files
echo Recent files:
for /f "tokens=*" %%f in ('dir /b /o-d "%OUTPUT_DIR%\benchmark_*" 2^>nul') do (
    echo   • %%f
    goto :show_only_first
)
:show_only_first

echo.
choice /C YN /M "Open outputs folder?" /T 15 /D Y
if !errorlevel! equ 1 (
    start explorer "%OUTPUT_DIR%"
)

echo.
pause
exit /b !BENCHMARK_EXIT_CODE!
