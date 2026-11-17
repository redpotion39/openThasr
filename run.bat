@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo [OpenThai 2.0 ASR] No audio file specified.
    echo.
    echo Usage:
    echo   run.bat "path\to\audio.wav"
    echo   run.bat "path\to\speech.mp3"
    echo.
    echo Tip: You can drag and drop any audio file directly onto this run.bat!
    echo.
    pause
    exit /b
)

"%~dp0.venv\Scripts\python.exe" "%~dp0asr.py" "%~1"
pause
