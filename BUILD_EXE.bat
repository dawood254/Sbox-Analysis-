@echo off
title S-Box Analyzer — Build Standalone .exe
color 0B
echo.
echo  ============================================================
echo   S-BOX ANALYZER  ^|  BUILD STANDALONE .EXE
echo  ============================================================
echo  Builds dist\SBoxAnalyzer.exe (no Python required to run it)
echo.

:: Install build/runtime dependencies
echo  [1/4] Installing PyInstaller and runtime dependencies...
python -m pip install --upgrade pyinstaller PyQt5 numpy --quiet
if errorlevel 1 (
    echo  ERROR: pip install failed.
    pause & exit /b 1
)
echo  Done.
echo.

:: Remove old build artifacts
echo  [2/4] Cleaning old build...
if exist "dist\SBoxAnalyzer.exe" del /q "dist\SBoxAnalyzer.exe"
if exist "build\SBoxAnalyzer" rmdir /s /q "build\SBoxAnalyzer"

:: Build the EXE
echo  [3/4] Building .exe (takes 2-5 minutes)...
python -m PyInstaller --clean --noconfirm SBoxAnalyzer.spec

if errorlevel 1 (
    echo.
    echo  ERROR: PyInstaller build failed. See output above.
    pause & exit /b 1
)

echo.
echo  [4/4] Done!
echo.
echo  ============================================================
echo   Output: dist\SBoxAnalyzer.exe
echo  ============================================================
echo.
echo  To build the installer, run:
echo    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" SBoxAnalyzerInstaller.iss
echo.
pause
