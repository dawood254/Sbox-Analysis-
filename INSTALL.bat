@echo off
title S-Box Analyzer — Installer
color 0A
echo.
echo  ============================================================
echo   S-BOX CRYPTOGRAPHIC ANALYZER  ^|  INSTALLER
echo  ============================================================
echo.

:: ── Step 1: Verify Python ────────────────────────────────────────────────
echo  [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python is not installed or not on PATH.
    echo.
    echo  Please install Python 3.9 or newer from:
    echo     https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During installation tick "Add Python to PATH"
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  Found: %%v
echo.

:: ── Step 2: Upgrade pip ──────────────────────────────────────────────────
echo  [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip ready.
echo.

:: ── Step 3: Install required libraries ───────────────────────────────────
echo  [3/5] Installing required libraries (PyQt5, numpy)...
echo        This may take 1-3 minutes on first install.
echo.
python -m pip install PyQt5 numpy
if errorlevel 1 (
    echo.
    echo  ERROR: Library installation failed.
    echo  Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo.
echo  Libraries installed successfully.
echo.

:: ── Step 4: Verify installation ──────────────────────────────────────────
echo  [4/5] Verifying installation...
python -c "from PyQt5.QtWidgets import QApplication; import numpy; print('  PyQt5 OK  |  numpy OK')"
if errorlevel 1 (
    echo  ERROR: Verification failed. Something went wrong.
    pause
    exit /b 1
)
echo.

:: ── Step 5: Create shortcuts ─────────────────────────────────────────────
echo  [5/5] Creating shortcuts...

set "APP_DIR=%~dp0"
set "APP_PY=%APP_DIR%SBoxAnalyzer.py"
set "DESKTOP=%USERPROFILE%\Desktop"
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

:: Desktop shortcut
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $s = $ws.CreateShortcut('%DESKTOP%\S-Box Analyzer.lnk'); ^
   $s.TargetPath = 'python'; ^
   $s.Arguments = '\"%APP_PY%\"'; ^
   $s.WorkingDirectory = '%APP_DIR%'; ^
   $s.Description = 'S-Box Cryptographic Analyzer'; ^
   $s.Save()"

if exist "%DESKTOP%\S-Box Analyzer.lnk" (
    echo  Desktop shortcut created.
) else (
    echo  [WARN] Could not create desktop shortcut ^(optional^).
)

:: Start menu shortcut
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $s = $ws.CreateShortcut('%STARTMENU%\S-Box Analyzer.lnk'); ^
   $s.TargetPath = 'python'; ^
   $s.Arguments = '\"%APP_PY%\"'; ^
   $s.WorkingDirectory = '%APP_DIR%'; ^
   $s.Description = 'S-Box Cryptographic Analyzer'; ^
   $s.Save()"

if exist "%STARTMENU%\S-Box Analyzer.lnk" (
    echo  Start Menu shortcut created.
) else (
    echo  [WARN] Could not create Start Menu shortcut ^(optional^).
)

echo.
echo  ============================================================
echo   INSTALLATION COMPLETE!
echo  ============================================================
echo.
echo  How to run the software:
echo    Option A: Double-click "S-Box Analyzer" on your Desktop
echo    Option B: Double-click "LAUNCH.bat" in this folder
echo    Option C: python SBoxAnalyzer.py  (from this folder)
echo.
echo  To build a standalone .exe (no Python needed to run):
echo    Double-click BUILD_EXE.bat
echo.
set /p RUN="  Launch the software now? [Y/N]: "
if /i "%RUN%"=="Y" (
    echo  Starting S-Box Analyzer...
    start "" python "%APP_PY%"
)
echo.
pause
