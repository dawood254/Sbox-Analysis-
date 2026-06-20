@echo off
:: Quick launcher — double-click to open S-Box Analyzer
cd /d "%~dp0"
python SBoxAnalyzer.py
if errorlevel 1 (
    echo.
    echo  Error starting application.
    echo  Run INSTALL.bat first to install required libraries.
    echo.
    pause
)
