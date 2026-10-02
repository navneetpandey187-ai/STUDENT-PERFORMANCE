@echo off
title EduMetrics AI - Student Performance Analyzer
cd /d "%~dp0"
echo ==============================================================
echo  Launching EduMetrics AI - Student Performance Analyzer
echo ==============================================================
echo.
python -m streamlit run app.py
pause
