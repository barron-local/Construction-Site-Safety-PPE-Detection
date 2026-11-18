@echo off
title PPE Detection System - Production Server
cd /d "%~dp0Model-Deployment"

echo ===================================================
echo   Starting PPE Detection System (Production WSGI)
echo   Host: http://localhost:3333
echo ===================================================

python run_production.py
pause
