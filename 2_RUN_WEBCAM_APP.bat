@echo off
chcp 65001 > nul
title PPE Construction Site Kiosk (Webcam Mode - Port 3333)
echo ========================================================
echo    STARTING PPE KIOSK WEBCAM WEB APPLICATION
echo    Open Browser: http://127.0.0.1:3333 or http://localhost:3333
echo ========================================================
echo.
cd /d "%~dp0Model-Deployment"
python app.py
pause
