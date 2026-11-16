@echo off
chcp 65001 > nul
title Installing PPE Detection Dependencies...
echo ========================================================
echo    INSTALLING PPE DETECTION DEPENDENCIES FOR COLLEGE DEMO
echo ========================================================
echo.
pip install ultralytics opencv-python flask flask-socketio eventlet pillow openvino lap
echo.
echo ========================================================
echo [+] Installation Complete! You are ready to run.
echo ========================================================
pause
