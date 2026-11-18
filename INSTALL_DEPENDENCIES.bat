@echo off
chcp 65001 >nul
title ติดตั้ง Dependencies สำหรับระบบตรวจจับ PPE (One-Click Setup)
color 0B

echo ================================================================
echo    ระบบตรวจจับความปลอดภัยและอุปกรณ์ PPE อัตโนมัติ (AI-PPE)
echo            สคริปต์ติดตั้งไลบรารีและ Dependencies ทั้งหมด
echo ================================================================
echo.

:: 1. ตรวจสอบว่ามี Python อยู่ในเครื่องหรือไม่
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] ไม่พบ Python ในเครื่อง! กรุณาติดตั้ง Python 3.10 ขึ้นไป และติ๊ก "Add Python to PATH"
    echo.
    pause
    exit /b
)

echo [+] ตรวจพบ Python:
python --version
echo.

:: 2. อัปเกรด pip ให้เป็นเวอร์ชันล่าสุด
echo [1/3] อัปเดตเครื่องมือ pip...
python -m pip install --upgrade pip
echo.

:: 3. ติดตั้ง PyTorch รองรับ GPU (NVIDIA CUDA 12.1)
echo [2/3] กำลังติดตั้ง PyTorch + Torchvision (CUDA 12.1 GPU Acceleration)...
echo (ขั้นตอนนี้อาจใช้เวลาสักครู่ขึ้นอยู่กับความเร็วอินเทอร์เน็ต)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.

:: 4. ติดตั้ง Libraries ทั้งหมดจาก requirements.txt
echo [3/3] กำลังติดตั้งไลบรารีที่จำเป็นทั้งหมด (Flask, OpenCV, YOLO, Waitress, pyserial, qrcode)...
python -m pip install -r requirements.txt
echo.

:: 5. ตรวจสอบสถานะการเชื่อมต่อ CUDA / GPU
echo ================================================================
echo                 ตรวจสอบสถานะการทำงานของระบบ
echo ================================================================
python -c "import torch; print('[+] PyTorch Version :', torch.__version__); print('[+] CUDA Available   :', torch.cuda.is_available()); print('[+] Active GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only')"
echo.

echo ================================================================
echo    [SUCCESS] ติดตั้ง Dependencies ทั้งหมดเรียบร้อยแล้ว!
echo    สามารถรันโปรเจกต์ได้เลยด้วยไฟล์: RUN_PRODUCTION_SERVER.bat
echo ================================================================
echo.
pause
