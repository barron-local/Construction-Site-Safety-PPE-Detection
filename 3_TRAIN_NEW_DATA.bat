@echo off
title Construction Site Safety PPE - Auto Fine-Tuning Pipeline
chcp 65001 >nul

echo ======================================================================
echo    CONSTRUCTION PPE DETECTION - AUTOMATED MODEL FINE-TUNING PIPELINE
echo ======================================================================
echo.
echo  [1] สำรองโมเดล Master 200-Epoch อัตโนมัติ (Safety Backup)
echo  [2] ตรวจจับและติดแท็ก Bounding Box อัตโนมัติ (Auto-Annotator)
echo  [3] เริ่มต้น Fine-Tuning ด้วยรูปภาพใหม่ในโฟลเดอร์ New-Training-Images
echo.
echo ======================================================================
echo.

python auto_train_new_images.py %*

echo.
echo ======================================================================
echo  เสร็จสิ้นการทำงาน (กดแป้นใดๆ เพื่อปิดหน้าต่างนี้)
echo ======================================================================
pause >nul
