# ==============================================================================
# Script ติดตั้ง Environment & Dependencies ทั้งหมดของโปรเจกต์ PPE Detection
# รันด้วยคำสั่ง: python setup_environment.py
# ==============================================================================
import sys
import subprocess

def run_cmd(cmd_list, description):
    print(f"\n[*] {description}...")
    try:
        subprocess.check_call(cmd_list)
        print(f"[✓] {description} สำเร็จ")
    except subprocess.CalledProcessError as e:
        print(f"[!] เกิดข้อผิดพลาดในขั้นตอนนี้: {e}")
        return False
    return True

def main():
    print("=" * 65)
    print("  PPE Detection System - One-Click Environment Setup")
    print(f"  Python Version: {sys.version.split()[0]}")
    print("=" * 65)

    # 1. Upgrade pip
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "อัปเดต pip เป็นเวอร์ชันล่าสุด")

    # 2. Install PyTorch with CUDA 12.1
    run_cmd([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], "ติดตั้ง PyTorch + TorchVision พร้อมการรองรับ NVIDIA CUDA 12.1 (GPU)")

    # 3. Install requirements.txt
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "ติดตั้งไลบรารีทั้งหมดจาก requirements.txt")

    # 4. Verify installation
    print("\n" + "=" * 65)
    print("  ตรวจสอบความพร้อมของระบบ (Hardware & Acceleration Check)")
    print("=" * 65)
    try:
        import torch
        print(f"[+] PyTorch: {torch.__version__}")
        print(f"[+] CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] Active GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[!] รันบน CPU (หากมี GPU NVIDIA แนะนำติดตั้งไดรเวอร์การ์ดจอ)")
        
        import ultralytics
        print(f"[+] Ultralytics YOLO: {ultralytics.__version__}")
        import cv2
        print(f"[+] OpenCV: {cv2.__version__}")
        import flask
        print(f"[+] Flask: {flask.__version__}")
        import waitress
        print(f"[+] Waitress WSGI: {waitress.__version__}")
        
        print("\n[SUCCESS] ติดตั้งสภาพแวดล้อมทั้งหมดเรียบร้อยแล้ว!")
        print("พร้อมใช้งาน! สามารถรันระบบได้ผ่าน RUN_PRODUCTION_SERVER.bat หรือ python Model-Deployment/app.py")
    except Exception as e:
        print(f"[!] เกิดข้อผิดพลาดในการตรวจสอบ: {e}")

if __name__ == '__main__':
    main()
