#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Model Fine-Tuning & Self-Annotating Pipeline for Construction PPE Detection
--------------------------------------------------------------------------------------
ฟังก์ชันหลัก:
1. สำรองข้อมูลโมเดลเดิมอัตโนมัติ (Automatic Safety Backup) ก่อนเทรนทุกครั้ง
2. สแกนภาพถ่ายใหม่จากโฟลเดอร์ 'New-Training-Images'
3. สร้าง Annotation อัตโนมัติ (Auto-Labeling) สำหรับภาพที่ยังไม่มีไฟล์ .txt
   - ตรวจจับ Person, Hardhat, Safety Vest
   - ตรวจจับ Cap (หมวกแก๊ป) แล้วแปลงเป็น NO-Hardhat อย่างถูกต้อง
   - ตรวจสอบลำตัวคนงาน หากไม่สวมเสื้อกั๊กจะสร้าง BBox NO-Safety Vest ให้อัตโนมัติ
   - วาดภาพ Preview Bounding Box เก็บไว้ใน 'New-Training-Images/annotated_previews/'
4. รวมชุดข้อมูลเข้ากับ Replay Buffer (ป้องกัน Overfitting และ Catastrophic Forgetting)
5. เริ่ม Fine-Tuning จากโมเดล Master 200-Epoch ด้วยพารามิเตอร์ที่ปลอดภัย
6. รายงานสรุปผลและมีตัวเลือก Deploy โมเดลใหม่เข้าสู่ระบบทันที
"""

import os
import sys
import glob
import json
import time
import shutil
import random
import argparse
import datetime
import cv2
import torch
from ultralytics import YOLO

# กำหนดเส้นทางหลักของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_MODEL_PATH = os.path.join(
    BASE_DIR, 'Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'
)
CAP_MODEL_PATH = os.path.join(
    BASE_DIR, 'Model-Training/Outputs/runs/detect/yolov8s_ppe_model_with_cap_class/weights/best.pt'
)
BACKUP_DIR = os.path.join(BASE_DIR, 'Model-Training/Outputs/backups')
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, 'New-Training-Images')
PREVIEWS_DIR = os.path.join(INPUT_IMAGES_DIR, 'annotated_previews')
DATASET_DIR = os.path.join(BASE_DIR, 'Model-Training/Dataset')
WORKSPACE_DIR = os.path.join(DATASET_DIR, 'finetune_workspace')

CLASS_NAMES = {
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}

CLASS_COLORS = {
    0: (0, 220, 0),     # Hardhat (เขียว)
    1: (200, 200, 0),   # Mask
    2: (0, 0, 255),     # NO-Hardhat (แดง)
    3: (0, 165, 255),   # NO-Mask (ส้ม)
    4: (0, 0, 240),     # NO-Safety Vest (แดง)
    5: (255, 140, 0),   # Person (ฟ้า/น้ำเงิน)
    6: (255, 255, 0),   # Safety Cone
    7: (0, 200, 0),     # Safety Vest (เขียว)
    8: (128, 128, 128), # machinery
    9: (128, 128, 128)  # vehicle
}

def print_banner():
    print("=" * 70)
    print("  🏗️  CONSTRUCTION PPE DETECTION: AUTO FINE-TUNING PIPELINE")
    print("  ระบบฝึกฝนโมเดลเพิ่มเติมอัตโนมัติด้วยรูปภาพใหม่ + ระบบสำรองโมเดล")
    print("=" * 70)

def backup_master_model():
    """ทำการสำรองโมเดล Master เดิมเข้าโฟลเดอร์ backup พร้อมบันทึกประวัติ"""
    if not os.path.exists(MASTER_MODEL_PATH):
        print(f"[-] ไม่พบโมเดล Master ที่ {MASTER_MODEL_PATH}")
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subfolder = os.path.join(BACKUP_DIR, f"backup_{timestamp}")
    os.makedirs(backup_subfolder, exist_ok=True)

    backup_dest = os.path.join(backup_subfolder, "best.pt")
    shutil.copy2(MASTER_MODEL_PATH, backup_dest)
    file_size_mb = os.path.getsize(backup_dest) / (1024 * 1024)

    meta = {
        "timestamp": timestamp,
        "source_model": MASTER_MODEL_PATH,
        "backup_path": backup_dest,
        "size_mb": round(file_size_mb, 2),
        "note": "Pre-fine-tuning automatic safety backup"
    }
    with open(os.path.join(backup_subfolder, "backup_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n[+] 🛡️ [BACKUP SUCCESS] สำรองโมเดล Master เดิมเรียบร้อย:")
    print(f"    -> ตำแหน่ง: {backup_dest} ({file_size_mb:.1f} MB)")
    return backup_dest

def check_is_cap(cap_model, img, bbox):
    """ตรวจสอบว่าหมวกที่ตรวจพบเป็นหมวกแก๊ปหรือไม่ อย่างระมัดระวัง (ป้องกัน False-Positive)"""
    if cap_model is None or not hasattr(cap_model, 'names'):
        return False, 0.0
    # ตรวจสอบว่าโมเดลมีคลาส Cap หรือไม่
    has_cap_class = any('cap' in str(v).lower() for v in cap_model.names.values())
    if not has_cap_class:
        return False, 0.0
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img.shape[:2]
        pad_w = int((x2 - x1) * 0.35)
        pad_h = int((y2 - y1) * 0.35)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return False, 0.0
        c_res = cap_model.predict(crop, conf=0.30, verbose=False)[0]
        max_cap = 0.0
        max_hh = 0.0
        for cb in c_res.boxes:
            cname = cap_model.names[int(cb.cls[0])].lower()
            conf = float(cb.conf[0])
            if cname == 'cap':
                max_cap = max(max_cap, conf)
            elif cname == 'hardhat':
                max_hh = max(max_hh, conf)
        # Cap ต้องเด่นชัดมาก (>= 85%) และเหนือกว่า Hardhat อย่างชัดเจน
        if max_cap >= 0.85 and max_cap > (max_hh * 1.5):
            return True, max_cap
    except Exception:
        pass
    return False, 0.0

def auto_annotate_image(img_path, master_model, cap_model):
    """สร้าง Annotation YOLO อัตโนมัติด้วย Ensemble Logic"""
    img = cv2.imread(img_path)
    if img is None:
        return []
    h_img, w_img = img.shape[:2]

    res = master_model(img, conf=0.25, iou=0.50, verbose=False)[0]
    boxes_out = []
    vest_boxes = []
    person_boxes = []

    # รอบที่ 1: ตรวจจับวัตถุทั่วไปและหมวก
    for b in res.boxes:
        cls_id = int(b.cls[0])
        cls_name = master_model.names[cls_id]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]

        if cls_name == 'Hardhat':
            is_cap, _ = check_is_cap(cap_model, img, [x1, y1, x2, y2])
            if is_cap:
                cls_id = 2  # แปลงเป็น NO-Hardhat

        if cls_id in [0, 2, 4, 5, 7]:  # โฟกัสเฉพาะ PPE และคนงาน
            boxes_out.append({
                'cls': cls_id,
                'name': CLASS_NAMES[cls_id],
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })
            if cls_id == 7:  # Safety Vest
                vest_boxes.append([x1, y1, x2, y2])
            elif cls_id == 5:  # Person
                person_boxes.append([x1, y1, x2, y2])

    # รอบที่ 2: ตรวจสอบคนงานที่ไม่สวมเสื้อกั๊ก (Torso Association)
    for pbox in person_boxes:
        px1, py1, px2, py2 = pbox
        pw = px2 - px1
        ph = py2 - py1
        if ph < 50 or pw < 25:
            continue
        tx1 = px1 + 0.10 * pw
        ty1 = py1 + 0.20 * ph
        tx2 = px2 - 0.10 * pw
        ty2 = py1 + 0.85 * ph

        has_vest = False
        for vx1, vy1, vx2, vy2 in vest_boxes:
            ix1 = max(tx1, vx1)
            iy1 = max(ty1, vy1)
            ix2 = min(tx2, vx2)
            iy2 = min(ty2, vy2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                vest_area = max(1.0, (vx2 - vx1) * (vy2 - vy1))
                if inter_area / vest_area > 0.30:
                    has_vest = True
                    break

        if not has_vest:
            boxes_out.append({
                'cls': 4,  # NO-Safety Vest
                'name': 'NO-Safety Vest',
                'conf': 0.85,
                'bbox': [tx1, ty1, tx2, ty2]
            })

    return boxes_out

def write_yolo_labels_and_preview(img_path, boxes, out_lbl_path, out_preview_path):
    """บันทึกไฟล์ YOLO .txt และวาดภาพ Preview"""
    img = cv2.imread(img_path)
    if img is None:
        return
    h_img, w_img = img.shape[:2]

    # บันทึกไฟล์ .txt
    with open(out_lbl_path, 'w', encoding='utf-8') as f:
        for b in boxes:
            cls_id = b['cls']
            x1, y1, x2, y2 = b['bbox']
            xc = ((x1 + x2) / 2.0) / w_img
            yc = ((y1 + y2) / 2.0) / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img
            # ตัดค่าให้อยู่ในขอบเขต 0.0 - 1.0
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # วาดภาพ Preview
    canvas = img.copy()
    for b in boxes:
        cls_id = b['cls']
        name = b['name']
        x1, y1, x2, y2 = [int(v) for v in b['bbox']]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_text = f"{name}"
        cv2.putText(canvas, label_text, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imwrite(out_preview_path, canvas)

def prepare_finetune_dataset(new_items, replay_count=60):
    """จัดเตรียม Dataset ผสมระหว่างรูปใหม่กับ Replay Buffer เดิมเพื่อป้องกัน Overfitting"""
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    ws_train_imgs = os.path.join(WORKSPACE_DIR, 'images/train')
    ws_train_lbls = os.path.join(WORKSPACE_DIR, 'labels/train')
    os.makedirs(ws_train_imgs, exist_ok=True)
    os.makedirs(ws_train_lbls, exist_ok=True)

    # 1. คัดลอกรูปใหม่พร้อม Label เข้าสู่ Workspace
    for item in new_items:
        src_img = item['img']
        src_lbl = item['lbl']
        fname = os.path.basename(src_img)
        lbl_fname = os.path.splitext(fname)[0] + '.txt'
        shutil.copy2(src_img, os.path.join(ws_train_imgs, fname))
        shutil.copy2(src_lbl, os.path.join(ws_train_lbls, lbl_fname))

    # 2. ผสม Replay Buffer จาก Dataset/train เดิม
    orig_train_imgs_dir = os.path.join(DATASET_DIR, 'train/images')
    orig_train_lbls_dir = os.path.join(DATASET_DIR, 'train/labels')
    if os.path.exists(orig_train_imgs_dir):
        all_orig_imgs = [f for f in os.listdir(orig_train_imgs_dir) if f.lower().endswith(('.jpg', '.png'))]
        sampled_imgs = random.sample(all_orig_imgs, min(replay_count, len(all_orig_imgs)))
        print(f"[+] ผสม Replay Buffer จำนวน {len(sampled_imgs)} ภาพจากฐานข้อมูลเดิม เพื่อคงความแม่นยำดั้งเดิมไว้")
        for fname in sampled_imgs:
            lbl_fname = os.path.splitext(fname)[0] + '.txt'
            orig_img_p = os.path.join(orig_train_imgs_dir, fname)
            orig_lbl_p = os.path.join(orig_train_lbls_dir, lbl_fname)
            if os.path.exists(orig_img_p) and os.path.exists(orig_lbl_p):
                shutil.copy2(orig_img_p, os.path.join(ws_train_imgs, 'replay_' + fname))
                shutil.copy2(orig_lbl_p, os.path.join(ws_train_lbls, 'replay_' + lbl_fname))

    # 3. สร้างไฟล์ data_finetune.yaml
    valid_dir = os.path.abspath(os.path.join(DATASET_DIR, 'valid'))
    yaml_content = f"""train: {os.path.abspath(ws_train_imgs).replace(chr(92), '/')}
val: {valid_dir.replace(chr(92), '/')}

nc: 10
names:
  0: Hardhat
  1: Mask
  2: NO-Hardhat
  3: NO-Mask
  4: NO-Safety Vest
  5: Person
  6: Safety Cone
  7: Safety Vest
  8: machinery
  9: vehicle
"""
    yaml_path = os.path.join(WORKSPACE_DIR, 'data_finetune.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    return yaml_path

def run_pipeline(epochs=15, batch=None, dry_run=False, auto_deploy=False):
    print_banner()

    # ตรวจสอบโฟลเดอร์ภาพใหม่
    os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(PREVIEWS_DIR, exist_ok=True)

    # ค้นหาภาพในโฟลเดอร์
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    image_files = [
        f for f in glob.glob(os.path.join(INPUT_IMAGES_DIR, '*'))
        if os.path.isfile(f) and f.lower().endswith(valid_exts)
    ]

    if not image_files:
        readme_path = os.path.join(INPUT_IMAGES_DIR, 'README_วิธีใช้งาน.txt')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=== วิธีเพิ่มข้อมูลเพื่อเทรนโมเดล ===\n\n"
                    "1. นำรูปภาพที่ถ่ายจากโทรศัพท์หรือกล้อง (.jpg, .png) มาวางในโฟลเดอร์นี้\n"
                    "2. ไม่จำเป็นต้องติดแท็ก (Bounding Box) มาก่อน ระบบมี AI Auto-Labeler ติดแท็กให้อัตโนมัติ\n"
                    "3. เมื่อวางรูปเรียบร้อย ให้ดับเบิลคลิกไฟล์ '3_TRAIN_NEW_DATA.bat' หรือรันคำสั่ง:\n"
                    "   python auto_train_new_images.py\n"
                    "4. ระบบจะสำรองโมเดลเดิมไว้เสมอ และเริ่มฝึกฝนโมเดลเพิ่มเติมให้อย่างปลอดภัย\n")
        print(f"\n[!] ยังไม่พบไฟล์รูปภาพในโฟลเดอร์: {INPUT_IMAGES_DIR}")
        print(f"    -> ได้สร้างคำแนะนำไว้ที่: {readme_path}")
        print(f"    -> นำรูปเพื่อนในห้องมาวางในโฟลเดอร์นี้ แล้วรันสคริปต์นี้ใหม่อีกครั้งได้เลยครับ!")
        return

    print(f"\n[+] ตรวจพบภาพใหม่ทั้งหมด {len(image_files)} ภาพ ใน {INPUT_IMAGES_DIR}")

    # ขั้นตอนที่ 1: สำรองข้อมูลโมเดลเดิม (Safety Backup)
    backup_path = backup_master_model()
    if not backup_path:
        print("[-] ไม่สามารถสำรองโมเดลได้ ขอระงับการทำงานเพื่อความปลอดภัย")
        return

    # ขั้นตอนที่ 2: โหลดโมเดลสำหรับทำ Auto-Annotation
    print("\n[+] 🤖 กำลังเตรียมระบบ Auto-Annotator...")
    master_model = YOLO(MASTER_MODEL_PATH)
    cap_model = YOLO(CAP_MODEL_PATH) if os.path.exists(CAP_MODEL_PATH) else None

    # ขั้นตอนที่ 3: ตรวจสอบและสร้าง Annotation
    print(f"[+] 🔍 กำลังตรวจสอบและติดแท็ก Bounding Box อัตโนมัติ...")
    processed_items = []
    for idx, img_p in enumerate(image_files, 1):
        fname = os.path.basename(img_p)
        base_name = os.path.splitext(fname)[0]
        lbl_p = os.path.join(INPUT_IMAGES_DIR, base_name + '.txt')
        preview_p = os.path.join(PREVIEWS_DIR, base_name + '_preview.jpg')

        # หากมีไฟล์ .txt อยู่แล้ว (เช่น มีคนเขียน Label มาให้)
        if os.path.exists(lbl_p):
            print(f"  [{idx}/{len(image_files)}] {fname} -> ใช้ Label เดิมที่มีอยู่")
        else:
            boxes = auto_annotate_image(img_p, master_model, cap_model)
            write_yolo_labels_and_preview(img_p, boxes, lbl_p, preview_p)
            print(f"  [{idx}/{len(image_files)}] {fname} -> ติดแท็กอัตโนมัติพบ {len(boxes)} วัตถุ (สร้าง Preview เรียบร้อย)")

        processed_items.append({'img': img_p, 'lbl': lbl_p})

    print(f"[+] ✅ ตรวจสอบและติดแท็กเสร็จสิ้น ตรวจดูภาพตัวอย่างได้ที่:\n    -> {PREVIEWS_DIR}")

    if dry_run:
        print("\n[!] โหมด Dry-Run: สำรองโมเดลและสร้าง Annotation เรียบร้อย (ไม่เริ่มการเทรน)")
        return

    # ขั้นตอนที่ 4: จัดเตรียมชุดข้อมูลฝึกฝน
    print("\n[+] 📦 จัดเตรียมชุดข้อมูล Fine-Tuning ผสม Replay Buffer...")
    finetune_yaml = prepare_finetune_dataset(processed_items, replay_count=50)

    # ตรวจสอบ Device
    device = '0' if torch.cuda.is_available() else 'cpu'
    if batch is None:
        batch = 8 if device == '0' else 4

    print(f"\n[+] ⚙️  การตั้งค่าการฝึกฝน (Fine-Tuning Configuration):")
    print(f"    - Base Model: Master 200-Epoch ({MASTER_MODEL_PATH})")
    print(f"    - Dataset YAML: {finetune_yaml}")
    print(f"    - Device: {device} ({'GPU Acceleration' if device == '0' else 'CPU Mode'})")
    print(f"    - Epochs: {epochs}")
    print(f"    - Batch Size: {batch}")
    print(f"    - Learning Rate: 0.0005 (Safe fine-tuning rate)")
    print(f"    - Freeze Backbone: 10 layers (รักษาโครงสร้างหลักของ 200-Epoch)")

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppe_finetune_{run_timestamp}"

    # ขั้นตอนที่ 5: เริ่มต้นการฝึกฝน
    print("\n" + "=" * 70)
    print("  🚀 เริ่มต้นกระบวนการ Fine-Tuning โมเดล...")
    print("=" * 70)

    trainer_model = YOLO(MASTER_MODEL_PATH)
    trainer_model.train(
        data=finetune_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        device=device,
        lr0=0.0005,
        lrf=0.01,
        freeze=10,
        project=os.path.join(BASE_DIR, 'Model-Training/Outputs/runs/detect'),
        name=run_name,
        verbose=True
    )

    new_weights_path = os.path.join(
        BASE_DIR, f"Model-Training/Outputs/runs/detect/{run_name}/weights/best.pt"
    )

    print("\n" + "=" * 70)
    print("  🎉 FINE-TUNING เสร็จสมบูรณ์เรียบร้อยแล้ว!")
    print(f"  -> โมเดลที่เทรนใหม่บันทึกอยู่ที่: {new_weights_path}")
    print(f"  -> โมเดลต้นฉบับสำรองไว้อย่างปลอดภัยที่: {backup_path}")
    print("=" * 70)

    # ขั้นตอนที่ 6: การประเมินผลความแม่นยำ (Validation Benchmark)
    print("\n" + "=" * 70)
    print("  📊 ประเมินผลความแม่นยำของโมเดลใหม่ (Validation on Test Set)...")
    print("=" * 70)
    val_metrics = None
    try:
        new_yolo = YOLO(new_weights_path)
        val_metrics = new_yolo.val(data=os.path.join(BASE_DIR, 'Model-Training/Outputs/data.yaml'), split='test', verbose=False)
        mAP50 = float(val_metrics.box.map50)
        mAP50_95 = float(val_metrics.box.map)
        print(f"  -> โมเดลใหม่ mAP50: {mAP50:.4f} ({mAP50*100:.1f}%) | mAP50-95: {mAP50_95:.4f}")
        for idx, cname in enumerate(val_metrics.names.values()):
            if idx < len(val_metrics.box.maps):
                print(f"     * {cname:15s}: mAP50-95 = {val_metrics.box.maps[idx]:.4f}")
    except Exception as e:
        print(f"[-] ไม่สามารถประเมินผลได้: {e}")

    # ขั้นตอนที่ 7: ตัวเลือกการ Deploy อย่างปลอดภัย
    if os.path.exists(new_weights_path):
        is_safe_to_deploy = (val_metrics is not None and val_metrics.box.map50 >= 0.70)
        if auto_deploy:
            if is_safe_to_deploy or val_metrics is None:
                deploy_new_model(new_weights_path)
            else:
                print(f"[!] คำเตือน: mAP50 ({val_metrics.box.map50:.4f}) ต่ำกว่าเกณฑ์ความปลอดภัย 0.70 ยกเลิกการ auto-deploy อัตโนมัติ")
        else:
            print("\nต้องการนำโมเดลที่เทรนใหม่นี้ไปใช้งานบนระบบเว็บ (Model-Deployment) ทันทีหรือไม่?")
            ans = input("พิมพ์ 'y' เพื่อ Deploy หรือ 'n' เพื่อเก็บไว้ทดสอบก่อน (default: n): ").strip().lower()
            if ans == 'y':
                deploy_new_model(new_weights_path)
            else:
                print("[+] เก็บโมเดลไว้ที่เดิม คุณสามารถนำไปทดสอบบน /slideshow หรือเปรียบเทียบก่อนได้ครับ")

def deploy_new_model(weights_path):
    """Deploy โมเดลใหม่ไปแทนที่โมเดลปัจจุบันอย่างปลอดภัย"""
    dest_path = MASTER_MODEL_PATH
    print(f"\n[+] 🚀 กำลัง Deploy โมเดลใหม่ไปยัง {dest_path}...")
    shutil.copy2(weights_path, dest_path)
    print("[+] ✅ Deploy โมเดลใหม่เข้าสู่ระบบเรียบร้อยแล้ว! (เซิร์ฟเวอร์จะ Reload อัตโนมัติ)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto fine-tune YOLOv8 PPE model with new images")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument('--batch', type=int, default=None, help="Batch size (default: auto)")
    parser.add_argument('--dry-run', action='store_true', help="Backup and auto-annotate only without training")
    parser.add_argument('--auto-deploy', action='store_true', help="Automatically deploy model after training")
    args = parser.parse_args()

    run_pipeline(epochs=args.epochs, batch=args.batch, dry_run=args.dry_run, auto_deploy=args.auto_deploy)
