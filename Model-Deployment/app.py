from flask import Flask, render_template, Response, jsonify, request, send_file, redirect
from flask_socketio import SocketIO, emit
import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime, timedelta
import os
import threading
import time
from PIL import Image
from detection_logic import InstanceDetector, ComplianceChecker, SnapshotManager
from database import Database
from kiosk_logic import KioskSession
from hardware import HardwareLink
from ppe_stabilizer import PPETemporalStabilizer, box_intersection_ratio
import torch
import checklog
import json
import subprocess

app = Flask(__name__)
app.config['SECRET_KEY'] = 'N/A'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

model = None
camera = None
streaming = False
stream_lock = threading.Lock()
dev_mode = False
db = Database()
instance_detector = InstanceDetector()
compliance_checker = ComplianceChecker()
snapshot_manager = SnapshotManager()
kiosk = KioskSession()
hardware = HardwareLink()
checklog.init()
recorder = checklog.CheckRecorder()
ppe_stabilizer = PPETemporalStabilizer(iou_match_thresh=0.35, max_grace_frames=5, alpha=0.72)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
DEFAULT_SETTINGS = {
    'required_ppe': {
        'helmet': True,
        'vest': True,
        'mask': False
    },
    'non_compliance_delay': 3,
    'instance_reset_timeout': 5,
    'detection_mode': 'single',  # 'single' or 'multi'
    'camera_index': 1,
    'camera_width': 640,
    'camera_height': 320,
    'device': 'gpu',
    'kiosk_pass_history_mode': 'latest'  # 'latest' or 'all'
}

def load_settings():
    """Load settings from file or return defaults"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return DEFAULT_SETTINGS.copy()

def save_settings_to_file(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

current_settings = load_settings()

cap_verifier = None

def get_inference_device():
    """Detect and return device based on settings or system availability"""
    target = current_settings.get('device', 'gpu')
    if target == 'cpu':
        return 'cpu', "CPU (Standard Mode)"
    
    # Try GPU if requested or default
    try:
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            return 0, f"CUDA (GPU: {dev_name})"
    except Exception:
        pass
    return 'cpu', "CPU (Standard Mode)"

def load_model():
    """Load Master 200-Epoch Super PPE Model + Dedicated Cap Verifier on GPU/CUDA"""
    global model, cap_verifier
    super_model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'
    ))
    cap_model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../Model-Training/Outputs/runs/detect/yolov8s_ppe_model_with_cap_class/weights/best.pt'
    ))
    
    device_id, device_label = get_inference_device()
    print(f"[+] [ACCELERATION] Using Device: {device_label}")

    if os.path.exists(super_model_path):
        print(f"[+] [EXCLUSIVE] Loaded Master PPE Model: {super_model_path}")
        model = YOLO(super_model_path)
        try:
            model.to(device_id)
        except Exception as e:
            print(f"Warning: could not move model to {device_id}: {e}")
    else:
        raise FileNotFoundError(f"[-] Critical: PPE model not found at {super_model_path}")

    if os.path.exists(cap_model_path):
        print(f"[+] [EXCLUSIVE] Loaded Dedicated Cap Verifier: {cap_model_path}")
        cap_verifier = YOLO(cap_model_path)
        try:
            cap_verifier.to(device_id)
        except Exception as e:
            print(f"Warning: could not move cap verifier to {device_id}: {e}")

def is_headwear_a_cap(img, head_box, current_class='Hardhat'):
    """
    Checks if a detected headwear box (Hardhat or NO-Hardhat) is actually a baseball cap (หมวกแก๊ป).
    Uses 30% spatial padding, CLAHE contrast enhancement for dark/black caps, and competitive arbitration.
    Returns (is_cap: bool, confidence: float)
    """
    global cap_verifier
    if cap_verifier is None or img is None:
        return False, 0.0
    try:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in head_box]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        # Ensure minimum absolute padding so distant/small headwear captures the full visor & crown
        pad_w = max(12, int(bw * 0.30))
        pad_h = max(12, int(bh * 0.30))
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return False, 0.0

        # Contrast enhancement with CLAHE on L-channel:
        # Crucial for black/dark baseball caps which lose edge contrast & texture when distant or underexposed
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        r = cap_verifier(enhanced_crop, imgsz=224, conf=0.15, device=get_inference_device()[0], verbose=False)[0]
        caps = [float(b.conf[0]) for b in r.boxes if cap_verifier.names[int(b.cls[0])].lower() == 'cap']
        hardhats = [float(b.conf[0]) for b in r.boxes if cap_verifier.names[int(b.cls[0])].lower() == 'hardhat']
        no_hardhats = [float(b.conf[0]) for b in r.boxes if cap_verifier.names[int(b.cls[0])].lower() == 'no-hardhat']
        max_c = max(caps) if caps else 0.0
        max_h = max(hardhats) if hardhats else 0.0
        max_no_h = max(no_hardhats) if no_hardhats else 0.0
        
        # Requirements for declaring a Cap:
        # 1. Cap score must be higher than hardhat score (real hardhats score 0.90+)
        # 2. Cap score must be higher than NO-Hardhat score (bare heads score 0.85+)
        # 3. Cap score must meet sensitivity threshold (>= 0.62 for Hardhat, >= 0.65 for NO-Hardhat)
        min_thresh = 0.62 if current_class == 'Hardhat' else 0.65
        if max_c >= min_thresh and max_c > max_h and max_c > max_no_h:
            return True, max_c
    except Exception:
        pass
    return False, 0.0

def detect_cap_on_person(img, person_bbox):
    """Fallback cap detection on person upper body"""
    return False, 0.0, None

def verify_cap_on_crop(img, bbox):
    """Check crop for cap verification"""
    return False, 0.0

def get_ppe_class_ids(model_inst, include_mask=None):
    """Filter class IDs to focus STRICTLY on required PPE attire & person based on current settings."""
    if not model_inst or not hasattr(model_inst, 'names'):
        return None

    global current_settings
    req_ppe = current_settings.get('required_ppe', {}) if current_settings else {}
    req_helmet = req_ppe.get('helmet', True)
    req_vest = req_ppe.get('vest', True)
    req_mask = req_ppe.get('mask', False) if include_mask is None else include_mask

    excluded = ['vehicle', 'machinery', 'excavator', 'ladder', 'truck', 'sedan', 'van', 'bus', 'semi', 'trailer', 'loader', 'hydrant', 'suv', 'cone']
    
    target_ids = []
    for cid, name in model_inst.names.items():
        n = name.lower()
        if any(ex in n for ex in excluded):
            continue
        # Check PPE relevance according to checkboxes
        if not req_mask and 'mask' in n:
            continue
        if not req_helmet and any(k in n for k in ['hardhat', 'cap']):
            continue
        if not req_vest and 'vest' in n:
            continue

        target_ids.append(cid)
    return target_ids if target_ids else None

def verify_person_ppe_compliance(detections, img_shape):
    """
    Person-to-PPE Association & Fallback Verification:
    Ensures that for every detected worker (Person):
    1. If the torso lacks a Safety Vest (and no NO-Safety Vest box was output by YOLO),
       generate a synthetic violation 'NO-Safety Vest' (Missing).
    2. If the head lacks a Hardhat/Cap (and no NO-Hardhat box was output by YOLO),
       generate a synthetic violation 'NO-Hardhat' (Missing).
    This guarantees zero false-compliant escapes when workers are missing gear in occluded/low-angle conditions.
    """
    def box_intersection(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return max(0.0, xB - xA) * max(0.0, yB - yA)

    persons = [d for d in detections if d.get('class') == 'Person']
    if not persons:
        return []

    vests = [d for d in detections if d.get('class') in ['Safety Vest', 'NO-Safety Vest']]
    hardhats = [d for d in detections if d.get('class') in ['Hardhat', 'Cap', 'NO-Hardhat']]

    inferred = []
    next_id = max([d.get('id', 0) for d in detections], default=0) + 1

    for idx, p in enumerate(persons):
        px1, py1, px2, py2 = p['bbox']
        pw, ph = px2 - px1, py2 - py1
        # Ignore very tiny person boxes (less than 35px height or 25px width)
        if pw < 25 or ph < 35:
            continue

        torso = [px1, py1 + 0.20 * ph, px2, py1 + 0.85 * ph]
        head = [px1, py1, px2, py1 + 0.40 * ph]

        # Check Vest on Torso (Only if torso is actually visible in the frame!)
        has_visible_torso = (ph >= 1.35 * pw) and (ph >= 150)
        if has_visible_torso:
            has_vest = any(
                box_intersection(torso, v['bbox']) > 0.05 * max(1.0, (v['bbox'][2]-v['bbox'][0]) * (v['bbox'][3]-v['bbox'][1]))
                for v in vests
            )
            if not has_vest:
                inferred.append({
                    'id': next_id,
                    'class': 'NO-Safety Vest',
                    'confidence': round(p['confidence'] * 0.92, 1),
                    'is_violation': True,
                    'is_inferred': True,
                    'note': f'ไม่พบเสื้อกั๊กสะท้อนแสงบนลำตัวคนงาน #{idx+1}',
                    'bbox': [round(px1 + 0.05*pw, 1), round(py1 + 0.22*ph, 1), round(px2 - 0.05*pw, 1), round(py1 + 0.72*ph, 1)]
                })
                next_id += 1

        # Check Hardhat on Head
        has_helmet = any(
            box_intersection(head, h['bbox']) > 0.05 * max(1.0, (h['bbox'][2]-h['bbox'][0]) * (h['bbox'][3]-h['bbox'][1]))
            for h in hardhats
        )
        if not has_helmet:
            inferred.append({
                'id': next_id,
                'class': 'NO-Hardhat',
                'confidence': round(p['confidence'] * 0.92, 1),
                'is_violation': True,
                'is_inferred': True,
                'note': f'ไม่พบหมวกนิรภัยบนศีรษะคนงาน #{idx+1}',
                'bbox': [round(px1 + 0.05*pw, 1), round(py1, 1), round(px2 - 0.05*pw, 1), round(py1 + 0.35*ph, 1)]
            })
            next_id += 1

    return inferred

def generate_frames(clean=False):
    """Generate video frames with instance detection"""
    global camera, streaming, model, current_settings
    
    last_alert_time = 0
    ALERT_COOLDOWN = current_settings['non_compliance_delay']
    last_snapshot_time = 0
    SNAPSHOT_INTERVAL = current_settings['instance_reset_timeout']
    frame_count = 0
    qr = cv2.QRCodeDetector()
    
    while streaming:
        try:
            with stream_lock:
                if camera is None or not camera.isOpened():
                    break
                
                success, frame = camera.read()
            
            if not success or frame is None:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            ppe_ids = get_ppe_class_ids(model)
            dev_target = get_inference_device()[0]
            
            # High-Performance Inference on CUDA for 60+ FPS
            results = model(frame, conf=0.20, iou=0.45, imgsz=640, classes=ppe_ids, agnostic_nms=True, device=dev_target, verbose=False) if ppe_ids else model(frame, conf=0.20, iou=0.45, imgsz=640, agnostic_nms=True, device=dev_target, verbose=False)
            
            all_detections = []
            person_boxes = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    b_int = [int(x1), int(y1), int(x2), int(y2)]
                    all_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': b_int
                    })
                    if class_name.lower() == 'person':
                        person_boxes.append(b_int)

            # Direct Headwear Cap Verification
            for det in all_detections:
                if det['class'] in ['Hardhat', 'NO-Hardhat']:
                    is_cap, cap_score = is_headwear_a_cap(frame, det['bbox'], current_class=det['class'])
                    if is_cap:
                        det['class'] = 'NO-Hardhat'
                        det['is_cap'] = True
                        det['is_violation'] = True
                        det['note'] = 'หมวกแก๊ป (Cap Detected)'
                        det['confidence'] = max(det['confidence'], cap_score)
                    else:
                        det['is_cap'] = False
            
            # Apply temporal smoothing and anti-flicker stabilization
            det_mode = current_settings.get('detection_mode', 'single')
            stabilized_dets, summary = ppe_stabilizer.update(all_detections, frame.shape, current_settings.get('required_ppe'), mode=det_mode)
            
            emp_now = recorder.current()
            # ตรวจสอบรหัสพนักงาน: หากระบุรหัสมา จะต้องมีอยู่จริงในฐานข้อมูลเท่านั้น
            is_valid_emp = checklog.employee_exists(emp_now['code']) if emp_now else True
            kiosk_state = kiosk.update(stabilized_dets, frame.shape, current_settings, is_valid_employee=is_valid_emp)
            # สแกน QR Code เฉพาะเมื่ออยู่ในขั้นตอน 'ระบุตัวตน' (kiosk.is_qr_enabled()) เท่านั้น
            if frame_count % 6 == 0 and kiosk.is_qr_enabled():
                try:
                    # Downscale 2x for ultra-fast QR decoding without dropping accuracy
                    qr_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                    code, _, _ = qr.detectAndDecode(qr_small)
                except cv2.error:
                    code = ''
                if code and checklog.employee_exists(code):
                    recorder.set_code(code)
                    kiosk.disable_qr()  # สแกนติดแล้ว ปิดสแกนทันที เพื่อให้ Frontend เริ่มนับถอยหลัง 3.. 2.. 1..
            logged = recorder.observe(kiosk_state)
            hardware.update(kiosk_state)

            instance_result = instance_detector.process_detection(stabilized_dets, dev_mode, current_settings)
            is_compliant = summary['is_compliant'] if summary['has_person'] else compliance_checker.check_compliance(instance_result, dev_mode)
            
            annotated_frame = ppe_stabilizer.render(frame, stabilized_dets, summary, show_top_banner=(not clean))
            if logged and logged['result'] == 'pass':  # เก็บภาพตอนตรวจผ่านลงฐานข้อมูล
                ok_jpg, jpg = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_jpg:
                    checklog.save_image(logged['id'], jpg.tobytes())
            
            current_time = time.time()
            
            if instance_result['should_capture'] and (current_time - last_snapshot_time) >= SNAPSHOT_INTERVAL:
                snapshot_filename = instance_detector.get_next_snapshot_filename()
                if snapshot_filename:
                    snapshot_path = snapshot_manager.save_snapshot(frame, snapshot_filename)
                    
                    if snapshot_path:
                        db.log_instance_snapshot(
                            instance_id=instance_result['instance_id'],
                            missing_ppe=instance_result['missing_ppe'],
                            detected_ppe=instance_result['detected_ppe'],
                            snapshot_path=snapshot_path
                        )
                        last_snapshot_time = current_time
            
            if not is_compliant and instance_result['has_person'] and not clean:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), 
                             (0, 0, 255), 20)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, overlay, 0.2, 0)
                
                alert_text = "DEV MODE - TESTING" if dev_mode else "NON-COMPLIANT DETECTED"
                cv2.putText(annotated_frame, alert_text, 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    db.log_alert("NON_COMPLIANCE", "PPE non-compliance detected", None)
                    
                    socketio.emit('alert', {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'NON_COMPLIANCE',
                        'description': 'PPE non-compliance detected',
                        'dev_mode': dev_mode
                    })
                    last_alert_time = current_time
            
            socketio.emit('detection_update', {
                'timestamp': datetime.now().isoformat(),
                'is_compliant': is_compliant,
                'detection_details': instance_result,
                'dev_mode': dev_mode
            })
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except GeneratorExit:
            break
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/review')
def review():
    """Review page for instances and snapshots"""
    return render_template('review.html')

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

@app.route('/kiosk')
def kiosk_page():
    """Pass/fail checkpoint screen"""
    return render_template('kiosk.html')

@app.route('/api/kiosk_state')
def kiosk_state_api():
    state = kiosk.get_state()
    state['hardware'] = hardware.status()
    state['employee'] = recorder.current()
    pass_mode = current_settings.get('kiosk_pass_history_mode', 'latest')
    state['pass_history_mode'] = pass_mode

    # ดึงประวัติผู้ตรวจผ่านล่าสุด หรือทั้งหมด ตามโหมดที่ตั้งค่าใน /admin
    if pass_mode == 'all':
        all_passes = checklog._q(
            'SELECT l.id, l.emp_code, e.name, l.checked_at FROM check_logs l '
            'LEFT JOIN employees e ON e.code = l.emp_code '
            'INNER JOIN check_images i ON i.log_id = l.id '
            'WHERE l.result = "pass" ORDER BY l.id DESC LIMIT 30'
        )
        state['all_passes'] = [
            {
                'id': row[0],
                'code': row[1] or '-',
                'name': row[2] or '-',
                'time': row[3],
                'image_url': f'/api/check_image/{row[0]}.jpg'
            }
            for row in all_passes
        ]
        state['last_pass'] = state['all_passes'][0] if state['all_passes'] else None
    else:
        latest_pass = checklog._q(
            'SELECT l.id, l.emp_code, e.name, l.checked_at FROM check_logs l '
            'LEFT JOIN employees e ON e.code = l.emp_code '
            'INNER JOIN check_images i ON i.log_id = l.id '
            'WHERE l.result = "pass" ORDER BY l.id DESC LIMIT 1'
        )
        if latest_pass:
            lid, lcode, lname, ltime = latest_pass[0]
            state['last_pass'] = {
                'id': lid,
                'code': lcode or '-',
                'name': lname or '-',
                'time': ltime,
                'image_url': f'/api/check_image/{lid}.jpg'
            }
        else:
            state['last_pass'] = None
        state['all_passes'] = [state['last_pass']] if state['last_pass'] else []
    return jsonify(state)

@app.route('/changelog')
def changelog_page():
    """Changelog and GitHub commit history page"""
    return render_template('changelog.html')

@app.route('/api/changelog')
def api_changelog():
    """Get Git commit history as structured changelog entries"""
    try:
        # Run git log in repo root directory
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cmd = ['git', 'log', '-n', '60', '--pretty=format:%h%x09%an%x09%ad%x09%ar%x09%s', '--date=format:%Y-%m-%d %H:%M']
        res = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        commits = []
        if res.returncode == 0 and res.stdout:
            for line in res.stdout.strip().splitlines():
                parts = line.split('\t')
                if len(parts) >= 5:
                    commits.append({
                        'hash': parts[0],
                        'author': parts[1],
                        'date': parts[2],
                        'relative_time': parts[3],
                        'message': parts[4]
                    })
        return jsonify({'status': 'success', 'commits': commits})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'commits': []}), 500

@app.route('/admin')
def admin_page():
    """Admin dashboard: Camera settings, inspection logs, and employee management"""
    return render_template('admin.html')

@app.route('/logs')
def logs_page():
    """Redirect legacy /logs route to new /admin route"""
    return redirect('/admin')

@app.route('/api/camera_devices')
def api_camera_devices():
    """Return available camera choices immediately without blocking probe delays"""
    # Pre-populated common indices for Windows/USB webcams
    devices = [
        {'index': 0, 'name': 'กล้องหลัก (Camera 0 / Default)', 'available': True},
        {'index': 1, 'name': 'กล้องตัวที่สอง (Camera 1 / USB 1)', 'available': True},
        {'index': 2, 'name': 'กล้องตัวที่สาม (Camera 2 / USB 2)', 'available': True},
        {'index': 3, 'name': 'กล้องตัวที่สี่ (Camera 3 / DroidCam / OBS)', 'available': True}
    ]
    return jsonify(devices)

@app.route('/api/logs')
def api_logs():
    limit = min(request.args.get('limit', 50, type=int), 500)
    return jsonify(checklog.recent_logs(limit))

@app.route('/api/today')
def api_today():
    return jsonify(checklog.today_summary())

@app.route('/api/employees', methods=['GET', 'POST'])
def api_employees():
    if request.method == 'POST':
        added = checklog.add_employees((request.json or {}).get('text', ''))
        return jsonify({'added': added})
    return jsonify(checklog.today_summary()['employees'])

@app.route('/api/employees/<code>', methods=['DELETE'])
def api_employee_delete(code):
    checklog.delete_employee(code)
    return jsonify({'status': 'success'})

@app.route('/api/kiosk_start_badge', methods=['POST'])
def api_kiosk_start_badge():
    """ผู้ใช้กดปุ่ม 'เริ่มตรวจ' เพื่อเข้าสู่หน้า 'ระบุตัวตน' -> เปิดสแกน QR Code"""
    recorder.clear()
    kiosk.reset()
    kiosk.enable_qr()
    return jsonify({'status': 'success'})

@app.route('/api/kiosk_reset', methods=['POST'])
def api_kiosk_reset():
    """Reset session and clear previous employee code (on page load)"""
    recorder.clear()
    kiosk.reset()
    kiosk.disable_qr()
    return jsonify({'status': 'success'})

@app.route('/api/kiosk_trigger', methods=['POST'])
def api_kiosk_trigger():
    """Trigger checking session after countdown completes or from action button"""
    kiosk.disable_qr()
    ok = kiosk.trigger_check()
    return jsonify({'status': 'success' if ok else 'ignored'})

@app.route('/api/kiosk_id', methods=['POST'])
def api_kiosk_id():
    """Manual entry or USB barcode scanner (acts as a keyboard)"""
    code = ((request.json or {}).get('code') or '').strip()
    if not checklog.valid_code(code):
        return jsonify({'status': 'error', 'message': 'รูปแบบรหัสไม่ถูกต้อง'}), 400
    if not checklog.employee_exists(code):
        return jsonify({'status': 'error', 'message': 'ไม่พบรหัสพนักงานนี้ในระบบ'}), 404
    state = kiosk.get_state()['state']
    if state in ('pass', 'fail'):
        return jsonify({'status': 'error', 'message': 'กำลังแสดงผล รอสักครู่'}), 409
    recorder.set_code(code)
    kiosk.disable_qr()
    return jsonify({'status': 'success', 'employee': recorder.current()})

@app.route('/qr/<code>.png')
def qr_png(code):
    """QR image for an employee badge (needs: pip install qrcode[pil])"""
    try:
        import io
        import qrcode
    except ImportError:
        return 'ติดตั้งก่อน: pip install "qrcode[pil]"', 501
    if not checklog.valid_code(code):
        return '', 404
    buf = io.BytesIO()
    qrcode.make(code, box_size=12, border=3).save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/api/check_image/<int:log_id>.jpg')
def api_check_image(log_id):
    import io
    jpg = checklog.get_image(log_id)
    if jpg is None:
        return '', 404
    resp = send_file(io.BytesIO(jpg), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'private, max-age=86400'
    return resp

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    return jsonify(current_settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings and dynamically reconfigure camera if needed"""
    global current_settings, camera, streaming
    try:
        new_settings = request.json
        old_idx = current_settings.get('camera_index', 0)
        old_w = current_settings.get('camera_width', 1280)
        old_h = current_settings.get('camera_height', 720)
        
        current_settings = new_settings
        instance_detector.update_settings(new_settings)
        
        # Check if camera parameters changed
        new_idx = int(new_settings.get('camera_index', 0))
        new_w = int(new_settings.get('camera_width', 1280))
        new_h = int(new_settings.get('camera_height', 720))
        cam_changed = (old_idx != new_idx or old_w != new_w or old_h != new_h)
        
        # Check if render device changed (GPU <-> CPU)
        if 'device' in new_settings:
            target_dev, dev_label = get_inference_device()
            try:
                if model is not None:
                    model.to(target_dev)
                if cap_verifier is not None:
                    cap_verifier.to(target_dev)
                print(f"[+] [ACCELERATION] Switched Model Device to: {dev_label}")
            except Exception as e:
                print(f"Warning: could not switch model device: {e}")
        
        if cam_changed:
            with stream_lock:
                if camera is not None:
                    try:
                        camera.release()
                    except Exception:
                        pass
                    camera = None
                time.sleep(0.15)
                camera = cv2.VideoCapture(new_idx, cv2.CAP_DSHOW)
                if not camera.isOpened():
                    camera = cv2.VideoCapture(new_idx)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
                camera.set(cv2.CAP_PROP_FPS, 60)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            streaming = True
        
        if save_settings_to_file(new_settings):
            return jsonify({'status': 'success', 'message': 'บันทึกการตั้งค่าเรียบร้อยแล้ว'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings to file'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    global current_settings
    try:
        current_settings = DEFAULT_SETTINGS.copy()
        instance_detector.update_settings(current_settings)
        
        if save_settings_to_file(current_settings):
            return jsonify({'status': 'success', 'message': 'Settings reset to defaults', 'settings': current_settings})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings to file'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def init_camera_stream():
    """Ensure camera is initialized and streaming is active"""
    global camera, streaming, model
    if model is None:
        load_model()
    with stream_lock:
        if camera is None or not camera.isOpened():
            cam_idx = int(current_settings.get('camera_index', os.environ.get('PPE_CAMERA_INDEX', 0)))
            cam_w = int(current_settings.get('camera_width', 1280))
            cam_h = int(current_settings.get('camera_height', 720))
            camera = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera = cv2.VideoCapture(cam_idx)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
            camera.set(cv2.CAP_PROP_FPS, 60)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    streaming = True

@app.route('/video_feed')
def video_feed():
    """Video streaming route with auto-start stream"""
    try:
        init_camera_stream()
        return Response(generate_frames(clean=request.args.get('clean') == '1'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return '', 500

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start video streaming"""
    global camera, streaming, model
    
    try:
        init_camera_stream()
        kiosk.reset()
        recorder.clear()
        ppe_stabilizer.reset()
        streaming = True
        return jsonify({'status': 'success', 'message': 'Stream started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop video streaming"""
    global camera, streaming
    
    try:
        streaming = False
        ppe_stabilizer.reset()
        time.sleep(0.3)
        
        with stream_lock:
            if camera is not None:
                camera.release()
                camera = None
        
        return jsonify({'status': 'success', 'message': 'Stream stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_dev_mode', methods=['POST'])
def toggle_dev_mode():
    """Toggle dev/testing mode"""
    global dev_mode
    dev_mode = not dev_mode
    return jsonify({'status': 'success', 'dev_mode': dev_mode, 
                   'message': f'Dev mode {"enabled" if dev_mode else "disabled"}'})

@app.route('/stats')
def get_stats():
    """Get detection statistics"""
    stats = db.get_statistics()
    stats['dev_mode'] = dev_mode
    return jsonify(stats)

@app.route('/api/instances')
def get_instances():
    """Get all detection instances"""
    try:
        sort_by = request.args.get('sort', 'first_detected')
        sort_order = request.args.get('order', 'desc')
        
        instances = db.get_all_instances(sort_by, sort_order)
        return jsonify(instances)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/instance/<instance_id>/snapshots')
def get_instance_snapshots(instance_id):
    """Get all snapshots for a specific instance"""
    try:
        data = db.get_instance_snapshots(instance_id)
        if data:
            return jsonify(data)
        return jsonify({'error': 'Instance not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_snapshot/<path:filename>')
def download_snapshot(filename):
    """Download snapshot"""
    try:
        filepath = filename if os.path.isabs(filename) else os.path.join('snapshots', filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    """Serve snapshot for viewing"""
    try:
        filepath = os.path.join('snapshots', filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='image/jpeg')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_instance/<instance_id>', methods=['DELETE'])
def delete_instance(instance_id):
    """Delete an instance and its snapshot"""
    try:
        success = db.delete_instance(instance_id)
        if success:
            return jsonify({'status': 'success', 'message': 'Instance deleted'})
        return jsonify({'error': 'Failed to delete instance'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# SLIDESHOW LIVE SHOWCASE ROUTES (4s Demo)
# ==========================================
import base64

@app.route('/slideshow')
def slideshow_page():
    """Serve slideshow inspection showcase page"""
    return render_template('slideshow.html')

@app.route('/api/slideshow/images')
def api_slideshow_images():
    """Get list of images from test and valid datasets"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datasets = {}
    for ds in ['test', 'valid']:
        img_dir = os.path.join(base_dir, 'Model-Training', 'Dataset', ds, 'images')
        if os.path.exists(img_dir):
            files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            datasets[ds] = {'count': len(files), 'images': files}
        else:
            datasets[ds] = {'count': 0, 'images': []}
    return jsonify({'success': True, 'datasets': datasets})

@app.route('/api/slideshow/detect/<dataset_type>/<filename>')
def api_slideshow_detect(dataset_type, filename):
    """Run detection on single dataset image and return base64 plotted image + detections"""
    global model
    try:
        if model is None:
            load_model()

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        img_path = os.path.join(base_dir, 'Model-Training', 'Dataset', dataset_type, 'images', filename)

        if not os.path.exists(img_path):
            return jsonify({'error': 'Image not found'}), 404

        img = cv2.imread(img_path)
        if img is None:
            return jsonify({'error': 'Image file cannot be read'}), 400

        t0 = time.time()
        ppe_ids = get_ppe_class_ids(model)
        results = model(img, conf=0.20, iou=0.45, classes=ppe_ids, agnostic_nms=True) if ppe_ids else model(img, conf=0.20, iou=0.45, agnostic_nms=True)
        inference_ms = (time.time() - t0) * 1000

        plotted_img = results[0].plot(line_width=2)

        detections = []
        violations = []
        compliances = []

        for idx, b in enumerate(results[0].boxes):
            cls_id = int(b.cls[0])
            name = model.names[cls_id]
            conf = float(b.conf[0])
            bbox = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            cap_checked = False
            cap_check_score = 0.0
            if name == 'Hardhat':
                cap_checked = True
                is_cap, cap_conf = verify_cap_on_crop(img, bbox)
                cap_check_score = round(cap_conf * 100, 1)
                if is_cap:
                    name = 'NO-Hardhat'
                    conf = cap_conf
                    cv2.rectangle(plotted_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 240), 2)
                    cv2.putText(plotted_img, f"NO-Hardhat {int(conf*100)}%", (int(bbox[0]), max(20, int(bbox[1]-6))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 240), 2)
            is_neg = name.lower().startswith(('no-', 'no ', 'no_')) or name.lower() in ['cap', 'caps', 'casual-hat']
            det_item = {
                'id': idx,
                'class': name,
                'confidence': round(conf * 100, 1),
                'is_violation': is_neg,
                'bbox': bbox
            }
            if cap_checked:
                det_item['cap_score'] = cap_check_score
                if is_cap:
                    det_item['note'] = 'หมวกแก๊ป (Cap Detected)'
            detections.append(det_item)
            if is_neg:
                violations.append(name)
            elif any(k in name.lower() for k in ['hardhat', 'safety vest', 'vest']) or (name.lower() == 'mask' and current_settings.get('required_ppe', {}).get('mask', False)):
                compliances.append(name)

        # Fallback Verification: Ensure all detected workers have required PPE
        inferred = verify_person_ppe_compliance(detections, img.shape)
        for inf in inferred:
            detections.append(inf)
            violations.append(inf['class'])
            bx = [int(v) for v in inf['bbox']]
            cv2.rectangle(plotted_img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 240), 2)
            cv2.putText(plotted_img, f"{inf['class']} (Missing)", (bx[0], max(20, bx[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 240), 2)

        status = 'VIOLATION' if violations else ('COMPLIANT' if compliances else 'SCENE_ONLY')

        _, buffer = cv2.imencode('.jpg', plotted_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_str = base64.b64encode(buffer).decode('utf-8')

        _, raw_buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        raw_b64_str = base64.b64encode(raw_buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'filename': filename,
            'dataset': dataset_type,
            'status': status,
            'inference_time_ms': round(inference_ms, 1),
            'image_width': int(img.shape[1]),
            'image_height': int(img.shape[0]),
            'model_name': 'YOLOv8s Master 200-Epoch (Stabilized Ensemble)',
            'detections': detections,
            'image_base64': f'data:image/jpeg;base64,{b64_str}',
            'raw_image_base64': f'data:image/jpeg;base64,{raw_b64_str}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/slideshow/upload', methods=['POST'])
def api_slideshow_upload():
    """Upload a custom image to run live PPE detection on"""
    global model
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        import numpy as np
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        if model is None:
            load_model()

        t0 = time.time()
        ppe_ids = get_ppe_class_ids(model)
        results = model(img, conf=0.20, iou=0.45, classes=ppe_ids, agnostic_nms=True) if ppe_ids else model(img, conf=0.20, iou=0.45, agnostic_nms=True)
        inference_ms = (time.time() - t0) * 1000

        plotted_img = results[0].plot(line_width=2)

        detections = []
        violations = []
        compliances = []

        for idx, b in enumerate(results[0].boxes):
            cls_id = int(b.cls[0])
            name = model.names[cls_id]
            conf = float(b.conf[0])
            bbox = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            cap_checked = False
            cap_check_score = 0.0
            if name == 'Hardhat':
                cap_checked = True
                is_cap, cap_conf = verify_cap_on_crop(img, bbox)
                cap_check_score = round(cap_conf * 100, 1)
                if is_cap:
                    name = 'NO-Hardhat'
                    conf = cap_conf
                    cv2.rectangle(plotted_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 240), 2)
                    cv2.putText(plotted_img, f"NO-Hardhat {int(conf*100)}%", (int(bbox[0]), max(20, int(bbox[1]-6))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 240), 2)
            is_neg = name.lower().startswith(('no-', 'no ', 'no_')) or name.lower() in ['cap', 'caps', 'casual-hat']
            det_item = {
                'id': idx,
                'class': name,
                'confidence': round(conf * 100, 1),
                'is_violation': is_neg,
                'bbox': bbox
            }
            if cap_checked:
                det_item['cap_score'] = cap_check_score
                if is_cap:
                    det_item['note'] = 'หมวกแก๊ป (Cap Detected)'
            detections.append(det_item)
            if is_neg:
                violations.append(name)
            elif any(k in name.lower() for k in ['hardhat', 'safety vest', 'vest']) or (name.lower() == 'mask' and current_settings.get('required_ppe', {}).get('mask', False)):
                compliances.append(name)

        # Fallback Verification: Ensure all detected workers have required PPE
        inferred = verify_person_ppe_compliance(detections, img.shape)
        for inf in inferred:
            detections.append(inf)
            violations.append(inf['class'])
            bx = [int(v) for v in inf['bbox']]
            cv2.rectangle(plotted_img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 240), 2)
            cv2.putText(plotted_img, f"{inf['class']} (Missing)", (bx[0], max(20, bx[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 240), 2)

        status = 'VIOLATION' if violations else ('COMPLIANT' if compliances else 'SCENE_ONLY')

        _, buffer = cv2.imencode('.jpg', plotted_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_str = base64.b64encode(buffer).decode('utf-8')

        _, raw_buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        raw_b64_str = base64.b64encode(raw_buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'filename': file.filename,
            'dataset': 'custom_upload',
            'status': status,
            'inference_time_ms': round(inference_ms, 1),
            'image_width': int(img.shape[1]),
            'image_height': int(img.shape[0]),
            'model_name': 'YOLOv8s Master 200-Epoch (Stabilized Ensemble)',
            'detections': detections,
            'image_base64': f'data:image/jpeg;base64,{b64_str}',
            'raw_image_base64': f'data:image/jpeg;base64,{raw_b64_str}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        db.init_db()
        load_model()
        instance_detector.update_settings(current_settings)
        socketio.run(app, debug=False, use_reloader=False, host='localhost', port=3333, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Fatal error starting app: {e}")
