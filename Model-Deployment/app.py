from flask import Flask, render_template, Response, jsonify, request, send_file
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
import checklog
import json

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

SETTINGS_FILE = 'settings.json'
DEFAULT_SETTINGS = {
    'required_ppe': {
        'helmet': True,
        'vest': True,
        'mask': False
    },
    'non_compliance_delay': 3,
    'instance_reset_timeout': 5,
    'detection_mode': 'single'  # 'single' or 'multi'
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

def load_model():
    """Load Master 200-Epoch Super PPE Model + Dedicated Cap Verification Model"""
    global model, cap_verifier
    super_model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'
    ))
    cap_model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../Model-Training/Outputs/runs/detect/yolov8s_ppe_model_with_cap_class/weights/best.pt'
    ))
    
    candidates = [
        super_model_path,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../Model-Training/Outputs/runs/detect/yolov8s_ppe_finetuned_super/weights/best.pt'))
    ]
    chosen = None
    for p in candidates:
        if os.path.exists(p):
            chosen = p
            break

    if chosen:
        print(f"[+] [EXCLUSIVE] Loaded Master PPE Model: {chosen}")
        model = YOLO(chosen)
    else:
        raise FileNotFoundError(f"[-] Critical: PPE model not found at {super_model_path}")

    if os.path.exists(cap_model_path):
        print(f"[+] [EXCLUSIVE] Loaded Cap Verification Model: {cap_model_path}")
        cap_verifier = YOLO(cap_model_path)
    else:
        print(f"[!] Warning: Cap verification model not found at {cap_model_path}")

def verify_cap_on_crop(img, bbox):
    """If a Hardhat is candidate-detected, verify whether it is actually a baseball cap (ผ้า/หมวกแก๊ป)"""
    global cap_verifier
    if cap_verifier is None:
        return False, 0.0
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img.shape[:2]
        pad_w = int((x2 - x1) * 0.12)
        pad_h = int((y2 - y1) * 0.12)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return False, 0.0
        c_res = cap_verifier.predict(crop, conf=0.50, verbose=False)[0]
        max_cap = 0.0
        for cb in c_res.boxes:
            if cap_verifier.names[int(cb.cls[0])].lower() == 'cap':
                max_cap = max(max_cap, float(cb.conf[0]))
        if max_cap >= 0.82:
            return True, max_cap
    except Exception:
        pass
    return False, 0.0

def get_ppe_class_ids(model_inst, include_mask=None):
    """Filter class IDs to focus STRICTLY on worker PPE attire & person, ignoring vehicles & machinery.
    Mask & NO-Mask are excluded by default because masks are not standard construction PPE,
    unless explicitly enabled in Settings ('mask': True).
    """
    if not model_inst or not hasattr(model_inst, 'names'):
        return None

    global current_settings
    if include_mask is None:
        include_mask = current_settings.get('required_ppe', {}).get('mask', False) if current_settings else False

    excluded = ['vehicle', 'machinery', 'excavator', 'ladder', 'truck', 'sedan', 'van', 'bus', 'semi', 'trailer', 'loader', 'hydrant', 'suv', 'cone']
    allowed = ['hardhat', 'vest', 'person', 'glove', 'cap']
    if include_mask:
        allowed.append('mask')

    target_ids = []
    for cid, name in model_inst.names.items():
        n = name.lower()
        if any(ex in n for ex in excluded):
            continue
        if not include_mask and 'mask' in n:
            continue
        if any(al in n for al in allowed):
            target_ids.append(cid)
    return target_ids if target_ids else None

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
            results = model(frame, classes=ppe_ids, agnostic_nms=True) if ppe_ids else model(frame, agnostic_nms=True)
            
            all_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    if class_name == 'Hardhat':
                        is_cap, cap_conf = verify_cap_on_crop(frame, [x1, y1, x2, y2])
                        if is_cap:
                            class_name = 'Cap'
                            conf = cap_conf
                    
                    all_detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            kiosk_state = kiosk.update(all_detections, frame.shape, current_settings)
            if frame_count % 3 == 0 and kiosk_state['state'] in ('idle', 'checking'):
                try:
                    code, _, _ = qr.detectAndDecode(frame)
                except cv2.error:
                    code = ''
                if code and recorder.set_code(code) and kiosk_state['state'] == 'checking':
                    kiosk.restart()  # เริ่มนับเวลาตรวจใหม่หลังแสดงบัตร
            logged = recorder.observe(kiosk_state)
            hardware.update(kiosk_state)

            instance_result = instance_detector.process_detection(all_detections, dev_mode, current_settings)
            is_compliant = compliance_checker.check_compliance(instance_result, dev_mode)
            
            annotated_frame = results[0].plot()
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
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    return jsonify(state)

@app.route('/logs')
def logs_page():
    """Check history and employee list"""
    return render_template('logs.html')

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

@app.route('/api/kiosk_id', methods=['POST'])
def api_kiosk_id():
    """Manual entry or USB barcode scanner (acts as a keyboard)"""
    code = ((request.json or {}).get('code') or '').strip()
    if not checklog.valid_code(code):
        return jsonify({'status': 'error', 'message': 'รหัสไม่ถูกต้อง'}), 400
    state = kiosk.get_state()['state']
    if state in ('pass', 'fail'):
        return jsonify({'status': 'error', 'message': 'กำลังแสดงผล รอสักครู่'}), 409
    if recorder.set_code(code) and state == 'checking':
        kiosk.restart()
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
    """Update settings"""
    global current_settings
    try:
        new_settings = request.json
        current_settings = new_settings
        
        instance_detector.update_settings(new_settings)
        
        if save_settings_to_file(new_settings):
            return jsonify({'status': 'success', 'message': 'Settings saved'})
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

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
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
        if model is None:
            load_model()
        
        with stream_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(int(os.environ.get('PPE_CAMERA_INDEX', 0)))
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        kiosk.reset()
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
        results = model(img, classes=ppe_ids, agnostic_nms=True) if ppe_ids else model(img, agnostic_nms=True)
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
                    name = 'Cap'
                    conf = cap_conf
                    cv2.rectangle(plotted_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 240), 2)
                    cv2.putText(plotted_img, f"Cap {int(conf*100)}%", (int(bbox[0]), max(20, int(bbox[1]-6))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 240), 2)
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
            detections.append(det_item)
            if is_neg:
                violations.append(name)
            elif any(k in name.lower() for k in ['hardhat', 'safety vest', 'vest']) or (name.lower() == 'mask' and current_settings.get('required_ppe', {}).get('mask', False)):
                compliances.append(name)

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
            'model_name': 'YOLOv8s Ensemble (Master 200-Epoch + Cap Verifier)',
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
        results = model(img, classes=ppe_ids, agnostic_nms=True) if ppe_ids else model(img, agnostic_nms=True)
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
                    name = 'Cap'
                    conf = cap_conf
                    cv2.rectangle(plotted_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 240), 2)
                    cv2.putText(plotted_img, f"Cap {int(conf*100)}%", (int(bbox[0]), max(20, int(bbox[1]-6))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 240), 2)
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
            detections.append(det_item)
            if is_neg:
                violations.append(name)
            elif any(k in name.lower() for k in ['hardhat', 'safety vest', 'vest']) or (name.lower() == 'mask' and current_settings.get('required_ppe', {}).get('mask', False)):
                compliances.append(name)

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
            'model_name': 'YOLOv8s Ensemble (Master 200-Epoch + Cap Verifier)',
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
        socketio.run(app, debug=True, host='localhost', port=3333, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Fatal error starting app: {e}")
