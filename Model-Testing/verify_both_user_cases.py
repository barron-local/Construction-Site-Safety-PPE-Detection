import cv2
import os
import sys
from ultralytics import YOLO

sys.path.append(os.path.abspath('Model-Deployment'))
from ppe_stabilizer import PPETemporalStabilizer

master_path = os.path.abspath('Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt')
cap_model_path = os.path.abspath('Model-Training/Outputs/runs/detect/yolov8s_ppe_model_with_cap_class/weights/best.pt')

master = YOLO(master_path)
cap_verifier = YOLO(cap_model_path)

def is_headwear_a_cap(img, head_box, current_class='Hardhat'):
    """
    Checks if a detected headwear box (Hardhat or NO-Hardhat) is actually a baseball cap.
    Uses 30% spatial context padding, CLAHE contrast enhancement for dark/black caps, and competitive arbitration.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in head_box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_w = max(12, int(bw * 0.30))
    pad_h = max(12, int(bh * 0.30))
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
        return False, 0.0

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    r = cap_verifier(enhanced_crop, imgsz=224, conf=0.15, verbose=False)[0]
    caps = [float(b.conf[0]) for b in r.boxes if cap_verifier.names[int(b.cls[0])].lower() == 'cap']
    hardhats = [float(b.conf[0]) for b in r.boxes if cap_verifier.names[int(b.cls[0])].lower() == 'hardhat']
    no_hardhats = [float(b.conf[0]) for b in r.boxes if cap_verifier.names[int(b.cls[0])].lower() == 'no-hardhat']
    max_c = max(caps) if caps else 0.0
    max_h = max(hardhats) if hardhats else 0.0
    max_no_h = max(no_hardhats) if no_hardhats else 0.0
    
    min_thresh = 0.62 if current_class == 'Hardhat' else 0.65
    if max_c >= min_thresh and max_c > max_h and max_c > max_no_h:
        return True, max_c
    return False, 0.0

def process_and_test(img_path, title):
    print("=" * 60)
    print(f" TESTING: {title}")
    print("=" * 60)
    img = cv2.imread(img_path)
    assert img is not None, f"Could not load {img_path}"
    
    res = master(img, conf=0.25, verbose=False)[0]
    raw_dets = []
    for b in res.boxes:
        cname = master.names[int(b.cls[0])]
        conf = float(b.conf[0])
        box = [int(v) for v in b.xyxy[0]]
        is_cap = False
        note = None
        if cname in ['Hardhat', 'NO-Hardhat']:
            is_c, cap_score = is_headwear_a_cap(img, box, current_class=cname)
            if is_c:
                cname = 'NO-Hardhat'
                is_cap = True
                conf = max(conf, cap_score)
                note = 'หมวกแก๊ป (Cap Detected)'
        
        raw_dets.append({
            'class': cname,
            'confidence': conf,
            'bbox': box,
            'is_cap': is_cap,
            'is_violation': (cname.lower().startswith(('no-', 'no ', 'no_')) or is_cap),
            'note': note
        })
    
    stabilizer = PPETemporalStabilizer()
    st_dets, summary = stabilizer.update(raw_dets, img.shape)
    annotated = stabilizer.render(img, st_dets, summary)
    out_name = f"Model-Testing/output_{os.path.basename(img_path)}.jpg"
    cv2.imwrite(out_name, annotated)
    
    for d in st_dets:
        print(f"  Class: {d['class']}, Conf: {d['confidence']:.2f}, is_cap: {d.get('is_cap', False)}, is_violation: {d.get('is_violation', False)}")
    print(f"  Summary Compliance: {summary.get('is_compliant')}, Missing: {summary.get('missing_ppe')}")
    print(f"  Saved render to {out_name}")

if __name__ == '__main__':
    yellow_img = r'C:\Users\poohc\.gemini\antigravity\brain\2840c8b6-38ae-4a8d-b752-727b78940139\.user_uploaded\media_1794894660412_7dc25b74.jpg'
    polo_img = r'C:\Users\poohc\.gemini\antigravity\brain\2840c8b6-38ae-4a8d-b752-727b78940139\.user_uploaded\media_1794895226772_ce244cb5.png'
    
    process_and_test(yellow_img, "USER YELLOW HARDHAT (MUST BE COMPLIANT HARDHAT)")
    process_and_test(polo_img, "USER POLO CAP (MUST BE VIOLATION NO-HARDHAT (CAP))")
