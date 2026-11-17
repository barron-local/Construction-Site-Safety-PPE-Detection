"""
Real-Time PPE Detection with Temporal Smoothing & Anti-Flicker Stabilizer
Run this script to test your webcam feed with rock-solid, smooth bounding boxes.
"""
import cv2
import os
import sys
from ultralytics import YOLO

# Add Model-Deployment to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Model-Deployment')))
from ppe_stabilizer import PPETemporalStabilizer, box_intersection_ratio

def main():
    model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'
    ))
    
    print("=" * 65)
    print("  🏗️  PPE DETECTION - STABILIZED REAL-TIME WEBCAM (NO-FLICKER)")
    print(f"  -> Model: {os.path.basename(model_path)}")
    print("  -> Press 'q' to exit")
    print("=" * 65)

    model = YOLO(model_path)
    cap_model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../Model-Training/Outputs/runs/detect/yolov8s_ppe_model_with_cap_class/weights/best.pt'
    ))
    cap_verifier = YOLO(cap_model_path) if os.path.exists(cap_model_path) else None
    stabilizer = PPETemporalStabilizer(iou_match_thresh=0.35, max_grace_frames=5, alpha=0.72)

    def is_headwear_a_cap(img, head_box, current_class='Hardhat'):
        if cap_verifier is None or img is None:
            return False, 0.0
        try:
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
        except Exception:
            pass
        return False, 0.0

    cam_idx = int(os.environ.get('PPE_CAMERA_INDEX', 0))
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[-] Cannot open webcam at index {cam_idx}. Trying fallback index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[-] No webcam found. Exiting.")
            return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Inference with 200-epoch master model (strict NMS, sensitive 0.20 conf for rapid vest pickup)
        results = model(frame, conf=0.20, iou=0.45, imgsz=640, agnostic_nms=True, verbose=False)[0]
        raw_dets = []
        person_boxes = []
        for b in results.boxes:
            cls_name = model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            bbox = [float(x) for x in b.xyxy[0].tolist()]
            raw_dets.append({'class': cls_name, 'confidence': conf, 'bbox': bbox})
            if cls_name.lower() == 'person':
                person_boxes.append(bbox)

        # Direct Headwear Cap Verification
        for det in raw_dets:
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
        stable_dets, summary = stabilizer.update(raw_dets, frame.shape)

        # Render smoothed, non-jittering bounding boxes
        annotated_frame = stabilizer.render(frame, stable_dets, summary)

        cv2.imshow('Stabilized PPE Detection (Smooth & Rock Solid)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
