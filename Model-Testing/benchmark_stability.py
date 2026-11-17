"""
Benchmark Stability & Flicker Measurement
Compares Raw YOLO Detection vs Stabilized Temporal PPE Pipeline across video frames.
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add Model-Deployment to sys.path to import PPETemporalStabilizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Model-Deployment')))
from ppe_stabilizer import PPETemporalStabilizer

def run_benchmark():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'))
    print(f"[+] Loading model from: {model_path}")
    model = YOLO(model_path)

    videos = [
        ('indianworkers.mp4', 120),
        ('hardhat.mp4', 120),
        ('JapanPPE.mp4', 120),
    ]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Model-Training/source_files'))

    print("\n" + "="*75)
    print("  PPE DETECTION FLICKER & STABILITY BENCHMARK (120 Frames per Video)")
    print("="*75)

    for vname, max_frames in videos:
        vpath = os.path.join(base_dir, vname)
        if not os.path.exists(vpath):
            print(f"[-] Video not found: {vpath}")
            continue

        cap = cv2.VideoCapture(vpath)
        stabilizer_single = PPETemporalStabilizer(iou_match_thresh=0.40, max_grace_frames=6, alpha=0.72)
        stabilizer_multi = PPETemporalStabilizer(iou_match_thresh=0.40, max_grace_frames=6, alpha=0.72)

        raw_history = []
        smooth_history = []
        raw_status_history = []
        smooth_status_single = []
        smooth_status_multi = []

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Raw detection
            results = model(frame, conf=0.25, agnostic_nms=True, verbose=False)[0]
            raw_dets = []
            for b in results.boxes:
                cls_name = model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                bbox = [float(x) for x in b.xyxy[0].tolist()]
                raw_dets.append({'class': cls_name, 'confidence': conf, 'bbox': bbox})

            # Check raw compliance
            raw_persons = any(d['class'].lower() == 'person' for d in raw_dets)
            raw_has_violation = any(d['class'].lower().startswith(('no-', 'no ', 'no_')) for d in raw_dets)
            raw_compliant = (not raw_has_violation) if raw_persons else True
            raw_status_history.append(raw_compliant)
            raw_history.append([d['class'] for d in raw_dets])

            # Smoothed & stabilized detection (single mode)
            _, sum_single = stabilizer_single.update(raw_dets, frame.shape, mode='single')
            smooth_status_single.append(sum_single['is_compliant'])

            # Smoothed & stabilized detection (multi mode)
            smooth_dets_m, sum_multi = stabilizer_multi.update(raw_dets, frame.shape, mode='multi')
            smooth_status_multi.append(sum_multi['is_compliant'])
            smooth_history.append([d['class'] for d in smooth_dets_m])

        cap.release()

        # Compute transitions / flickers
        print(f"\n🎥 Video: {vname} ({frame_idx} frames evaluated)")
        print("-" * 70)
        
        # Overall Compliance Status flips
        raw_status_flips = sum(1 for i in range(1, len(raw_status_history)) if raw_status_history[i] != raw_status_history[i-1])
        single_flips = sum(1 for i in range(1, len(smooth_status_single)) if smooth_status_single[i] != smooth_status_single[i-1])
        multi_flips = sum(1 for i in range(1, len(smooth_status_multi)) if smooth_status_multi[i] != smooth_status_multi[i-1])
        
        print(f"  🔴 Compliance Verdict Flips (Green/Red Flashing):")
        print(f"     -> Raw: {raw_status_flips} flips")
        print(f"     -> Smoothed Single Mode (Kiosk / Webcam): {single_flips} flips (Reduction: {((raw_status_flips - single_flips)/max(1, raw_status_flips)*100):.1f}%)")
        print(f"     -> Smoothed Multi Mode (Full Scene): {multi_flips} flips (Reduction: {((raw_status_flips - multi_flips)/max(1, raw_status_flips)*100):.1f}%)")

        # Class flickers
        print(f"  📦 Per-Class Presence & Rapid Flips (Multi Mode):")
        tracked_classes = ['Hardhat', 'NO-Hardhat', 'Safety Vest', 'NO-Safety Vest', 'Person']
        for c in tracked_classes:
            raw_pres = sum(1 for f in raw_history if c in f)
            smooth_pres = sum(1 for f in smooth_history if c in f)
            raw_flips = sum(1 for i in range(1, len(raw_history)) if (c in raw_history[i]) != (c in raw_history[i-1]))
            smooth_flips = sum(1 for i in range(1, len(smooth_history)) if (c in smooth_history[i]) != (c in smooth_history[i-1]))
            print(f"     - {c:15s}: Raw {raw_pres:3d}/{frame_idx} (flips: {raw_flips:2d})  -->  Smooth {smooth_pres:3d}/{frame_idx} (flips: {smooth_flips:2d})")

    print("\n" + "="*75)
    print("  BENCHMARK COMPLETE")
    print("="*75)

if __name__ == '__main__':
    run_benchmark()
