import cv2
import os
import sys
from ultralytics import YOLO

sys.path.append(os.path.abspath('Model-Deployment'))
from ppe_stabilizer import PPETemporalStabilizer

master_path = os.path.abspath('Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt')
master = YOLO(master_path)
img_path = r'C:\Users\poohc\.gemini\antigravity\brain\2840c8b6-38ae-4a8d-b752-727b78940139\.user_uploaded\media_1794897335662_ac600e77.jpg'
img = cv2.imread(img_path)

res = master(img, conf=0.20, verbose=False)[0]
raw_dets = []
for b in res.boxes:
    c = master.names[int(b.cls[0])]
    conf = float(b.conf[0])
    box = [int(v) for v in b.xyxy[0]]
    raw_dets.append({'class': c, 'confidence': conf, 'bbox': box})

print("Raw detections from YOLO:")
for d in raw_dets:
    print(f"  {d['class']}: conf={d['confidence']:.2f}, box={d['bbox']}")

st = PPETemporalStabilizer()
for f in range(5):
    st_dets, summary = st.update(raw_dets, img.shape, mode='single')

print("\nStabilized detections in SINGLE mode:")
for d in st_dets:
    print(f"  {d['class']}: conf={d['confidence']:.2f}, box={d['bbox']}")

print(f"\nSummary: is_compliant={summary['is_compliant']}, missing={summary['missing_ppe']}")
