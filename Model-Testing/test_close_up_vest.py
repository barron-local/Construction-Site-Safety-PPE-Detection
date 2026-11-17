import cv2
import os
import sys
from ultralytics import YOLO

sys.path.append(os.path.abspath('Model-Deployment'))
from ppe_stabilizer import PPETemporalStabilizer

master_path = os.path.abspath('Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt')
master = YOLO(master_path)
img_path = r'C:\Users\poohc\.gemini\antigravity\brain\2840c8b6-38ae-4a8d-b752-727b78940139\.user_uploaded\media_1794896109034_f480641c.jpg'
img = cv2.imread(img_path)

res = master(img, conf=0.25, verbose=False)[0]
raw_dets = []
for b in res.boxes:
    c = master.names[int(b.cls[0])]
    conf = float(b.conf[0])
    box = [int(v) for v in b.xyxy[0]]
    raw_dets.append({'class': c, 'confidence': conf, 'bbox': box})

st = PPETemporalStabilizer()
for _ in range(10):
    st_dets, summary = st.update(raw_dets, img.shape, mode='single')

print("--- DETECTIONS IN SINGLE (KIOSK) MODE ---")
for d in st_dets:
    print(f"  {d['class']}: conf={d['confidence']:.2f}, inferred={d.get('is_inferred', False)}, box={d['bbox']}")

print(f"Summary compliance: {summary['is_compliant']}")
print(f"Missing PPE: {summary['missing_ppe']}")

annotated = st.render(img, st_dets, summary)
out_path = 'Model-Testing/rendered_close_up_fixed.jpg'
cv2.imwrite(out_path, annotated)
print(f"Rendered output saved to {out_path}")
