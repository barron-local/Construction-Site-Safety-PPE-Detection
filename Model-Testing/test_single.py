from ultralytics import YOLO
import os

model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt'
))
model = YOLO(model_path)

test_img = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '../Model-Training/Dataset/test/images/ppe_1073_jpg.rf.72ea8a293a4f3e1135219e33701b1099.jpg'
))

if os.path.exists(test_img):
    results = model(test_img, conf=0.25)
    print("Detections:")
    for b in results[0].boxes:
        print(f"  {model.names[int(b.cls[0])]}: {float(b.conf[0]):.2f}")
    results[0].show()
else:
    print(f"Test image not found at {test_img}")
