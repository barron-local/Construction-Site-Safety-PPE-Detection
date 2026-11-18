import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "Model-Training" / "Outputs" / "runs" / "detect" / "yolov8s_ppe_css_200_epochs" / "results.csv"
DATASET_DIR = BASE_DIR / "Model-Training" / "Dataset"


def get_detection_error(log_path):
    """Calculate Detection Error Rate from Best Checkpoint (Epoch 166)."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with open(log_path, mode="r", encoding="utf-8") as f:
        rows = [{k.strip(): float(v.strip()) for k, v in r.items() if v.strip()} for r in csv.DictReader(f)]

    best_row = max(rows, key=lambda r: r.get("metrics/mAP50(B)", 0))
    precision = best_row.get("metrics/precision(B)", 0)
    return (1.0 - precision) * 100


def get_total_dataset(dataset_dir):
    """Count total images (4 sub-images per raw file) and annotated objects."""
    total_imgs = 0
    total_boxes = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if img_dir.exists():
            total_imgs += sum(1 for f in img_dir.iterdir() if f.suffix.lower() in valid_exts) * 4

        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.suffix.lower() == ".txt":
                    with open(f, "r", encoding="utf-8") as fp:
                        total_boxes += sum(1 for line in fp if line.strip())

    return total_imgs, total_boxes


def main():
    error_rate = get_detection_error(LOG_PATH)
    total_imgs, total_boxes = get_total_dataset(DATASET_DIR)

    print(f"Detection Error Rate: {error_rate:.2f}%")
    print(f"Total Dataset: {total_imgs:,} [Objects: {total_boxes:,}]")


if __name__ == "__main__":
    main()
