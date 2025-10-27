import os
import yaml
import subprocess
import torch

# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # map van dit script
DATASET_DIR = os.path.join(BASE_DIR, "SimpleFruits-1")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

MODEL_NAME = "yolov5n.pt"
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640
PROJECT_DIR = os.path.join(BASE_DIR, "runs", "train")
RUN_NAME = "yolov5_auto_run"

# === 1️⃣ Pas data.yaml paden aan naar absolute paden ===
with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

# Absolute paden berekenen
data["train"] = os.path.abspath(os.path.join(DATASET_DIR, "train/images"))
data["val"]   = os.path.abspath(os.path.join(DATASET_DIR, "valid/images"))

with open(DATA_YAML, "w") as f:
    yaml.safe_dump(data, f)

print("✅ data.yaml bijgewerkt met absolute paden:")
print(f"train: {data['train']}")
print(f"val:   {data['val']}\n")

# === 2️⃣ Selecteer device automatisch ===
device = "0" if torch.cuda.is_available() else "cpu"
print(f"💻 Apparaat: {device}\n")

# === 3️⃣ Start YOLOv5 training ===
cmd = [
    "python", "-m", "yolov5.train",
    "--data", DATA_YAML,
    "--weights", MODEL_NAME,
    "--epochs", str(EPOCHS),
    "--img", str(IMG_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--project", PROJECT_DIR,
    "--name", RUN_NAME,
    "--exist-ok",
    "--device", device
]

print("🚀 Training starten...\n")
subprocess.run(cmd, check=True)
