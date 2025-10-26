"""
train_yolov5_tensorboard.py
----------------------------
Volledig script om YOLOv5 te trainen met TensorBoard-integratie.

Vereisten:
    pip install torch torchvision torchaudio
    pip install git+https://github.com/ultralytics/yolov5.git
    pip install tensorboard

Gebruik:
    python train_yolov5_tensorboard.py
"""

import os
import subprocess
import time
import torch
import glob

# ========= CONFIG =========
DATA_YAML = "SimpleFruits-1/data.yaml"   # Pad naar je dataset YAML
MODEL_NAME = "yolov5n.pt"                # Basismodel (n, s, m, l, x)
EPOCHS = 30                              # Aantal epochs
IMG_SIZE = 640                           # Beeldresolutie
BATCH_SIZE = 16                          # Batchgrootte
RUN_NAME = "yolov5_tensorboard_run"      # Naam van de run
PROJECT_DIR = "runs/train"               # Locatie voor resultaten
START_TENSORBOARD = True                 # Start TensorBoard automatisch
# ===========================


def check_dataset_structure():
    """Controleer of de dataset correct is opgezet."""
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"❌ data.yaml niet gevonden: {DATA_YAML}")

    base_dir = os.path.dirname(DATA_YAML)
    for folder in ["train/images", "valid/images"]:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Map ontbreekt: {path}")

    print("✅ Datasetstructuur correct.\n")


def start_tensorboard(logdir="runs"):
    """Start TensorBoard in de achtergrond."""
    print("📊 TensorBoard starten...")
    subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", "6006"])
    time.sleep(3)
    print("✅ TensorBoard draait op: http://localhost:6006\n")


def check_tfevents(logdir="runs"):
    """Controleer of TensorBoard logs bestaan."""
    log_files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents*"), recursive=True)
    if not log_files:
        print("⚠️  Geen TensorBoard logs gevonden. Controleer of training correct liep.")
    else:
        print(f"✅ {len(log_files)} TensorBoard logbestand(en) gevonden.\n")


def train_yolov5():
    """Train YOLOv5 met TensorBoard logging."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Apparaat: {device}\n")

    check_dataset_structure()

    # Start TensorBoard
    if START_TENSORBOARD:
        start_tensorboard(PROJECT_DIR)

    print("🚀 Training starten...\n")

    # Voer YOLOv5-trainingsscript aan via subprocess
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

    subprocess.run(cmd, check=True)

    print("\n✅ Training voltooid!")
    run_dir = os.path.join(PROJECT_DIR, RUN_NAME)
    print(f"📁 Resultaten opgeslagen in: {run_dir}\n")

    check_tfevents(run_dir)
    print("📈 Bekijk live resultaten op: http://localhost:6006\n")

    # Return pad naar best.pt
    best_model_path = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"⭐ Beste model opgeslagen op: {best_model_path}")
    else:
        print("⚠️  Kon geen best.pt vinden, controleer de trainingsoutput.")

    return best_model_path


if __name__ == "__main__":
    best_model = train_yolov5()
