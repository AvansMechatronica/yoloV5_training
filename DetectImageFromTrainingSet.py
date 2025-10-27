"""
detect_yolov8_image.py
----------------------
Voer objectdetectie uit met YOLOv8 op één afbeelding.
Vereisten:
    pip install ultralytics opencv-python
"""

import cv2
from ultralytics import YOLO
import easygui
import random
import os

# === CONFIG ===
MODEL_PATH = "runs/detect/yolo8_tensorboard_fixed/weights/best.pt"  # Je eigen model
IMAGE_PATH = "test_image.jpg"                                 # Testafbeelding
CONF_THRESHOLD = 0.3
DATASET_DIR = "SimpleFruits-1"  # <--- pas aan indien nodig
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train", "images")
# Minimale zekerheid
# ===============

# === Willekeurige afbeelding kiezen ===
image_files = [
    f for f in os.listdir(TRAIN_IMAGES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
if not image_files:
    raise FileNotFoundError(f"Geen afbeeldingen gevonden in {TRAIN_IMAGES_DIR}")

random_image = random.choice(image_files)
bestand = os.path.join(TRAIN_IMAGES_DIR, random_image)


# 🔹 Laad model
model = YOLO(MODEL_PATH)
print(f"✅ Model geladen: {MODEL_PATH}")

# 🔍 Voer detectie uit
results = model.predict(bestand, conf=CONF_THRESHOLD)

# 🎯 Haal eerste resultaat (voor één afbeelding)
result = results[0]
img = result.orig_img.copy()

print("\n=== Detectieresultaten ===")
for box in result.boxes:
    cls_id = int(box.cls[0])              # Klasse-index
    label = result.names[cls_id]          # Klassenaam
    conf = float(box.conf[0])             # Betrouwbaarheid
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coördinaten

    # Print resultaat
    print(f"🧩 {label} ({conf:.2f}) op ({x1}, {y1}), ({x2}, {y2})")

    # Teken bounding box en label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 🖼️ Toon resultaat
cv2.imshow("YOLOv8 Objectdetectie", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
