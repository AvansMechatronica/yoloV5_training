"""
realtime_yolov5_webcam_fixed.py
-------------------------------
Realtime objectdetectie met YOLOv5 via webcam.
Toont bounding boxes, labels, FPS en print resultaten in terminal.
"""

import torch
import cv2
import time
import numpy as np

# 🔹 YOLOv5 laden (automatisch downloaden indien nodig)
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Gebruik eventueel je eigen model:
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/yolov5_tensorboard_run/weights/best.pt')

# 🎥 Webcam openen (0 = standaardcamera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kan de webcam niet openen.")
    exit()

print("✅ Webcam gestart — druk op 'q' om te stoppen.\n")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Geen frame ontvangen.")
        break

    # 🔍 YOLOv5 voorspelling uitvoeren
    results = model(frame)

    # 🧾 Resultaten in terminal tonen
    df = results.pandas().xyxy[0]
    for _, row in df.iterrows():
        label = row['name']
        conf = row['confidence']
        coords = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
        print(f"📦 {label} ({conf:.2f}) — {coords}")

    # 🎯 Bounding boxes tekenen
    annotated_frame = np.copy(results.render()[0])  # 👈 maak array beschrijfbaar

    # ⏱️ FPS berekenen
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # 🎞️ FPS weergeven in beeld
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 🖼️ Frame tonen
    cv2.imshow("YOLOv5 Realtime Webcam", annotated_frame)

    # ⏹️ Stop met 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Opruimen
cap.release()
cv2.destroyAllWindows()
print("\n✅ Webcam gestopt. Programma beëindigd.")
