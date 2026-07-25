"""
Ball Detection Qualifier Task — AI Version (YOLOv8)
Uses a small pretrained object detection model (YOLOv8-nano) to detect
a "sports ball" in real time. Works on any ball color/pattern, unlike
color-matching. Runs on CPU.

Model: YOLOv8n, pretrained on COCO dataset (Ultralytics, open-source).
COCO class 32 = "sports ball" — disclosed here per hackathon rules
(pretrained weights allowed if architecture/training is disclosed).

Requirements:
    pip install ultralytics opencv-python --break-system-packages
    (first run auto-downloads yolov8n.pt, ~6MB)

Run:
    python ball_detection_ai.py

Controls:
    q = quit
"""

"""
Ball Detection Qualifier Task — AI Version (YOLOv8) — Speed Optimized
Target: ~20 FPS on CPU via frame skipping + reduced inference size.import cv2
import time
"""
from ultralytics import YOLO

BALL_CLASS_ID = 32  # "sports ball" in COCO dataset
CONFIDENCE_THRESHOLD = 0.5  # lower = more detections but more false positives


def main():
    print("Loading YOLOv8n model (first run downloads ~6MB)...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    prev_time = time.time()
    fps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- run AI detection on this frame ----
        results = model(frame, verbose=False)[0]

        detected = False
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == BALL_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"Ball {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---- live FPS ----
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else fps
        prev_time = curr_time

        status = "DETECTED" if detected else "NOT FOUND"
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Model: YOLOv8n (AI)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Ball Detection (AI)", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
