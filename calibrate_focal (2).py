
"""
Focal Length Calibration Helper — run this FIRST before face_distance.py.

Why: the depth formula Z = (f * W) / w_px needs YOUR camera's focal
length in pixels. Every laptop camera is different, so we measure it.

How to use:
    1. Stand exactly 0.5 meters (50 cm) away from your camera — use a
       ruler/tape measure, be precise.
    2. Run this script. It detects your face and shows its width in pixels.
    3. Note that pixel width number.
    4. Calculate: FOCAL_LENGTH = (pixel_width * 0.5) / 0.15
       (0.5 = your known distance in meters, 0.15 = average face width in meters)
    5. Paste that FOCAL_LENGTH number into face_distance.py

Requirements:
    pip install opencv-python --break-system-packages

Run:
    py calibrate_focal.py
"""

import cv2

KNOWN_DISTANCE_M = 0.5     # stand exactly this far from the camera
REAL_FACE_WIDTH_M = 0.15   # average face width

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        focal_length = (fw * KNOWN_DISTANCE_M) / REAL_FACE_WIDTH_M
        cv2.putText(frame, f"Face width (px): {fw}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Calculated FOCAL_LENGTH: {focal_length:.0f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"\rFace width: {fw}px  ->  FOCAL_LENGTH = {focal_length:.0f}", end="")

    cv2.imshow("Calibration - stand 0.5m away", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
