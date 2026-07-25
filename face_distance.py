"""
Monocular Face Distance Estimation — Task 2

Uses the pinhole camera model:
    Depth:  Z = (f * W) / w_px
    Angle:  theta = arctan((x - c_x) / f)

Before running this: run calibrate_focal.py first, stand exactly
0.5m from your camera, and paste the FOCAL_LENGTH value it prints
into FOCAL_LENGTH below.

Run:
    py face_distance.py
"""

import cv2
import math

# ---- PASTE YOUR CALIBRATED VALUE HERE ----
FOCAL_LENGTH = 245      # <-- replace with number from calibrate_focal.py
REAL_FACE_WIDTH_M = 0.15  # average real face width (must match calibration)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
c_x = frame_w / 2
c_y = frame_h / 2

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        # face center pixel
        face_center_x = x + fw / 2

        # --- Depth: Z = (f * W) / w_px ---
        Z = (FOCAL_LENGTH * REAL_FACE_WIDTH_M) / fw

        # --- Angle: theta = arctan((x - c_x) / f) ---
        theta_rad = math.atan((face_center_x - c_x) / FOCAL_LENGTH)
        theta_deg = math.degrees(theta_rad)

        label = f"Z={Z*100:.1f}cm  theta={theta_deg:.1f}deg"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        print(f"\rDepth: {Z*100:.1f} cm   Angle: {theta_deg:.1f} deg", end="")

    # draw image center crosshair for reference
    cv2.drawMarker(frame, (int(c_x), int(c_y)), (255, 0, 0),
                    markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

    cv2.imshow("Face Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()