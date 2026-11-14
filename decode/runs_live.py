import sys
import cv2
from ultralytics import YOLO

print("1. Loading model weights...")
try:
  model = YOLO("runs/pose_train/weights/best.pt")
  print("-> Model loaded successfully!")
except Exception as e:
  print(f"-> Error loading model: {e}")
  sys.exit(1)

print("2. Attempting to open webcam (index 0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("-> Error: Could not open webcam index 0.")
  print("   Make sure no other app (like Zoom or Teams) is using your camera.")
  sys.exit(1)

print("-> Webcam opened successfully! Starting video loop. Press 'q' to quit.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("-> Error: Failed to read frame from webcam.")
    break

  results = model(frame, verbose=False)
  annotated_frame = results[0].plot()

  cv2.imshow("Hand Pose PoC - Live Test", annotated_frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    print("-> Exiting by user request.")
    break

cap.release()
cv2.destroyAllWindows()