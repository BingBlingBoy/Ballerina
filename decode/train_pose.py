from pathlib import Path
from ultralytics import YOLO


def main():
  # 1. Define your decode directory path dynamically
  decode_dir = Path(__file__).resolve().parent

  # 2. Load the pretrained YOLOv8 nano pose model
  model = YOLO("yolov8n-pose.pt")

  # 3. Train the model and force output runs inside decode/runs/
  model.train(
      data="robot_pose_poc/dataset/data.yaml",
      epochs=30,
      imgsz=640,
      device="cpu",
      batch=2,
      project=str(decode_dir / "runs"),  # Forces base output into decode/runs
      name="pose_train",  # Folder name for this specific run
  )


if __name__ == "__main__":
  main()