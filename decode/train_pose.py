from ultralytics import YOLO


def main():
  # 1. Load the pretrained YOLOv8 nano pose model
  model = YOLO("yolov8n-pose.pt")

  # 2. Train the model using your custom dataset configuration
  # Assuming your dataset YAML is located inside robot_pose_poc/dataset/data.yaml
  model.train(
      data="robot_pose_poc/dataset/data.yaml",
      epochs=30,  # 30 epochs is ideal for a small custom PoC
      imgsz=640,  # Standard input image size for YOLO
      device="cpu",  # Forces local CPU execution
      batch=2,  # Small batch size to run smoothly on local hardware
  )


if __name__ == "__main__":
  main()
