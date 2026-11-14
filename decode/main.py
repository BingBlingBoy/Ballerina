from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from robot_vision.robot_pose_features import (
    CALIBRATION_FILE,
    LETTER_LABELS,
    MODEL_SCHEMA_VERSION,
    camera_matrix_for_frame,
    detect_markers,
    estimate_marker_poses,
    extract_pose_features,
    load_camera_calibration,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIR / "models" / "pose_letter_classifier.joblib"
CAMERA_INDEX = 0
MIN_CONFIDENCE = 0.60
SMOOTHING_WINDOW = 5
MIN_STABLE_VOTES = 3


class PredictionSmoother:
    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.state = None

    def reset(self):
        self.history.clear()
        self.state = None

    def update(self, label):
        self.history.append(label)
        if label is not None:
            count = sum(value == label for value in self.history)
            if count >= MIN_STABLE_VOTES:
                self.state = label
                return self.state

        if self.state is not None:
            count = sum(value == self.state for value in self.history)
            if count >= 2:
                return self.state

        self.state = None
        return None


def load_model():
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_FILE}. "
            "Run train_classifier.py after collecting A-Z poses."
        )
    bundle = joblib.load(MODEL_FILE)
    if bundle.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(
            "Model feature schema is incompatible. Retrain with train_classifier.py."
        )
    classes = tuple(bundle.get("classes", ()))
    if not classes or any(label not in LETTER_LABELS for label in classes):
        raise ValueError(
            "Model bundle has no valid letter classes. Retrain the model."
        )
    model = bundle.get("model")
    if model is None or tuple(model.classes_) != classes:
        raise ValueError(
            "Model classes do not match the saved class metadata. Retrain the model."
        )
    return model, classes


def predict_letter(model, features):
    probabilities = model.predict_proba(np.asarray(features).reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    label = model.classes_[best_index]
    return (label if confidence >= MIN_CONFIDENCE else None), confidence


def process_camera():
    model, classes = load_model()
    calibration = load_camera_calibration(CALIBRATION_FILE)
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}.")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoother = PredictionSmoother()
    consecutive_read_failures = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 10:
                    raise RuntimeError("Camera repeatedly failed to provide frames.")
                continue
            consecutive_read_failures = 0

            camera_matrix, distortion = camera_matrix_for_frame(calibration, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray)
            poses = estimate_marker_poses(corners, ids, camera_matrix, distortion)
            features = extract_pose_features(poses)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if features is None:
                smoother.reset()
                text = "Unknown: show all markers 0-5"
                colour = (0, 0, 255)
            else:
                label, confidence = predict_letter(model, features)
                state = smoother.update(label)
                text = (
                    f"{state} ({confidence:.0%})"
                    if state is not None
                    else f"Uncertain ({confidence:.0%})"
                )
                colour = (0, 255, 0) if state is not None else (0, 165, 255)

            cv2.putText(
                frame,
                text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colour,
                2,
            )
            cv2.putText(
                frame,
                f"Classes {','.join(classes)} | Esc: quit",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Robot A-Z Pose Classifier", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_camera()
