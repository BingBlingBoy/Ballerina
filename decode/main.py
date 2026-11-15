from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from robot_vision.robot_pose_features import (
    CALIBRATION_FILE,
    LETTER_LABELS,
    MIN_CLASSIFICATION_PARTS,
    MODEL_SCHEMA_VERSION,
    POSE_FEATURE_GROUPS,
    camera_matrix_for_frame,
    detect_markers,
    estimate_marker_poses,
    extract_pose_observation,
    feature_indices_for_mask,
    load_camera_calibration,
    observable_part_count,
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
    models = bundle.get("model")
    if not isinstance(models, dict) or not models:
        raise ValueError(
            "Model bundle has no visibility-aware classifiers. Retrain the model."
        )
    if any(tuple(model.classes_) != classes for model in models.values()):
        raise ValueError(
            "Model classes do not match the saved class metadata. Retrain the model."
        )
    return models, classes


def predict_letter(model, features, group_mask=None):
    if group_mask is None:
        group_mask = (True,) * len(POSE_FEATURE_GROUPS)
    group_mask = tuple(group_mask)
    part_count = observable_part_count(group_mask)
    if part_count < MIN_CLASSIFICATION_PARTS:
        return None, 0.0
    mask_model = model.get(group_mask)
    if mask_model is None:
        return None, 0.0
    feature_indices = feature_indices_for_mask(group_mask)
    observed_features = np.asarray(features, dtype=np.float64)[list(feature_indices)]
    if not np.isfinite(observed_features).all():
        return None, 0.0

    probabilities = mask_model.predict_proba(observed_features.reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    evidence_factor = 0.56 + 0.11 * part_count
    confidence = float(probabilities[best_index]) * evidence_factor
    label = mask_model.classes_[best_index]
    return (label if confidence >= MIN_CONFIDENCE else None), confidence


def process_camera():
    model, classes = load_model()
    calibration = load_camera_calibration(CALIBRATION_FILE)
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}.")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoother = PredictionSmoother()
    previous_group_mask = None
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
            features, group_mask = extract_pose_observation(poses)
            visible_markers = sorted(poses)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if previous_group_mask is not None and group_mask != previous_group_mask:
                smoother.reset()
            previous_group_mask = group_mask

            if observable_part_count(group_mask) < MIN_CLASSIFICATION_PARTS:
                smoother.reset()
                text = "Unknown: insufficient visible pose"
                colour = (0, 0, 255)
            elif group_mask not in model:
                smoother.reset()
                text = "Unknown: visibility pattern not trained"
                colour = (0, 0, 255)
            else:
                label, confidence = predict_letter(model, features, group_mask)
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
            visible_text = (
                f"Valid marker poses: {','.join(map(str, visible_markers)) or 'none'}"
                f" | Features: {', '.join(name for name, visible in zip(POSE_FEATURE_GROUPS, group_mask) if visible) or 'none'}"
            )
            cv2.putText(
                frame,
                visible_text,
                (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )
            cv2.imshow("Robot A-Z Pose Classifier", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_camera()
