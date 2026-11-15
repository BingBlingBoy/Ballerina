import argparse
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from robot_vision.robot_pose_features import (
    CALIBRATION_FILE,
    LETTER_LABELS,
    LED_FEATURE_COLUMNS,
    MARKER_IDS,
    MARKER_SIGNAL_FEATURE_COLUMNS,
    MARKER_SIGNAL_IDS,
    MIN_CLASSIFICATION_PARTS,
    MODEL_SCHEMA_VERSION,
    POSE_CLASS_LABELS,
    POSE_FEATURE_GROUPS,
    POSE_FEATURE_COLUMNS,
    REPEAT_LABEL,
    SPACE_LABEL,
    camera_matrix_for_frame,
    detect_markers,
    estimate_marker_poses,
    extract_pose_observation,
    feature_indices_for_mask,
    load_camera_calibration,
    marker_signal_from_ids,
    observable_part_count,
)
from robot_vision.led_features import (
    detect_led_states,
    led_state_label,
    load_led_calibration,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_DIR / "models" / "pose_letter_classifier.joblib"
CAMERA_INDEX = 0
MIN_CONFIDENCE = 0.50
SMOOTHING_WINDOW = 5
MIN_STABLE_VOTES = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the robot pose classifier with a selected camera."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Camera device index (default: {CAMERA_INDEX}).",
    )
    return parser.parse_args()


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


class LetterWordBuffer:
    def __init__(self):
        self.letters = []
        self.last_letter = None
        self.last_finalized = ""
        self.completed_words = []

    def accept(self, label):
        if label is None:
            self.last_letter = None
            return None
        if label == SPACE_LABEL:
            if self.last_letter == SPACE_LABEL:
                return None
            self.last_letter = SPACE_LABEL
            return self._finish_word("")
        if label == REPEAT_LABEL:
            if self.last_letter != REPEAT_LABEL:
                self.repeat_last_letter()
                self.last_letter = REPEAT_LABEL
            return None
        if label != self.last_letter:
            self.letters.append(label)
            self.last_letter = label
        return None

    def repeat_last_letter(self):
        if not self.letters or self.letters[-1] == REPEAT_LABEL:
            return False
        self.letters.append(REPEAT_LABEL)
        return True

    def _finish_word(self, terminator):
        if not self.letters:
            return None
        self.last_finalized = "".join(self.letters) + terminator
        self.letters.clear()
        self.completed_words.append(self.last_finalized)
        return self.last_finalized

    def finish(self):
        finalized = self._finish_word("!")
        self.last_letter = None
        return finalized

    def finish_word(self):
        finalized = self._finish_word("")
        self.last_letter = None
        return finalized

    @property
    def output_text(self):
        return " ".join(self.completed_words)

    @property
    def current_word(self):
        return "".join(self.letters)


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
    if not classes or any(label not in POSE_CLASS_LABELS for label in classes):
        raise ValueError(
            "Model bundle has no valid pose classes. Retrain the model."
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


def _classifier_options(model, group_mask, led_states, marker_signal):
    led_values = None
    if led_states is not None:
        candidate_led_values = np.asarray(led_states, dtype=np.float64)
        if (
            candidate_led_values.shape == (len(LED_FEATURE_COLUMNS),)
            and np.isin(candidate_led_values, (0, 1)).all()
        ):
            led_values = candidate_led_values
    valid_marker_signal = (
        marker_signal if marker_signal in MARKER_SIGNAL_IDS else None
    )
    requested_options = (
        (led_values is not None, valid_marker_signal is not None),
        (False, valid_marker_signal is not None),
        (led_values is not None, False),
        (False, False),
    )
    seen = set()
    for use_leds, use_marker_signal in requested_options:
        key = (group_mask, use_leds, use_marker_signal)
        if key in seen:
            continue
        seen.add(key)
        if key in model:
            yield key, led_values, valid_marker_signal


def _has_model_for_observation(model, group_mask, led_states, marker_signal):
    return next(
        _classifier_options(model, group_mask, led_states, marker_signal),
        None,
    ) is not None


def predict_letter(
    model,
    features,
    group_mask=None,
    led_states=None,
    marker_signal=None,
):
    if group_mask is None:
        group_mask = (True,) * len(POSE_FEATURE_GROUPS)
    group_mask = tuple(group_mask)
    part_count = observable_part_count(group_mask)
    if part_count < MIN_CLASSIFICATION_PARTS:
        return None, 0.0
    selected = next(
        _classifier_options(model, group_mask, led_states, marker_signal),
        None,
    )
    if selected is None:
        return None, 0.0
    model_key, led_values, valid_marker_signal = selected
    _, use_leds, use_marker_signal = model_key
    mask_model = model[model_key]
    feature_indices = feature_indices_for_mask(group_mask)
    observed_features = np.asarray(features, dtype=np.float64)
    observed_features = np.concatenate(
        (
            observed_features,
            led_values
            if use_leds
            else np.zeros(len(LED_FEATURE_COLUMNS), dtype=np.float64),
            np.array(
                [
                    valid_marker_signal == 1,
                    valid_marker_signal == 2,
                ],
                dtype=np.float64,
            )
            if use_marker_signal
            else np.zeros(len(MARKER_SIGNAL_FEATURE_COLUMNS), dtype=np.float64),
        )
    )
    if use_leds:
        feature_indices = (
            *feature_indices,
            *range(
                len(POSE_FEATURE_COLUMNS),
                len(POSE_FEATURE_COLUMNS) + len(LED_FEATURE_COLUMNS),
            ),
        )
    if use_marker_signal:
        feature_start = len(POSE_FEATURE_COLUMNS) + len(LED_FEATURE_COLUMNS)
        feature_indices = (
            *feature_indices,
            *range(
                feature_start,
                feature_start + len(MARKER_SIGNAL_FEATURE_COLUMNS),
            ),
        )
    observed_features = observed_features[list(feature_indices)]
    if not np.isfinite(observed_features).all():
        return None, 0.0

    probabilities = mask_model.predict_proba(observed_features.reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    evidence_factor = 0.56 + 0.11 * part_count
    confidence = float(probabilities[best_index]) * evidence_factor
    label = mask_model.classes_[best_index]
    return label, confidence


def process_camera(camera_index=CAMERA_INDEX):
    model, classes = load_model()
    calibration = load_camera_calibration(CALIBRATION_FILE)
    led_calibration = load_led_calibration()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    smoother = PredictionSmoother()
    word_buffer = LetterWordBuffer()
    previous_group_mask = None
    previous_marker_signal = None
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
            marker_signal = marker_signal_from_ids(ids)
            poses = estimate_marker_poses(corners, ids, camera_matrix, distortion)
            features, group_mask = extract_pose_observation(poses)
            led_states = detect_led_states(
                frame,
                poses.get(MARKER_IDS["head"]),
                camera_matrix,
                distortion,
                led_calibration,
            )
            visible_markers = sorted(poses)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if (
                previous_group_mask is not None
                and group_mask != previous_group_mask
            ) or (
                previous_marker_signal is not None
                and marker_signal != previous_marker_signal
            ):
                smoother.reset()
                word_buffer.accept(None)
            previous_group_mask = group_mask
            previous_marker_signal = marker_signal

            if observable_part_count(group_mask) < MIN_CLASSIFICATION_PARTS:
                smoother.reset()
                word_buffer.accept(None)
                text = "Unknown: insufficient visible pose"
                colour = (0, 0, 255)
            elif not _has_model_for_observation(
                model,
                group_mask,
                led_states,
                marker_signal,
            ):
                smoother.reset()
                word_buffer.accept(None)
                text = "Unknown: visibility pattern not trained"
                colour = (0, 0, 255)
            else:
                label, confidence = predict_letter(
                    model,
                    features,
                    group_mask,
                    led_states,
                    marker_signal,
                )
                stable_label = smoother.update(
                    label
                )
                finalized_word = word_buffer.accept(stable_label)
                if finalized_word is not None:
                    print(f"Word: {finalized_word}")
                text = (
                    f"{label} ({confidence:.0%})"
                    if confidence > MIN_CONFIDENCE
                    else f"{label} - Uncertain ({confidence:.0%})"
                )
                colour = (0, 255, 0) if confidence > MIN_CONFIDENCE else (0, 165, 255)

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
                f"Classes {','.join(classes)} | SPACE pose/Enter: new word | !: end message | Esc: quit",
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
            cv2.putText(
                frame,
                f"LED red: {led_state_label(led_states, 0)} | "
                f"blue: {led_state_label(led_states, 1)}",
                (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                (
                    f"Signal marker: ID {marker_signal}"
                    if marker_signal is not None
                    else "Signal marker: UNKNOWN"
                ),
                (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )
            cv2.imshow("Robot A-Z Pose Classifier", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (10, 13):
                finalized = word_buffer.finish_word()
                if finalized is not None:
                    print(f"Word: {finalized}")
            if key == ord("!"):
                finalized = word_buffer.finish()
                if finalized is not None:
                    print(f"Final message: {finalized}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_camera(parse_args().camera)
