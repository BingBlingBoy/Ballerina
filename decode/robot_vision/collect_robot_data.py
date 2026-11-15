import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .robot_pose_features import (
        CALIBRATION_FILE,
        LETTER_DATASET_COLUMNS,
        LETTER_LABELS,
        POSE_FEATURE_GROUPS,
        camera_matrix_for_frame,
        detect_markers,
        estimate_marker_poses,
        extract_pose_observation,
        load_camera_calibration,
        observable_part_count,
    )
else:
    from robot_pose_features import (
        CALIBRATION_FILE,
        LETTER_DATASET_COLUMNS,
        LETTER_LABELS,
        POSE_FEATURE_GROUPS,
        camera_matrix_for_frame,
        detect_markers,
        estimate_marker_poses,
        extract_pose_observation,
        load_camera_calibration,
        observable_part_count,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "robot_pose_letter_dataset.csv"
CAMERA_INDEX = 0
BURST_SAMPLES = 20
SAMPLE_INTERVAL_SECONDS = 0.1
BURST_TIMEOUT_SECONDS = 15.0


def label_from_key(key):
    if ord("A") <= key <= ord("Z"):
        key += ord("a") - ord("A")
    if ord("a") <= key <= ord("z"):
        label = chr(key).upper()
        return label if label in LETTER_LABELS else None
    return None


def ensure_dataset():
    if not CSV_FILE.exists():
        with CSV_FILE.open("w", newline="", encoding="utf-8") as dataset:
            csv.writer(dataset).writerow(LETTER_DATASET_COLUMNS)
        return

    with CSV_FILE.open("r", newline="", encoding="utf-8") as dataset:
        header = next(csv.reader(dataset), None)
    if tuple(header or ()) != LETTER_DATASET_COLUMNS:
        raise RuntimeError(
            f"{CSV_FILE} has an incompatible legacy schema. Move it aside or "
            "start a new dataset before recording; existing rows were not changed."
        )


def write_sample(
    writer,
    session_id,
    capture_id,
    label,
    visible_markers,
    features,
    group_mask,
):
    writer.writerow(
        [
            session_id,
            capture_id,
            datetime.now(timezone.utc).isoformat(),
            label,
            ",".join(str(marker_id) for marker_id in sorted(visible_markers)),
            *[int(visible) for visible in group_mask],
            *[float(value) for value in features],
        ]
    )


def record_burst(capture, calibration, session_id, label):
    capture_id = uuid.uuid4().hex
    saved = 0
    deadline = time.monotonic() + BURST_TIMEOUT_SECONDS
    next_sample_at = time.monotonic()
    consecutive_read_failures = 0

    with CSV_FILE.open("a", newline="", encoding="utf-8") as dataset:
        writer = csv.writer(dataset)
        while saved < BURST_SAMPLES and time.monotonic() < deadline:
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
            poses = estimate_marker_poses(
                corners,
                ids,
                camera_matrix,
                distortion,
            )
            features, group_mask = extract_pose_observation(poses)
            visible_markers = sorted(poses)
            part_count = observable_part_count(group_mask)
            enough_observations = part_count >= 2

            display = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
            status = (
                f"Pose {label}: {saved}/{BURST_SAMPLES} | parts {part_count}/4"
                if enough_observations
                else (
                    "Need 2 body parts | valid markers: "
                    f"{','.join(map(str, visible_markers)) or 'none'}"
                )
            )
            cv2.putText(
                display,
                status,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            visible_text = (
                f"Markers: {','.join(map(str, visible_markers)) or 'none'} | "
                f"Features: {', '.join(name for name, visible in zip(POSE_FEATURE_GROUPS, group_mask) if visible) or 'none'}"
            )
            cv2.putText(
                display,
                visible_text,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                display,
                "Esc: quit",
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Robot Motion Data Recorder", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return False

            now = time.monotonic()
            if enough_observations and now >= next_sample_at:
                write_sample(
                    writer,
                    session_id,
                    capture_id,
                    label,
                    visible_markers,
                    features,
                    group_mask,
                )
                saved += 1
                next_sample_at = now + SAMPLE_INTERVAL_SECONDS

    if saved != BURST_SAMPLES:
        if saved:
            print(
                f"Captured {saved}/{BURST_SAMPLES} samples for pose {label}; "
                "those samples were retained."
            )
        else:
            print(
                f"No samples saved for pose {label}; make at least two body "
                "parts observable and try again."
            )
    else:
        print(f"Saved {saved} samples for pose {label} (capture {capture_id[:8]}).")
    return True


def process_camera():
    ensure_dataset()
    calibration = load_camera_calibration(CALIBRATION_FILE)
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}.")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    session_id = uuid.uuid4().hex
    consecutive_read_failures = 0
    print("Hold a settled pose and press its letter A-Z.")
    print("The letter labels the pose; Q records the Q pose like any other letter.")
    print(
        "Each capture records up to 20 samples at about 10 Hz when at least "
        "two body parts are observable."
    )
    print(
        "Record complete poses and separate, intentionally occluded captures "
        "across multiple sessions; unsupported visibility patterns are skipped "
        "during training."
    )
    print("Press Esc to quit.")

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
            _, group_mask = extract_pose_observation(poses)
            visible_markers = sorted(poses)
            part_count = observable_part_count(group_mask)
            enough_observations = part_count >= 2

            display = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
            status = (
                f"Ready: {part_count}/4 body parts"
                if enough_observations
                else "Need at least 2 observable body parts"
            )
            status += (
                f" | valid markers: {','.join(map(str, visible_markers)) or 'none'}"
            )
            cv2.putText(
                display,
                status,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0) if enough_observations else (0, 0, 255),
                2,
            )
            cv2.putText(
                display,
                (
                    "Features: "
                    + (
                        ", ".join(
                            name
                            for name, visible in zip(POSE_FEATURE_GROUPS, group_mask)
                            if visible
                        )
                        or "none"
                    )
                ),
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                display,
                "Move to pose, settle, press A-Z; Esc quits",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Robot Motion Data Recorder", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            label = label_from_key(key)
            if label is not None:
                if not record_burst(
                    capture,
                    calibration,
                    session_id,
                    label,
                ):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_camera()
