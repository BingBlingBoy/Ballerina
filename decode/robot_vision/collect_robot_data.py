import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import os

MARKER_IDS = {
    "head": 0,
    "left_shoulder": 1,
    "right_shoulder": 2,
    "left_hand": 3,
    "right_hand": 4
}

MARKER_SIZE = 0.04
CSV_FILE = "robot_pose_dataset.csv"
BURST_SAMPLES = 100


# Camera Configuration
CAMERA_MATRIX = np.array([
    [1000.0,    0.0, 640.0],
    [   0.0, 1000.0, 360.0],
    [   0.0,    0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)

# Creation of dataset file
if not os.path.exists(CSV_FILE):

    with open(CSV_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        header = ["label"]

        for i in range(30):
            header.append(f"feature_{i}")

        writer.writerow(header)


# Aruco stuff
dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

if hasattr(aruco, "ArucoDetector"):

    detector = aruco.ArucoDetector(
        dictionary,
        aruco.DetectorParameters()
    )

    def detect(gray):
        return detector.detectMarkers(gray)

else:

    parameters = aruco.DetectorParameters_create()

    def detect(gray):
        return aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters
        )

def save_sample(label, features):

    with open(CSV_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        row = [label]
        row.extend(features.tolist())

        writer.writerow(row)

def estimate_marker_poses(corners, ids):

    poses = {}

    if ids is None:
        return poses

    h = MARKER_SIZE / 2

    object_points = np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0]
    ], dtype=np.float32)

    for corner, marker_id in zip(
        corners,
        ids.flatten()
    ):

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            corner.reshape(4, 2).astype(np.float32),
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success:

            R, _ = cv2.Rodrigues(rvec)

            poses[int(marker_id)] = (
                R,
                tvec.reshape(3)
            )

    return poses

def extract_features(poses):

    reference_id = MARKER_IDS["head"]

    if reference_id not in poses:
        return None

    R_ref, t_ref = poses[reference_id]

    features = []

    for marker_id in MARKER_IDS.values():

        if marker_id not in poses:
            return None

        R, t = poses[marker_id]

        R_relative = R_ref.T @ R

        t_relative = R_ref.T @ (
            t - t_ref
        )

        rvec, _ = cv2.Rodrigues(
            R_relative
        )

        features.extend(
            t_relative.tolist()
        )

        features.extend(
            rvec.flatten().tolist()
        )

    return np.array(
        features,
        dtype=np.float32
    )

def record_burst(cap, label):

    print(
        f"\nRecording {BURST_SAMPLES} "
        f"samples for '{label}'"
    )

    saved = 0

    while saved < BURST_SAMPLES:

        ret, frame = cap.read()

        if not ret:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, _ = detect(gray)

        poses = estimate_marker_poses(
            corners,
            ids
        )

        features = extract_features(
            poses
        )

        if features is not None:

            save_sample(
                label,
                features
            )

            saved += 1

        display = frame.copy()

        if ids is not None:

            aruco.drawDetectedMarkers(
                display,
                corners,
                ids
            )

        cv2.putText(
            display,
            f"Recording {label}: "
            f"{saved}/{BURST_SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Robot Dataset Recorder",
            display
        )

        cv2.waitKey(1)

    print(
        f"Finished recording "
        f"{saved} samples for {label}"
    )

# loop
def process_camera():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera"
        )

    print()
    print("==============================")
    print("ROBOT DATASET COLLECTION")
    print("==============================")
    print()
    print("Markers required:")
    print("0 = head")
    print("1 = left shoulder")
    print("2 = right shoulder")
    print("3 = left hand")
    print("4 = right hand")
    print()
    print("Press A-Z to record")
    print(
        f"{BURST_SAMPLES} samples "
        "will be captured"
    )
    print("Press q to quit")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, _ = detect(gray)

        poses = estimate_marker_poses(
            corners,
            ids
        )

        features = extract_features(
            poses
        )

        if ids is not None:

            aruco.drawDetectedMarkers(
                frame,
                corners,
                ids
            )

        ready = (
            features is not None
        )

        cv2.putText(
            frame,
            "READY"
            if ready
            else "SHOW ALL 5 MARKERS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0)
            if ready
            else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "Press A-Z to record",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "Robot Dataset Recorder",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if ready:

            if ord("A") <= key <= ord("Z"):

                record_burst(
                    cap,
                    chr(key)
                )

            elif ord("a") <= key <= ord("z"):

                record_burst(
                    cap,
                    chr(key).upper()
                )

    cap.release()
    cv2.destroyAllWindows()

# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    process_camera()