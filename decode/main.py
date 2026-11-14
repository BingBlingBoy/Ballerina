import cv2
import cv2.aruco as aruco
import numpy as np
import joblib

# =====================================
# CONFIG
# =====================================

MARKER_IDS = {
    "head": 0,
    "left_shoulder": 1,
    "right_shoulder": 2,
    "left_hand": 3,
    "right_hand": 4
}

MARKER_SIZE = 0.04

# =====================================
# DEFAULT CAMERA MODEL
# =====================================

CAMERA_MATRIX = np.array([
    [1000.0, 0.0, 640.0],
    [0.0, 1000.0, 360.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((5, 1))

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load(
    "models/letter_classifier.joblib"
)

print("Model loaded")

# =====================================
# ARUCO
# =====================================

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

# =====================================
# POSE ESTIMATION
# =====================================

def estimate_marker_poses(corners, ids):

    poses = {}

    if ids is None:
        return poses

    h = MARKER_SIZE / 2

    object_points = np.array([
        [-h, h, 0],
        [h, h, 0],
        [h, -h, 0],
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

# =====================================
# FEATURES
# =====================================

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

# =====================================
# LIVE INFERENCE
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam"
    )

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

    if ids is not None:

        aruco.drawDetectedMarkers(
            frame,
            corners,
            ids
        )

    features = extract_features(
        poses
    )

    prediction_text = "Missing markers"

    colour = (0, 0, 255)

    if features is not None:

        sample = features.reshape(1, -1)

        prediction = model.predict(
            sample
        )[0]

        confidence = max(
            model.predict_proba(
                sample
            )[0]
        )

        prediction_text = (
            f"{prediction} "
            f"({confidence:.1%})"
        )

        colour = (0, 255, 0)

    cv2.putText(
        frame,
        prediction_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        colour,
        3
    )

    cv2.imshow(
        "Letter Classifier",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()