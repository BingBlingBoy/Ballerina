from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


MARKER_IDS = {
    "head": 0,
    "left_shoulder": 1,
    "right_shoulder": 2,
    "left_hand": 3,
    "right_hand": 4,
    "base": 5,
}
MARKER_SIZE_METERS = 0.04
CALIBRATION_FILE = Path(__file__).with_name("camera_calibration.npz")
MAX_REPROJECTION_ERROR_PX = 3.0
MODEL_SCHEMA_VERSION = 2
LETTER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

TARGET_LABELS = {
    "left_arm": ("down", "neutral", "up"),
    "right_arm": ("down", "neutral", "up"),
    "torso": ("left", "neutral", "right"),
    "head": ("down", "neutral", "up"),
}
FEATURE_COLUMNS = {
    target: tuple(f"feature_{index}" for index in range(6))
    for target in TARGET_LABELS
}
DATASET_COLUMNS = (
    "session_id",
    "capture_id",
    "timestamp_utc",
    "target",
    "label",
    *(f"feature_{index}" for index in range(6)),
)
POSE_FEATURE_COLUMNS = tuple(f"feature_{index}" for index in range(24))
LETTER_DATASET_COLUMNS = (
    "session_id",
    "capture_id",
    "timestamp_utc",
    "label",
    *POSE_FEATURE_COLUMNS,
)

_REQUIRED_IDS = {
    "head": MARKER_IDS["head"],
    "left_shoulder": MARKER_IDS["left_shoulder"],
    "right_shoulder": MARKER_IDS["right_shoulder"],
    "left_hand": MARKER_IDS["left_hand"],
    "right_hand": MARKER_IDS["right_hand"],
    "base": MARKER_IDS["base"],
}
_HALF_MARKER_SIZE = MARKER_SIZE_METERS / 2.0
_MARKER_OBJECT_POINTS = np.array(
    [
        [-_HALF_MARKER_SIZE, _HALF_MARKER_SIZE, 0],
        [_HALF_MARKER_SIZE, _HALF_MARKER_SIZE, 0],
        [_HALF_MARKER_SIZE, -_HALF_MARKER_SIZE, 0],
        [-_HALF_MARKER_SIZE, -_HALF_MARKER_SIZE, 0],
    ],
    dtype=np.float32,
)

_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
if hasattr(aruco, "DetectorParameters"):
    _detector_parameters = aruco.DetectorParameters()
else:
    _detector_parameters = aruco.DetectorParameters_create()
if hasattr(_detector_parameters, "cornerRefinementMethod"):
    _detector_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

if hasattr(aruco, "ArucoDetector"):
    _detector = aruco.ArucoDetector(_dictionary, _detector_parameters)

    def detect_markers(gray):
        return _detector.detectMarkers(gray)
else:
    _legacy_parameters = aruco.DetectorParameters_create()
    _legacy_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    def detect_markers(gray):
        return aruco.detectMarkers(
            gray,
            _dictionary,
            parameters=_legacy_parameters,
        )


def load_camera_calibration(path=CALIBRATION_FILE):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Camera calibration not found at {path}. "
            "Run robot_vision/calibrate_camera.py first."
        )

    with np.load(path, allow_pickle=False) as data:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
        image_width = int(data["image_width"])
        image_height = int(data["image_height"])

    if camera_matrix.shape != (3, 3) or distortion.size < 4:
        raise ValueError(f"Invalid camera calibration data in {path}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid calibration image dimensions in {path}")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise ValueError(f"Non-finite camera calibration data in {path}")

    return camera_matrix, distortion.reshape(-1, 1), image_width, image_height


def camera_matrix_for_frame(calibration, frame):
    camera_matrix, distortion, calibrated_width, calibrated_height = calibration
    frame_height, frame_width = frame.shape[:2]

    calibrated_aspect = calibrated_width / calibrated_height
    frame_aspect = frame_width / frame_height
    if abs(calibrated_aspect - frame_aspect) > 1e-3:
        raise ValueError(
            "Camera frame aspect ratio differs from calibration; "
            "recalibrate at this aspect ratio."
        )

    scaled = camera_matrix.copy()
    scaled[0, :] *= frame_width / calibrated_width
    scaled[1, :] *= frame_height / calibrated_height
    return scaled, distortion


def estimate_marker_poses(corners, ids, camera_matrix, distortion):
    poses = {}
    if ids is None:
        return poses

    flat_ids = np.asarray(ids).reshape(-1)
    known_id_counts = {
        marker_id: int(np.count_nonzero(flat_ids == marker_id))
        for marker_id in _REQUIRED_IDS.values()
    }

    for corner, raw_marker_id in zip(corners, flat_ids):
        marker_id = int(raw_marker_id)
        if marker_id not in _REQUIRED_IDS.values():
            continue
        if known_id_counts[marker_id] != 1:
            continue

        image_points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(image_points).all():
            continue

        success, rvec, tvec = cv2.solvePnP(
            _MARKER_OBJECT_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
            continue

        projected, _ = cv2.projectPoints(
            _MARKER_OBJECT_POINTS,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        errors = projected.reshape(4, 2) - image_points
        reprojection_error = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
        if not np.isfinite(reprojection_error):
            continue
        if reprojection_error > MAX_REPROJECTION_ERROR_PX:
            continue

        rotation, _ = cv2.Rodrigues(rvec)
        poses[marker_id] = (rotation, tvec.reshape(3))

    return poses


def _rotation_vector(rotation):
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return vector.reshape(3)


def _mean_rotation(rotations):
    average = np.mean(np.stack(rotations), axis=0)
    left, _, right = np.linalg.svd(average)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def extract_features(poses):
    """Return one six-value vector per observable joint, or None if unavailable."""
    features = {target: None for target in TARGET_LABELS}
    base = poses.get(MARKER_IDS["base"])
    if base is None:
        return features

    base_rotation, base_translation = base
    left_shoulder = poses.get(MARKER_IDS["left_shoulder"])
    right_shoulder = poses.get(MARKER_IDS["right_shoulder"])
    head = poses.get(MARKER_IDS["head"])
    left_hand = poses.get(MARKER_IDS["left_hand"])
    right_hand = poses.get(MARKER_IDS["right_hand"])

    if left_shoulder is not None and left_hand is not None:
        shoulder_rotation, shoulder_translation = left_shoulder
        hand_rotation, hand_translation = left_hand
        relative_position = shoulder_rotation.T @ (
            hand_translation - shoulder_translation
        )
        relative_rotation = shoulder_rotation.T @ hand_rotation
        features["left_arm"] = np.concatenate(
            (relative_position, _rotation_vector(relative_rotation))
        ).astype(np.float32)

    if right_shoulder is not None and right_hand is not None:
        shoulder_rotation, shoulder_translation = right_shoulder
        hand_rotation, hand_translation = right_hand
        relative_position = shoulder_rotation.T @ (
            hand_translation - shoulder_translation
        )
        relative_rotation = shoulder_rotation.T @ hand_rotation
        features["right_arm"] = np.concatenate(
            (relative_position, _rotation_vector(relative_rotation))
        ).astype(np.float32)

    if left_shoulder is None or right_shoulder is None:
        return features

    left_rotation, left_translation = left_shoulder
    right_rotation, right_translation = right_shoulder
    torso_rotation = _mean_rotation([left_rotation, right_rotation])
    shoulder_vector = base_rotation.T @ (
        right_translation - left_translation
    )
    torso_rotation_in_base = base_rotation.T @ torso_rotation
    features["torso"] = np.concatenate(
        (shoulder_vector, _rotation_vector(torso_rotation_in_base))
    ).astype(np.float32)

    if head is not None:
        head_rotation, head_translation = head
        shoulder_center = (left_translation + right_translation) / 2.0
        head_position_in_torso = torso_rotation.T @ (
            head_translation - shoulder_center
        )
        head_rotation_in_torso = torso_rotation.T @ head_rotation
        features["head"] = np.concatenate(
            (
                head_position_in_torso,
                _rotation_vector(head_rotation_in_torso),
            )
        ).astype(np.float32)

    for vector in features.values():
        if vector is not None and not np.isfinite(vector).all():
            return {target: None for target in TARGET_LABELS}

    return features


def extract_pose_features(poses):
    """Return one whole-body vector only when every required marker is visible."""
    joint_features = extract_features(poses)
    if any(vector is None for vector in joint_features.values()):
        return None

    pose_features = np.concatenate(
        [joint_features[target] for target in TARGET_LABELS]
    ).astype(np.float32)
    if pose_features.shape != (len(POSE_FEATURE_COLUMNS),):
        raise RuntimeError("Whole-body feature schema has an unexpected size.")
    if not np.isfinite(pose_features).all():
        return None
    return pose_features
