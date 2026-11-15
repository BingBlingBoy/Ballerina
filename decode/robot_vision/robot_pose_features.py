from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np


MARKER_IDS = {
    "head": 0,
    "left_hand": 3,
    "right_hand": 4,
    "base": 5,
}
MARKER_SIGNAL_IDS = (1, 2)
MARKER_SIZE_METERS = 0.04
CALIBRATION_FILE = Path(__file__).with_name("camera_calibration.npz")
MAX_REPROJECTION_ERROR_PX = 3.0
MODEL_SCHEMA_VERSION = 7
LETTER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REPEAT_LABEL = "~"
SPACE_LABEL = "SPACE"
POSE_CLASS_LABELS = (*LETTER_LABELS, SPACE_LABEL, REPEAT_LABEL)

TARGET_LABELS = {
    "left_arm": ("down", "neutral", "up"),
    "right_arm": ("down", "neutral", "up"),
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
POSE_FEATURE_COLUMNS = tuple(f"feature_{index}" for index in range(18))
POSE_FEATURE_GROUPS = {
    "left_arm": tuple(range(0, 6)),
    "right_arm": tuple(range(6, 12)),
    "head": tuple(range(12, 18)),
}
POSE_MASK_COLUMNS = {
    name: f"observed_{name}" for name in POSE_FEATURE_GROUPS
}
LED_FEATURE_COLUMNS = ("led_red_on", "led_blue_on")
LED_VALID_COLUMN = "led_state_valid"
LED_DETECTOR_COLUMN = "led_detector_version"
LED_DETECTOR_VERSION = "head-marker-camera-profile-v3"
MARKER_SIGNAL_COLUMN = "marker_signal"
MARKER_SIGNAL_FEATURE_COLUMNS = ("marker_signal_id1", "marker_signal_id2")
MIN_CLASSIFICATION_PARTS = 2
LETTER_DATASET_COLUMNS = (
    "session_id",
    "capture_id",
    "timestamp_utc",
    "label",
    "visible_marker_ids",
    *POSE_MASK_COLUMNS.values(),
    MARKER_SIGNAL_COLUMN,
    *LED_FEATURE_COLUMNS,
    LED_VALID_COLUMN,
    LED_DETECTOR_COLUMN,
    *POSE_FEATURE_COLUMNS,
)

_REQUIRED_IDS = frozenset(MARKER_IDS.values())
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
        for marker_id in _REQUIRED_IDS
    }

    for corner, raw_marker_id in zip(corners, flat_ids):
        marker_id = int(raw_marker_id)
        if marker_id not in _REQUIRED_IDS:
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


def marker_signal_from_ids(ids):
    """Return ID 1 or 2 when exactly one signal marker type is detected."""
    if ids is None:
        return None
    detected_ids = set(np.asarray(ids).reshape(-1).astype(int))
    present_signal_ids = detected_ids.intersection(MARKER_SIGNAL_IDS)
    if len(present_signal_ids) != 1:
        return None
    return present_signal_ids.pop()


def _rotation_vector(rotation):
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return vector.reshape(3)


def extract_features(poses):
    """Return hand and head poses relative to the stationary base marker."""
    features = {target: None for target in TARGET_LABELS}
    base = poses.get(MARKER_IDS["base"])
    head = poses.get(MARKER_IDS["head"])
    left_hand = poses.get(MARKER_IDS["left_hand"])
    right_hand = poses.get(MARKER_IDS["right_hand"])

    if base is not None:
        base_rotation, base_translation = base
        for target, marker_pose in (
            ("left_arm", left_hand),
            ("right_arm", right_hand),
            ("head", head),
        ):
            if marker_pose is None:
                continue
            marker_rotation, marker_translation = marker_pose
            relative_position = base_rotation.T @ (
                marker_translation - base_translation
            )
            relative_rotation = base_rotation.T @ marker_rotation
            features[target] = np.concatenate(
                (relative_position, _rotation_vector(relative_rotation))
            ).astype(np.float32)

    for target, vector in features.items():
        if vector is not None and not np.isfinite(vector).all():
            features[target] = np.where(
                np.isfinite(vector), vector, np.nan
            ).astype(np.float32)

    return features


def extract_pose_observation(poses):
    """Return a NaN-masked pose vector and the observable feature-group mask."""
    joint_features = extract_features(poses)
    pose_features = np.full(len(POSE_FEATURE_COLUMNS), np.nan, dtype=np.float32)
    available = {name: False for name in POSE_FEATURE_GROUPS}

    for target in POSE_FEATURE_GROUPS:
        vector = joint_features[target]
        if vector is not None and np.isfinite(vector).all():
            indices = POSE_FEATURE_GROUPS[target]
            pose_features[list(indices)] = vector
            available[target] = True

    group_mask = tuple(available[name] for name in POSE_FEATURE_GROUPS)
    return pose_features, group_mask


def feature_indices_for_mask(group_mask):
    if len(group_mask) != len(POSE_FEATURE_GROUPS):
        raise ValueError("Feature-group mask has an unexpected size.")
    return tuple(
        index
        for enabled, group_indices in zip(group_mask, POSE_FEATURE_GROUPS.values())
        if enabled
        for index in group_indices
    )


def observable_part_count(group_mask):
    if len(group_mask) != len(POSE_FEATURE_GROUPS):
        raise ValueError("Feature-group mask has an unexpected size.")
    return sum(bool(observed) for observed in group_mask)


def feature_group_mask_for_marker_ids(marker_ids):
    visible_ids = {int(marker_id) for marker_id in marker_ids}
    base = MARKER_IDS["base"] in visible_ids
    return (
        base and MARKER_IDS["left_hand"] in visible_ids,
        base and MARKER_IDS["right_hand"] in visible_ids,
        base and MARKER_IDS["head"] in visible_ids,
    )


def _supported_group_masks():
    marker_ids = tuple(MARKER_IDS.values())
    masks = set()
    for marker_bits in range(1 << len(marker_ids)):
        visible_ids = {
            marker_id
            for index, marker_id in enumerate(marker_ids)
            if marker_bits & (1 << index)
        }
        mask = feature_group_mask_for_marker_ids(visible_ids)
        if observable_part_count(mask) >= MIN_CLASSIFICATION_PARTS:
            masks.add(mask)
    return tuple(sorted(masks))


SUPPORTED_GROUP_MASKS = _supported_group_masks()


def extract_pose_features(poses):
    """Return a complete hand/head vector, or None when a feature is missing."""
    pose_features, group_mask = extract_pose_observation(poses)
    if not all(group_mask):
        return None

    if pose_features.shape != (len(POSE_FEATURE_COLUMNS),):
        raise RuntimeError("Hand/head feature schema has an unexpected size.")
    if not np.isfinite(pose_features).all():
        return None
    return pose_features
