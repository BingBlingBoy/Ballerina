import json
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .robot_pose_features import MARKER_IDS, MARKER_SIZE_METERS
else:
    from robot_pose_features import MARKER_IDS, MARKER_SIZE_METERS


LED_CALIBRATION_FILE = Path(__file__).with_name("led_marker_offsets.json")
LED_CALIBRATION_SCHEMA_VERSION = 3
MIN_MARKER_SIDE_PIXELS = 20.0
LED_ROI_RADIUS_MARKER_FRACTION = 0.25
LED_INNER_RADIUS_MARKER_FRACTION = 0.10
LED_BACKGROUND_INNER_RADIUS_MARKER_FRACTION = 0.15
LED_BACKGROUND_OUTER_RADIUS_MARKER_FRACTION = 0.23
MIN_LED_CONTRAST = 8.0
UNKNOWN_SCORE_MARGIN = 0.12


def load_led_calibration(path=LED_CALIBRATION_FILE):
    path = Path(path)
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)

    if calibration.get("schema_version") != LED_CALIBRATION_SCHEMA_VERSION:
        return None
    if calibration.get("marker_id") != MARKER_IDS["head"]:
        raise ValueError(f"LED calibration must use head marker {MARKER_IDS['head']}.")
    if not np.isclose(
        calibration.get("marker_size_meters"),
        MARKER_SIZE_METERS,
    ):
        raise ValueError("LED calibration marker size does not match the pose model.")

    points = calibration.get("points_meters")
    if not isinstance(points, dict):
        raise ValueError(f"Invalid LED positions in {path}")
    for color in ("red", "blue"):
        point = np.asarray(points.get(color), dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid {color} LED position in {path}")
        profiles = calibration.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError(f"Missing calibrated LED state samples in {path}")
        profile = profiles.get(color)
        if not isinstance(profile, dict):
            raise ValueError(f"Missing calibrated {color} LED state samples in {path}")
        off_sample = np.asarray(profile.get("off"), dtype=np.float64)
        on_samples = np.asarray(profile.get("on"), dtype=np.float64)
        if off_sample.shape != (3,) or not np.isfinite(off_sample).all():
            raise ValueError(f"Invalid {color} LED off sample in {path}")
        if (
            on_samples.ndim != 2
            or on_samples.shape[1] != 3
            or len(on_samples) < 1
            or not np.isfinite(on_samples).all()
        ):
            raise ValueError(f"Invalid {color} LED on samples in {path}")

    return calibration


def save_led_calibration(points_meters, profiles, path=LED_CALIBRATION_FILE):
    path = Path(path)
    normalized_points = {}
    for color in ("red", "blue"):
        point = np.asarray(points_meters[color], dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid {color} LED position.")
        normalized_points[color] = point.tolist()

    normalized_profiles = {}
    for color in ("red", "blue"):
        normalized_profiles[color] = {}
        off_sample = np.asarray(profiles[color]["off"], dtype=np.float64)
        on_samples = np.asarray(profiles[color]["on"], dtype=np.float64)
        if off_sample.shape != (3,) or not np.isfinite(off_sample).all():
            raise ValueError(f"Invalid {color} LED off sample.")
        if (
            on_samples.ndim != 2
            or on_samples.shape[1] != 3
            or len(on_samples) < 1
            or not np.isfinite(on_samples).all()
        ):
            raise ValueError(f"Invalid {color} LED on samples.")
        normalized_profiles[color]["off"] = off_sample.tolist()
        normalized_profiles[color]["on"] = on_samples.tolist()
        if np.linalg.norm(
            np.mean(on_samples, axis=0) - off_sample
        ) < MIN_LED_CONTRAST:
            raise ValueError(
                f"The {color} LED on/off samples are too similar. "
                "Check the LED state and capture the samples again."
            )

    calibration = {
        "schema_version": LED_CALIBRATION_SCHEMA_VERSION,
        "marker_id": MARKER_IDS["head"],
        "marker_size_meters": MARKER_SIZE_METERS,
        "points_meters": normalized_points,
        "profiles": normalized_profiles,
    }
    with path.open("w", encoding="utf-8") as calibration_file:
        json.dump(calibration, calibration_file, indent=2)
        calibration_file.write("\n")


def led_state_label(led_states, index):
    if led_states is None:
        return "UNKNOWN"
    return "ON" if led_states[index] else "OFF"


def marker_plane_point_from_click(image_point, marker_corners):
    half_size = MARKER_SIZE_METERS / 2.0
    object_points = np.array(
        [
            [-half_size, half_size],
            [half_size, half_size],
            [half_size, -half_size],
            [-half_size, -half_size],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    transform = cv2.getPerspectiveTransform(image_points, object_points)
    point = np.asarray(image_point, dtype=np.float32).reshape(1, 1, 2)
    marker_point = cv2.perspectiveTransform(point, transform).reshape(2)
    if not np.isfinite(marker_point).all():
        raise ValueError("Could not map LED click into the head marker plane.")
    return marker_point.astype(np.float64)


def _led_color_signal(frame, center, marker_side_pixels):
    height, width = frame.shape[:2]
    x, y = np.rint(center).astype(int)
    outer_radius = max(
        3,
        int(round(marker_side_pixels * LED_ROI_RADIUS_MARKER_FRACTION)),
    )
    if (
        x - outer_radius < 0
        or x + outer_radius >= width
        or y - outer_radius < 0
        or y + outer_radius >= height
    ):
        return None

    roi = frame[
        y - outer_radius : y + outer_radius + 1,
        x - outer_radius : x + outer_radius + 1,
    ]
    yy, xx = np.ogrid[-outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1]
    distances = np.sqrt(xx * xx + yy * yy)
    inner_radius = max(
        1,
        marker_side_pixels * LED_INNER_RADIUS_MARKER_FRACTION,
    )
    background_inner = (
        marker_side_pixels * LED_BACKGROUND_INNER_RADIUS_MARKER_FRACTION
    )
    background_outer = (
        marker_side_pixels * LED_BACKGROUND_OUTER_RADIUS_MARKER_FRACTION
    )
    inner_pixels = roi[distances <= inner_radius]
    background_pixels = roi[
        (distances >= background_inner) & (distances <= background_outer)
    ]
    if not len(inner_pixels) or not len(background_pixels):
        return None

    return (
        np.mean(inner_pixels, axis=0).astype(np.float64)
        - np.mean(background_pixels, axis=0).astype(np.float64)
    )


def led_signals_from_marker_corners(frame, marker_corners, points_meters):
    image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
    half_size = MARKER_SIZE_METERS / 2.0
    object_points = np.array(
        [
            [-half_size, half_size],
            [half_size, half_size],
            [half_size, -half_size],
            [-half_size, -half_size],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(object_points, image_points)
    marker_side_pixels = float(
        np.mean(
            [
                np.linalg.norm(
                    image_points[index] - image_points[(index + 1) % 4]
                )
                for index in range(4)
            ]
        )
    )
    if marker_side_pixels < MIN_MARKER_SIDE_PIXELS:
        return None

    marker_points = np.asarray(
        [[*points_meters[color]] for color in ("red", "blue")],
        dtype=np.float32,
    ).reshape(1, 2, 2)
    centers = cv2.perspectiveTransform(marker_points, transform).reshape(2, 2)
    signals = tuple(
        _led_color_signal(frame, center, marker_side_pixels)
        for center in centers
    )
    if any(signal is None for signal in signals):
        return None
    return signals


def _project_marker_points(points, rotation, translation, camera_matrix, distortion):
    rotation_vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return cv2.projectPoints(
        np.asarray(points, dtype=np.float64),
        rotation_vector,
        np.asarray(translation, dtype=np.float64).reshape(3, 1),
        camera_matrix,
        distortion,
    )[0].reshape(-1, 2)


def detect_led_states(
    frame,
    head_pose,
    camera_matrix,
    distortion,
    led_calibration,
):
    """Return camera-calibrated red/blue states, or None when uncertain."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("LED detection requires a three-channel BGR frame.")
    if (
        head_pose is None
        or camera_matrix is None
        or distortion is None
        or led_calibration is None
    ):
        return None

    rotation, translation = head_pose
    half_size = MARKER_SIZE_METERS / 2.0
    marker_corners = _project_marker_points(
        [
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ],
        rotation,
        translation,
        camera_matrix,
        distortion,
    )
    marker_side_pixels = float(
        np.mean(
            [
                np.linalg.norm(marker_corners[index] - marker_corners[(index + 1) % 4])
                for index in range(4)
            ]
        )
    )
    if marker_side_pixels < MIN_MARKER_SIDE_PIXELS:
        return None

    positions = led_calibration["points_meters"]
    centers = _project_marker_points(
        [
            [*positions["red"], 0],
            [*positions["blue"], 0],
        ],
        rotation,
        translation,
        camera_matrix,
        distortion,
    )
    signals = tuple(
        _led_color_signal(frame, center, marker_side_pixels)
        for center in centers
    )
    if any(signal is None for signal in signals):
        return None

    states = []
    for color, signal in zip(("red", "blue"), signals):
        profile = led_calibration["profiles"][color]
        off_signal = np.asarray(profile["off"], dtype=np.float64)
        on_signals = np.asarray(profile["on"], dtype=np.float64)
        off_distance = float(np.linalg.norm(signal - off_signal))
        on_distance = float(
            np.min(np.linalg.norm(on_signals - signal.reshape(1, 3), axis=1))
        )
        total_distance = off_distance + on_distance
        if total_distance < MIN_LED_CONTRAST:
            return None
        score = (off_distance - on_distance) / total_distance
        if abs(score) <= UNKNOWN_SCORE_MARGIN:
            return None
        states.append(score > 0)
    return tuple(states)
