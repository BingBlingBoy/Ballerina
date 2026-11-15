import argparse
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .robot_pose_features import (
        MARKER_IDS,
        detect_markers,
    )
    from .led_features import (
        LED_CALIBRATION_FILE,
        led_signals_from_marker_corners,
        marker_plane_point_from_click,
        save_led_calibration,
    )
else:
    from robot_pose_features import (
        MARKER_IDS,
        detect_markers,
    )
    from led_features import (
        LED_CALIBRATION_FILE,
        led_signals_from_marker_corners,
        marker_plane_point_from_click,
        save_led_calibration,
    )


CAMERA_INDEX = 0
WINDOW_NAME = "Calibrate Robot LEDs"
CALIBRATION_FRAMES = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate LED locations relative to the head ArUco marker."
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--output", type=Path, default=LED_CALIBRATION_FILE)
    return parser.parse_args()


def _head_marker_corners(corners, ids):
    if ids is None:
        return None
    head_indices = np.flatnonzero(
        np.asarray(ids).reshape(-1) == MARKER_IDS["head"]
    )
    if len(head_indices) != 1:
        return None
    return np.asarray(corners[int(head_indices[0])], dtype=np.float32).reshape(4, 2)


def calibrate(camera_index, output_path):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}.")

    clicks = []
    profiles = {"red": {"on": []}, "blue": {"on": []}}
    captured_modes = set()
    current = {
        "marker_corners": None,
        "error": None,
        "capture_mode": None,
        "samples": [],
    }

    def on_mouse(event, x, y, _flags, _parameter):
        if (
            event != cv2.EVENT_LBUTTONDOWN
            or len(clicks) >= 2
            or current["capture_mode"] is not None
        ):
            return
        marker_corners = current["marker_corners"]
        if marker_corners is None:
            current["error"] = "Keep the head marker visible while clicking."
            return
        try:
            clicks.append(
                marker_plane_point_from_click((x, y), marker_corners)
            )
            current["error"] = None
        except ValueError as error:
            current["error"] = str(error)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect_markers(gray)
            marker_corners = _head_marker_corners(corners, ids)
            current["marker_corners"] = marker_corners
            display = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)

            if marker_corners is None:
                instruction = "Show exactly one head marker (ID 0)"
            elif len(clicks) == 0:
                instruction = "Click the center of the RED LED"
            elif len(clicks) == 1:
                instruction = "Click the center of the BLUE LED"
            elif current["capture_mode"]:
                instruction = (
                    f"Sampling {current['capture_mode']} LED state: "
                    f"{len(current['samples'])}/{CALIBRATION_FRAMES}"
                )
            elif captured_modes != {"off", "red on", "blue on", "both on"}:
                instruction = (
                    "Capture OFF (O), red only (R), blue only (B), both ON (A)"
                )
            else:
                instruction = "Press Enter to save; O/R/B/A to recalibrate"

            cv2.putText(
                display,
                instruction,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "Keep the robot still and follow the requested LED state.",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )

            if current["error"]:
                cv2.putText(
                    display,
                    current["error"],
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                )

            if len(clicks) == 2 and marker_corners is not None:
                signals = led_signals_from_marker_corners(
                    frame,
                    marker_corners,
                    {"red": clicks[0], "blue": clicks[1]},
                )
                if current["capture_mode"] and signals is not None:
                    current["samples"].append(signals)
                    if len(current["samples"]) >= CALIBRATION_FRAMES:
                        mode = current["capture_mode"]
                        if mode == "off":
                            profiles["red"]["off"] = np.mean(
                                [sample[0] for sample in current["samples"]],
                                axis=0,
                            )
                            profiles["blue"]["off"] = np.mean(
                                [sample[1] for sample in current["samples"]],
                                axis=0,
                            )
                        elif mode == "red on":
                            profiles["red"]["on"].append(np.mean(
                                [sample[0] for sample in current["samples"]],
                                axis=0,
                            ).tolist())
                        elif mode == "blue on":
                            profiles["blue"]["on"].append(np.mean(
                                [sample[1] for sample in current["samples"]],
                                axis=0,
                            ).tolist())
                        else:
                            profiles["red"]["on"].append(np.mean(
                                [sample[0] for sample in current["samples"]],
                                axis=0,
                            ).tolist())
                            profiles["blue"]["on"].append(np.mean(
                                [sample[1] for sample in current["samples"]],
                                axis=0,
                            ).tolist())
                        captured_modes.add(mode)
                        current["capture_mode"] = None
                        current["samples"].clear()
                        current["error"] = None
                elif current["capture_mode"]:
                    current["error"] = (
                        "LED regions are outside the frame or marker is too small."
                    )

            sampled_states = (
                "Captured: "
                + ", ".join(
                    name
                    for mode, name in (
                        ("off", "off"),
                        ("red on", "red"),
                        ("blue on", "blue"),
                        ("both on", "both"),
                    )
                    if mode in captured_modes
                )
            )
            cv2.putText(
                display,
                sampled_states,
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 0),
                1,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return False
            if key in (ord("x"), ord("X")):
                clicks.clear()
                profiles = {"red": {"on": []}, "blue": {"on": []}}
                captured_modes.clear()
                current["capture_mode"] = None
                current["samples"].clear()
                current["error"] = None
            elif len(clicks) == 2 and current["capture_mode"] is None:
                if key in (ord("o"), ord("O")):
                    current["capture_mode"] = "off"
                    current["samples"].clear()
                elif key in (ord("r"), ord("R")):
                    current["capture_mode"] = "red on"
                    current["samples"].clear()
                elif key in (ord("b"), ord("B")):
                    current["capture_mode"] = "blue on"
                    current["samples"].clear()
                elif key in (ord("a"), ord("A")):
                    current["capture_mode"] = "both on"
                    current["samples"].clear()
                elif key in (10, 13):
                    if captured_modes != {"off", "red on", "blue on", "both on"}:
                        current["error"] = (
                            "Capture both OFF, red only ON, blue only ON, and "
                            "both ON samples."
                        )
                        continue
                    try:
                        save_led_calibration(
                            {"red": clicks[0], "blue": clicks[1]},
                            profiles,
                            output_path,
                        )
                    except ValueError as error:
                        current["error"] = str(error)
                        continue
                    print(f"Saved LED calibration to {output_path}")
                    return True
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    if not calibrate(args.camera, args.output):
        print("LED calibration cancelled; no file was saved.")


if __name__ == "__main__":
    main()
