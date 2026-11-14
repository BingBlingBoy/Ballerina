import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate the webcam used by the robot marker pipeline."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--columns", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.015,
        help="Printed checkerboard square width in metres (default: 0.015).",
    )
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("camera_calibration.npz"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.columns, args.rows, args.samples) <= 0 or args.square_size <= 0:
        raise ValueError("Board dimensions, sample count, and square size must be positive.")

    pattern = (args.columns, args.rows)
    object_template = np.zeros((args.rows * args.columns, 3), dtype=np.float32)
    object_template[:, :2] = (
        np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
        * args.square_size
    )

    object_points = []
    image_points = []
    image_size = None
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    try:
        print(
            f"Show the {args.columns}x{args.rows} inner-corner board at varied "
            "positions and angles. Press c to capture a detected board; q to finish."
        )
        while len(object_points) < args.samples:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Camera frame capture failed during calibration.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, pattern)
            display = frame.copy()
            if found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                cv2.drawChessboardCorners(display, pattern, corners, found)

            cv2.putText(
                display,
                f"Captured {len(object_points)}/{args.samples} - c capture, q finish",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Camera Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and found:
                object_points.append(object_template.copy())
                image_points.append(corners)
                print(f"Captured calibration view {len(object_points)}")

        if len(object_points) < 10:
            raise RuntimeError(
                f"Need at least 10 calibration views; captured {len(object_points)}."
            )

        rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=image_size[0],
            image_height=image_size[1],
            reprojection_error=rms,
        )
        print(f"Saved calibration to {args.output} (RMS reprojection error {rms:.3f}px)")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
