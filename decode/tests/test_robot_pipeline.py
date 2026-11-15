import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from main import PredictionSmoother, predict_letter
from robot_vision.robot_pose_features import (
    LETTER_DATASET_COLUMNS,
    LETTER_LABELS,
    LEGACY_LETTER_DATASET_COLUMNS,
    MARKER_IDS,
    MARKER_SIZE_METERS,
    LED_DETECTOR_COLUMN,
    LED_DETECTOR_VERSION,
    LED_FEATURE_COLUMNS,
    LED_VALID_COLUMN,
    PREVIOUS_LETTER_DATASET_COLUMNS,
    POSE_FEATURE_GROUPS,
    POSE_FEATURE_COLUMNS,
    POSE_MASK_COLUMNS,
    estimate_marker_poses,
    extract_features,
    extract_pose_observation,
    extract_pose_features,
)
from robot_vision.collect_robot_data import (
    ensure_dataset,
    label_from_key,
    write_sample,
)
from robot_vision.led_features import (
    detect_led_states,
    led_signals_from_marker_corners,
    load_led_calibration,
    marker_plane_point_from_click,
    save_led_calibration,
)
from train_classifier import parse_args, split_by_session, train_model


def pose_set(yaw=0.0, left_hand_height=0.3):
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def rotated(position):
        return rotation @ np.asarray(position, dtype=float)

    return {
        MARKER_IDS["base"]: (np.eye(3), np.zeros(3)),
        MARKER_IDS["left_shoulder"]: (
            rotation,
            rotated([-0.1, 0.0, 0.5]),
        ),
        MARKER_IDS["right_shoulder"]: (
            rotation,
            rotated([0.1, 0.0, 0.5]),
        ),
        MARKER_IDS["head"]: (rotation, rotated([0.0, 0.0, 0.7])),
        MARKER_IDS["left_hand"]: (
            rotation,
            rotated([-0.2, 0.0, left_hand_height]),
        ),
        MARKER_IDS["right_hand"]: (
            rotation,
            rotated([0.2, 0.0, 0.3]),
        ),
    }


class RobotFeatureTests(unittest.TestCase):
    def test_missing_base_preserves_arm_and_head_features(self):
        poses = pose_set()
        del poses[MARKER_IDS["base"]]
        features, mask = extract_pose_observation(poses)

        self.assertIsNone(extract_pose_features(poses))
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])
        self.assertFalse(mask[3])
        self.assertTrue(mask[4])
        self.assertTrue(
            np.isfinite(features[list(POSE_FEATURE_GROUPS["left_arm"])]).all()
        )

    def test_missing_hand_preserves_other_joint_features(self):
        poses = pose_set()
        del poses[MARKER_IDS["left_hand"]]
        features, mask = extract_pose_observation(poses)

        self.assertIsNone(extract_pose_features(poses))
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])
        self.assertTrue(mask[4])
        self.assertTrue(
            np.isnan(features[list(POSE_FEATURE_GROUPS["left_arm"])]).all()
        )

    def test_single_shoulder_supports_torso_orientation_not_geometry_or_head(self):
        poses = pose_set()
        del poses[MARKER_IDS["right_shoulder"]]
        features, mask = extract_pose_observation(poses)

        self.assertTrue(mask[0])
        self.assertTrue(mask[3])
        self.assertFalse(mask[2])
        self.assertFalse(mask[4])
        self.assertTrue(
            np.isfinite(features[list(POSE_FEATURE_GROUPS["torso_orientation"])]).all()
        )
        self.assertTrue(
            np.isnan(features[list(POSE_FEATURE_GROUPS["torso_geometry"])]).all()
        )

    def test_feature_groups_are_independent_when_base_is_missing(self):
        poses = pose_set()
        del poses[MARKER_IDS["base"]]
        features = extract_features(poses)

        self.assertIsNotNone(features["left_arm"])
        self.assertIsNotNone(features["right_arm"])
        self.assertIsNotNone(features["head"])
        self.assertIsNone(features["torso"])

    def test_torso_yaw_is_observable_and_head_feature_is_relative(self):
        neutral = extract_features(pose_set())
        rotated = extract_features(pose_set(yaw=np.deg2rad(30)))

        self.assertFalse(np.allclose(neutral["torso"], rotated["torso"]))
        np.testing.assert_allclose(neutral["head"], rotated["head"], atol=1e-6)

    def test_arm_height_changes_only_corresponding_arm_vector(self):
        low = extract_features(pose_set(left_hand_height=0.3))
        high = extract_features(pose_set(left_hand_height=0.6))

        self.assertFalse(np.allclose(low["left_arm"], high["left_arm"]))
        np.testing.assert_allclose(low["right_arm"], high["right_arm"], atol=1e-6)

    def test_frontal_square_marker_pose_passes_reprojection_validation(self):
        half_size = MARKER_SIZE_METERS / 2.0
        object_points = np.array(
            [
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0],
            ],
            dtype=np.float32,
        )
        camera_matrix = np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((5, 1))
        corners, _ = cv2.projectPoints(
            object_points,
            np.array([[np.pi], [0.0], [0.0]]),
            np.array([[0.0], [0.0], [0.8]]),
            camera_matrix,
            distortion,
        )

        poses = estimate_marker_poses(
            [corners.reshape(1, 4, 2)],
            np.array([[MARKER_IDS["base"]]]),
            camera_matrix,
            distortion,
        )

        self.assertIn(MARKER_IDS["base"], poses)
        np.testing.assert_allclose(
            poses[MARKER_IDS["base"]][1],
            [0.0, 0.0, 0.8],
            atol=1e-4,
        )


class AlphabetPipelineTests(unittest.TestCase):
    def test_training_letter_flags_select_requested_classes(self):
        with patch("sys.argv", ["train_classifier.py", "--a", "--c"]):
            args = parse_args()
        selected = tuple(
            label
            for label in ("A", "B", "C")
            if getattr(args, label.lower())
        )
        self.assertEqual(selected, ("A", "C"))

    def test_training_letters_option_remains_available(self):
        with patch("sys.argv", ["train_classifier.py", "--letters", "B", "D"]):
            args = parse_args()
        self.assertEqual(args.letters, ["B", "D"])

    def test_training_selection_options_are_mutually_exclusive(self):
        with patch(
            "sys.argv",
            ["train_classifier.py", "--a", "--letters", "B"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_every_letter_key_is_a_label_and_escape_is_not(self):
        for label in LETTER_LABELS:
            self.assertEqual(label_from_key(ord(label.lower())), label)
            self.assertEqual(label_from_key(ord(label)), label)
        self.assertIsNone(label_from_key(27))
        self.assertEqual(label_from_key(ord("q")), "Q")

    def test_collector_writes_marker_and_feature_masks(self):
        features = np.full(len(POSE_FEATURE_COLUMNS), np.nan)
        group_mask = (True, True, True, True, False)
        features[: len(POSE_FEATURE_COLUMNS) - 6] = 1.0
        output = StringIO()
        writer = csv.writer(output)

        write_sample(
            writer,
            "session",
            "capture",
            "A",
            (1, 2, 3, 4, 5),
            (False, True),
            features,
            group_mask,
        )

        row = dict(
            zip(
                LETTER_DATASET_COLUMNS,
                next(csv.reader(StringIO(output.getvalue()))),
            )
        )
        self.assertEqual(row["visible_marker_ids"], "1,2,3,4,5")
        for name, observed in zip(POSE_MASK_COLUMNS, group_mask):
            self.assertEqual(row[POSE_MASK_COLUMNS[name]], str(int(observed)))
        self.assertEqual(row[LED_FEATURE_COLUMNS[0]], "0")
        self.assertEqual(row[LED_FEATURE_COLUMNS[1]], "1")
        self.assertEqual(row[LED_VALID_COLUMN], "1")
        self.assertEqual(row[LED_DETECTOR_COLUMN], LED_DETECTOR_VERSION)
        self.assertEqual(row[POSE_FEATURE_COLUMNS[-1]], "nan")

    def test_collector_writes_unknown_led_states_as_invalid(self):
        features = np.full(len(POSE_FEATURE_COLUMNS), np.nan)
        output = StringIO()
        writer = csv.writer(output)
        write_sample(
            writer,
            "session",
            "capture",
            "A",
            (0, 1, 2),
            None,
            features,
            (True, False, False, False, True),
        )
        row = dict(zip(LETTER_DATASET_COLUMNS, next(csv.reader(StringIO(output.getvalue())))))
        self.assertEqual(row[LED_FEATURE_COLUMNS[0]], "")
        self.assertEqual(row[LED_FEATURE_COLUMNS[1]], "")
        self.assertEqual(row[LED_VALID_COLUMN], "0")
        self.assertEqual(row[LED_DETECTOR_COLUMN], "")

    def test_collector_migrates_legacy_dataset_without_losing_pose_data(self):
        legacy_row = [
            "session",
            "capture",
            "2026-01-01T00:00:00+00:00",
            "A",
            "0,1,2,3,4,5",
            *["1"] * len(POSE_MASK_COLUMNS),
            *["0.25"] * len(POSE_FEATURE_COLUMNS),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset:
                writer = csv.writer(dataset)
                writer.writerow(LEGACY_LETTER_DATASET_COLUMNS)
                writer.writerow(legacy_row)

            with patch("robot_vision.collect_robot_data.CSV_FILE", dataset_path):
                ensure_dataset()

            with dataset_path.open("r", newline="", encoding="utf-8") as dataset:
                reader = csv.DictReader(dataset)
                row = next(reader)
                self.assertEqual(tuple(reader.fieldnames), LETTER_DATASET_COLUMNS)
            self.assertEqual(row[LED_FEATURE_COLUMNS[0]], "")
            self.assertEqual(row[LED_FEATURE_COLUMNS[1]], "")
            self.assertEqual(row[LED_VALID_COLUMN], "0")
            self.assertEqual(row[LED_DETECTOR_COLUMN], "")
            self.assertEqual(row[POSE_FEATURE_COLUMNS[0]], "0.25")

    def test_previous_led_dataset_migration_disables_uncalibrated_features(self):
        previous_row = [
            "session",
            "capture",
            "2026-01-01T00:00:00+00:00",
            "A",
            "0,1,2,3,4,5",
            *["1"] * len(POSE_MASK_COLUMNS),
            "1",
            "0",
            "1",
            *["0.25"] * len(POSE_FEATURE_COLUMNS),
        ]
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset:
                writer = csv.writer(dataset)
                writer.writerow(PREVIOUS_LETTER_DATASET_COLUMNS)
                writer.writerow(previous_row)

            with patch("robot_vision.collect_robot_data.CSV_FILE", dataset_path):
                ensure_dataset()

            with dataset_path.open("r", newline="", encoding="utf-8") as dataset:
                reader = csv.DictReader(dataset)
                row = next(reader)
                self.assertEqual(tuple(reader.fieldnames), LETTER_DATASET_COLUMNS)
            self.assertEqual(row[LED_FEATURE_COLUMNS[0]], "1")
            self.assertEqual(row[LED_FEATURE_COLUMNS[1]], "0")
            self.assertEqual(row[LED_VALID_COLUMN], "1")
            self.assertEqual(row[LED_DETECTOR_COLUMN], "")
            self.assertEqual(row[POSE_FEATURE_COLUMNS[0]], "0.25")

    def test_led_detector_uses_head_marker_relative_rois(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        camera_matrix = np.array(
            [[1000.0, 0.0, 100.0], [0.0, 1000.0, 60.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros((5, 1))
        pose = (np.eye(3), np.array([-0.045, 0.0, 1.0]))
        positions = {"red": [0.01, 0.0], "blue": [-0.01, 0.0]}
        half_size = MARKER_SIZE_METERS / 2.0
        marker_object_points = np.array(
            [
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0],
            ],
            dtype=np.float32,
        )
        marker_corners, _ = cv2.projectPoints(
            marker_object_points,
            np.zeros((3, 1)),
            np.array([[-0.045], [0.0], [1.0]]),
            camera_matrix,
            distortion,
        )
        marker_corners = marker_corners.reshape(4, 2)
        off_signals = led_signals_from_marker_corners(
            frame,
            marker_corners,
            positions,
        )
        cv2.circle(frame, (65, 60), 5, (0, 100, 255), -1)
        red_only_signals = led_signals_from_marker_corners(
            frame,
            marker_corners,
            positions,
        )
        frame.fill(0)
        cv2.circle(frame, (45, 60), 5, (255, 100, 0), -1)
        blue_only_signals = led_signals_from_marker_corners(
            frame,
            marker_corners,
            positions,
        )
        cv2.circle(frame, (65, 60), 5, (0, 100, 255), -1)
        on_signals = led_signals_from_marker_corners(
            frame,
            marker_corners,
            positions,
        )
        calibration = {
            "points_meters": positions,
            "profiles": {
                color: {
                    "off": off_signals[index],
                    "on": [
                        (red_only_signals if index == 0 else blue_only_signals)[index],
                        on_signals[index],
                    ],
                }
                for index, color in enumerate(("red", "blue"))
            },
        }

        self.assertEqual(
            detect_led_states(
                frame,
                pose,
                camera_matrix,
                distortion,
                calibration,
            ),
            (True, True),
        )
        frame.fill(0)
        cv2.circle(frame, (65, 60), 5, (0, 100, 255), -1)
        self.assertEqual(
            detect_led_states(frame, pose, camera_matrix, distortion, calibration),
            (True, False),
        )
        frame.fill(0)
        cv2.circle(frame, (45, 60), 5, (255, 100, 0), -1)
        self.assertEqual(
            detect_led_states(frame, pose, camera_matrix, distortion, calibration),
            (False, True),
        )
        frame.fill(0)
        self.assertEqual(
            detect_led_states(
                frame,
                pose,
                camera_matrix,
                distortion,
                calibration,
            ),
            (False, False),
        )

    def test_led_states_are_unknown_when_head_marker_is_missing(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        self.assertIsNone(
            detect_led_states(
                frame,
                None,
                np.eye(3),
                np.zeros((5, 1)),
                {
                    "points_meters": {"red": [0.0, 0.0], "blue": [0.01, 0.0]},
                    "profiles": {
                        "red": {"off": [0.0, 0.0, 0.0], "on": [[0.0, 0.0, 255.0]]},
                        "blue": {"off": [0.0, 0.0, 0.0], "on": [[255.0, 0.0, 0.0]]},
                    },
                },
            )
        )

    def test_led_click_is_mapped_into_head_marker_plane(self):
        marker_corners = np.array(
            [[40, 20], [80, 20], [80, 60], [40, 60]],
            dtype=np.float32,
        )
        point = marker_plane_point_from_click((70, 30), marker_corners)
        np.testing.assert_allclose(point, [0.01, 0.01], atol=1e-5)

    def test_led_calibration_round_trips_camera_specific_profiles(self):
        points = {"red": [0.01, 0.0], "blue": [-0.01, 0.0]}
        profiles = {
            "red": {
                "off": [1.0, 2.0, 3.0],
                "on": [[5.0, 20.0, 80.0], [7.0, 25.0, 90.0]],
            },
            "blue": {
                "off": [2.0, 2.0, 2.0],
                "on": [[80.0, 20.0, 5.0], [90.0, 25.0, 7.0]],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "led_calibration.json"
            save_led_calibration(points, profiles, calibration_path)
            saved = load_led_calibration(calibration_path)
            self.assertEqual(
                saved["profiles"]["red"]["on"],
                [[5.0, 20.0, 80.0], [7.0, 25.0, 90.0]],
            )
            self.assertEqual(saved["profiles"]["blue"]["off"], [2.0, 2.0, 2.0])

    def test_session_split_keeps_sessions_disjoint(self):
        labels = pd.Series([label for _ in range(4) for label in LETTER_LABELS])
        sessions = pd.Series(
            [
                f"session-{session_index}"
                for session_index in range(4)
                for _ in LETTER_LABELS
            ]
        )
        features = np.arange(len(labels) * 2).reshape(len(labels), 2)

        train_indices, test_indices = split_by_session(
            features,
            labels,
            sessions,
            LETTER_LABELS,
        )

        self.assertFalse(
            set(sessions.iloc[train_indices]) & set(sessions.iloc[test_indices])
        )

    def test_smoother_requires_multiple_consistent_predictions(self):
        smoother = PredictionSmoother()
        self.assertIsNone(smoother.update("A"))
        self.assertIsNone(smoother.update("B"))
        self.assertIsNone(smoother.update("A"))
        self.assertEqual(smoother.update("A"), "A")
        smoother.reset()
        self.assertIsNone(smoother.update("B"))

    def test_all_letter_models_train_and_predict(self):
        rows = []
        full_mask = (True,) * len(POSE_FEATURE_GROUPS)
        partial_mask = (True, True, True, True, False)
        for label_index, label in enumerate(LETTER_LABELS):
            for session_index in range(4):
                capture_id = f"{label}-{session_index}"
                for sample_index in range(4):
                    bits = np.array(
                        [(label_index >> bit) & 1 for bit in range(5)] * 5
                    )[: len(POSE_FEATURE_COLUMNS)]
                    values = bits * 4.0
                    values = values + session_index * 0.002 + sample_index * 0.0001
                    group_mask = (
                        partial_mask if sample_index % 2 else full_mask
                    )
                    if not group_mask[-1]:
                        values[list(POSE_FEATURE_GROUPS["head"])] = np.nan
                    rows.append(
                        {
                            "session_id": f"session-{session_index}",
                            "capture_id": capture_id,
                            "timestamp_utc": f"2026-01-01T00:00:{sample_index:02d}+00:00",
                            "label": label,
                            "visible_marker_ids": (
                                "1,2,3,4,5" if not group_mask[-1] else "0,1,2,3,4,5"
                            ),
                            **dict(
                                zip(
                                    POSE_MASK_COLUMNS.values(),
                                    map(int, group_mask),
                                )
                            ),
                            LED_FEATURE_COLUMNS[0]: int(label_index & 1 != 0),
                            LED_FEATURE_COLUMNS[1]: int(label_index & 2 != 0),
                            LED_VALID_COLUMN: 1,
                            LED_DETECTOR_COLUMN: LED_DETECTOR_VERSION,
                            **dict(zip(POSE_FEATURE_COLUMNS, values)),
                        }
                    )

        dataset = pd.DataFrame(rows, columns=LETTER_DATASET_COLUMNS)
        bundle = train_model(dataset)
        self.assertEqual(tuple(bundle["classes"]), LETTER_LABELS)
        features = dataset.loc[0, POSE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        label, confidence = predict_letter(bundle["model"], features)
        self.assertIn(label, (*LETTER_LABELS, None))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        partial_label, partial_confidence = predict_letter(
            bundle["model"], features, partial_mask, (False, False)
        )
        self.assertIn(partial_label, (*LETTER_LABELS, None))
        self.assertGreaterEqual(partial_confidence, 0.0)
        self.assertLessEqual(partial_confidence, 1.0)
        torso_only_mask = (False, False, True, True, False)
        self.assertEqual(
            predict_letter(bundle["model"], features, torso_only_mask),
            (None, 0.0),
        )
        uncollected_mask = (True, True, False, False, False)
        self.assertIn((uncollected_mask, False), bundle["model"])
        self.assertIn((uncollected_mask, True), bundle["model"])

    def test_training_can_select_only_a_and_b(self):
        rows = []
        for label_index, label in enumerate(("A", "B")):
            for session_index in range(4):
                capture_id = f"{label}-{session_index}"
                for sample_index in range(4):
                    values = np.full(
                        len(POSE_FEATURE_COLUMNS),
                        label_index * 4.0 + session_index * 0.002,
                    )
                    values += np.arange(len(POSE_FEATURE_COLUMNS)) * 0.001
                    values += sample_index * 0.0001
                    group_mask = (
                        (True, True, True, True, False)
                        if label == "B"
                        else (True,) * len(POSE_FEATURE_GROUPS)
                    )
                    if not group_mask[-1]:
                        values[list(POSE_FEATURE_GROUPS["head"])] = np.nan
                    rows.append(
                        {
                            "session_id": f"session-{session_index}",
                            "capture_id": capture_id,
                            "timestamp_utc": f"2026-01-01T00:00:{sample_index:02d}+00:00",
                            "label": label,
                            "visible_marker_ids": (
                                "1,2,3,4,5" if not group_mask[-1] else "0,1,2,3,4,5"
                            ),
                            **dict(
                                zip(
                                    POSE_MASK_COLUMNS.values(),
                                    map(int, group_mask),
                                )
                            ),
                            LED_FEATURE_COLUMNS[0]: label_index,
                            LED_FEATURE_COLUMNS[1]: 0,
                            LED_VALID_COLUMN: 1,
                            LED_DETECTOR_COLUMN: LED_DETECTOR_VERSION,
                            **dict(zip(POSE_FEATURE_COLUMNS, values)),
                        }
                    )

        dataset = pd.DataFrame(rows, columns=LETTER_DATASET_COLUMNS)
        bundle = train_model(dataset, ("a", "b"))
        self.assertEqual(bundle["classes"], ("A", "B"))
        full_mask = (True,) * len(POSE_FEATURE_GROUPS)
        self.assertNotIn(full_mask, bundle["model"])
        partial_mask = (True, True, True, True, False)
        self.assertEqual(
            tuple(bundle["model"][(partial_mask, False)].classes_),
            ("A", "B"),
        )
        feature_values = dataset.loc[0, POSE_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )
        partial_label, partial_confidence = predict_letter(
            bundle["model"], feature_values, partial_mask, (0, 0)
        )
        self.assertIn(partial_label, ("A", "B"))
        self.assertGreaterEqual(partial_confidence, 0.60)
        self.assertIn((partial_mask, True), bundle["model"])
        self.assertIn(
            (partial_mask, False),
            bundle["model"],
        )


if __name__ == "__main__":
    unittest.main()
