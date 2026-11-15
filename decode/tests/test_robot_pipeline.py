import csv
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from main import (
    LetterWordBuffer,
    parse_args as parse_main_args,
    predict_letter,
)
from robot_vision.robot_pose_features import (
    LETTER_DATASET_COLUMNS,
    LETTER_LABELS,
    MARKER_IDS,
    MARKER_SIZE_METERS,
    LED_DETECTOR_COLUMN,
    LED_DETECTOR_VERSION,
    LED_FEATURE_COLUMNS,
    LED_VALID_COLUMN,
    MARKER_SIGNAL_COLUMN,
    MARKER_SIGNAL_FEATURE_COLUMNS,
    MARKER_SIGNAL_IDS,
    POSE_FEATURE_GROUPS,
    POSE_FEATURE_COLUMNS,
    POSE_MASK_COLUMNS,
    SPACE_LABEL,
    estimate_marker_poses,
    extract_features,
    extract_pose_observation,
    extract_pose_features,
    feature_group_mask_for_marker_ids,
    marker_signal_from_ids,
)
from robot_vision.collect_robot_data import (
    ensure_dataset,
    label_from_key,
    parse_args as parse_collector_args,
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
    def test_missing_base_disables_base_relative_features(self):
        poses = pose_set()
        del poses[MARKER_IDS["base"]]
        features, mask = extract_pose_observation(poses)

        self.assertIsNone(extract_pose_features(poses))
        self.assertEqual(mask, (False, False, False))
        self.assertTrue(np.isnan(features).all())

    def test_missing_hand_preserves_other_joint_features(self):
        poses = pose_set()
        del poses[MARKER_IDS["left_hand"]]
        features, mask = extract_pose_observation(poses)

        self.assertIsNone(extract_pose_features(poses))
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(
            np.isnan(features[list(POSE_FEATURE_GROUPS["left_arm"])]).all()
        )

    def test_only_head_hand_and_base_markers_are_registered(self):
        poses = pose_set()
        features, mask = extract_pose_observation(poses)

        self.assertEqual(set(MARKER_IDS), {"head", "left_hand", "right_hand", "base"})
        self.assertEqual(len(mask), 3)
        self.assertTrue(all(mask))
        self.assertEqual(set(extract_features(poses)), {"left_arm", "right_arm", "head"})

    def test_feature_visibility_uses_only_registered_markers(self):
        self.assertEqual(
            feature_group_mask_for_marker_ids((1, 2, 3, 4, 5)),
            (True, True, False),
        )

    def test_marker_signal_requires_exactly_one_signal_id(self):
        self.assertEqual(marker_signal_from_ids(np.array([[1], [3]])), 1)
        self.assertEqual(marker_signal_from_ids(np.array([[2]])), 2)
        self.assertIsNone(marker_signal_from_ids(np.array([[1], [2]])))
        self.assertIsNone(marker_signal_from_ids(np.array([[3]])))
        self.assertIsNone(marker_signal_from_ids(None))

    def test_feature_groups_are_independent_when_base_is_missing(self):
        poses = pose_set()
        del poses[MARKER_IDS["base"]]
        features = extract_features(poses)

        self.assertIsNone(features["left_arm"])
        self.assertIsNone(features["right_arm"])
        self.assertIsNone(features["head"])

    def test_head_and_arm_features_are_relative_to_stationary_base(self):
        neutral = extract_features(pose_set())
        rotated = extract_features(pose_set(yaw=np.deg2rad(30)))

        self.assertFalse(np.allclose(neutral["left_arm"], rotated["left_arm"]))
        self.assertFalse(np.allclose(neutral["head"], rotated["head"]))

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
        ignored_poses = estimate_marker_poses(
            [corners.reshape(1, 4, 2), corners.reshape(1, 4, 2)],
            np.array([[1], [2]]),
            camera_matrix,
            distortion,
        )
        self.assertEqual(ignored_poses, {})


class AlphabetPipelineTests(unittest.TestCase):
    def test_camera_flags_are_available_for_collection_and_live_inference(self):
        with patch("sys.argv", ["main.py", "--camera", "2"]):
            self.assertEqual(parse_main_args().camera, 2)
        with patch(
            "sys.argv",
            ["collect_robot_data.py", "--camera", "1"],
        ):
            self.assertEqual(parse_collector_args().camera, 1)

    def test_training_letter_flags_select_requested_classes(self):
        with patch(
            "sys.argv",
            ["train_classifier.py", "--a", "--c", "--h"],
        ):
            args = parse_args()
        selected = tuple(
            label
            for label in "ABCDEFGH"
            if getattr(args, label.lower())
        )
        self.assertEqual(selected, ("A", "C", "H"))

    def test_training_letters_option_remains_available(self):
        with patch("sys.argv", ["train_classifier.py", "--letters", "B", "D"]):
            args = parse_args()
        self.assertEqual(args.letters, ["B", "D"])

    def test_repeat_class_can_be_requested_with_letter_flags_or_letters(self):
        with patch("sys.argv", ["train_classifier.py", "--a", "--repeat"]):
            args = parse_args()
        self.assertTrue(args.a)
        self.assertTrue(args.repeat)

        with patch(
            "sys.argv",
            ["train_classifier.py", "--letters", "A", "B", "--repeat"],
        ):
            args = parse_args()
        self.assertEqual(args.letters, ["A", "B"])
        self.assertTrue(args.repeat)

    def test_space_class_can_be_requested_with_training_flags(self):
        with patch("sys.argv", ["train_classifier.py", "--a", "--space"]):
            args = parse_args()
        self.assertTrue(args.a)
        self.assertTrue(args.space)

    def test_training_selection_options_are_mutually_exclusive(self):
        with patch(
            "sys.argv",
            ["train_classifier.py", "--h", "--letters", "B"],
        ):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_every_letter_key_is_a_label_and_escape_is_not(self):
        for label in LETTER_LABELS:
            self.assertEqual(label_from_key(ord(label.lower())), label)
            self.assertEqual(label_from_key(ord(label)), label)
        self.assertEqual(label_from_key(ord("~")), "~")
        self.assertEqual(label_from_key(ord(" ")), SPACE_LABEL)
        self.assertIsNone(label_from_key(27))
        self.assertEqual(label_from_key(ord("q")), "Q")

    def test_collector_writes_marker_and_feature_masks(self):
        features = np.full(len(POSE_FEATURE_COLUMNS), np.nan)
        group_mask = (True, True, False)
        features[: len(POSE_FEATURE_COLUMNS) - 6] = 1.0
        output = StringIO()
        writer = csv.writer(output)

        write_sample(
            writer,
            "session",
            "capture",
            "A",
            (0, 3, 4, 5),
            (False, True),
            2,
            features,
            group_mask,
        )

        row = dict(
            zip(
                LETTER_DATASET_COLUMNS,
                next(csv.reader(StringIO(output.getvalue()))),
            )
        )
        self.assertEqual(row["visible_marker_ids"], "0,3,4,5")
        for name, observed in zip(POSE_MASK_COLUMNS, group_mask):
            self.assertEqual(row[POSE_MASK_COLUMNS[name]], str(int(observed)))
        self.assertEqual(row[MARKER_SIGNAL_COLUMN], "id2")
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
            (0, 3, 4, 5),
            None,
            None,
            features,
            (False, False, False),
        )
        row = dict(zip(LETTER_DATASET_COLUMNS, next(csv.reader(StringIO(output.getvalue())))))
        self.assertEqual(row[MARKER_SIGNAL_COLUMN], "")
        self.assertEqual(row[LED_FEATURE_COLUMNS[0]], "")
        self.assertEqual(row[LED_FEATURE_COLUMNS[1]], "")
        self.assertEqual(row[LED_VALID_COLUMN], "0")
        self.assertEqual(row[LED_DETECTOR_COLUMN], "")

    def test_collector_rejects_incompatible_dataset_without_changing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.csv"
            with dataset_path.open("w", newline="", encoding="utf-8") as dataset:
                writer = csv.writer(dataset)
                writer.writerow(("legacy", "old", "dataset"))
                writer.writerow(("keep", "all", "data"))

            with patch("robot_vision.collect_robot_data.CSV_FILE", dataset_path):
                with self.assertRaises(RuntimeError):
                    ensure_dataset()

            self.assertEqual(
                dataset_path.read_text(encoding="utf-8").splitlines(),
                ["legacy,old,dataset", "keep,all,data"],
            )

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

    def test_word_buffer_records_manual_inputs_and_finishes_words(self):
        buffer = LetterWordBuffer()
        self.assertFalse(buffer.record("~"))
        self.assertTrue(buffer.record("A"))
        self.assertTrue(buffer.record("A"))
        self.assertTrue(buffer.record("~"))
        self.assertFalse(buffer.record("~"))
        self.assertTrue(buffer.record("B"))
        self.assertEqual(buffer.current_word, "AA~B")

        self.assertFalse(buffer.record(SPACE_LABEL))
        self.assertEqual(buffer.current_word, "AA~B")
        self.assertEqual(buffer.output_text, "")
        self.assertTrue(buffer.record("C"))
        self.assertEqual(buffer.current_word, "AA~BC")
        self.assertEqual(buffer.finish_word(), "AA~BC")
        self.assertEqual(buffer.current_word, "")
        self.assertEqual(buffer.last_finalized, "AA~BC")
        self.assertEqual(buffer.output_text, "AA~BC")

        self.assertTrue(buffer.record("B"))
        self.assertEqual(buffer.current_word, "B")
        self.assertEqual(buffer.finish(), "B!")


    def test_all_letter_models_train_and_predict(self):
        rows = []
        full_mask = (True,) * len(POSE_FEATURE_GROUPS)
        partial_mask = (True, True, False)
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
                                "3,4,5" if not group_mask[-1] else "0,3,4,5"
                            ),
                            **dict(
                                zip(
                                    POSE_MASK_COLUMNS.values(),
                                    map(int, group_mask),
                                )
                            ),
                            MARKER_SIGNAL_COLUMN: (
                                "id1" if sample_index % 2 == 0 else "id2"
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
        self.assertIn(partial_label, LETTER_LABELS)
        self.assertGreaterEqual(partial_confidence, 0.0)
        self.assertLessEqual(partial_confidence, 1.0)
        uncertain_model = {
            ((True, True, False), False, False): type(
                "LowConfidenceModel",
                (),
                {
                    "classes_": np.array(["A", "B"]),
                    "predict_proba": lambda self, values: np.array([[0.55, 0.45]]),
                },
            )()
        }
        uncertain_label, uncertain_confidence = predict_letter(
            uncertain_model,
            features,
            partial_mask,
        )
        self.assertEqual(uncertain_label, "A")
        self.assertLess(uncertain_confidence, 0.60)
        head_only_mask = (False, False, True)
        self.assertEqual(
            predict_letter(bundle["model"], features, head_only_mask),
            (None, 0.0),
        )
        uncollected_mask = (False, True, True)
        self.assertIn((uncollected_mask, False, False), bundle["model"])
        self.assertIn((uncollected_mask, True, False), bundle["model"])
        self.assertIn((uncollected_mask, False, True), bundle["model"])
        self.assertEqual(
            len(MARKER_SIGNAL_FEATURE_COLUMNS),
            2,
        )

    def test_training_can_select_letters_and_repeat_pose(self):
        rows = []
        selected_classes = ("A", "B", SPACE_LABEL, "~")
        for label_index, label in enumerate(selected_classes):
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
                        (True, True, False)
                        if label in ("B", "~", SPACE_LABEL)
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
                                "3,4,5" if not group_mask[-1] else "0,3,4,5"
                            ),
                            **dict(
                                zip(
                                    POSE_MASK_COLUMNS.values(),
                                    map(int, group_mask),
                                )
                            ),
                            MARKER_SIGNAL_COLUMN: (
                                "id1" if label == "A" else "id2"
                            ),
                            LED_FEATURE_COLUMNS[0]: 0,
                            LED_FEATURE_COLUMNS[1]: 0,
                            LED_VALID_COLUMN: 1,
                            LED_DETECTOR_COLUMN: LED_DETECTOR_VERSION,
                            **dict(zip(POSE_FEATURE_COLUMNS, values)),
                        }
                    )

        dataset = pd.DataFrame(rows, columns=LETTER_DATASET_COLUMNS)
        bundle = train_model(dataset, ("a", "b", "~", SPACE_LABEL))
        self.assertEqual(bundle["classes"], selected_classes)
        full_mask = (True,) * len(POSE_FEATURE_GROUPS)
        self.assertNotIn(full_mask, bundle["model"])
        partial_mask = (True, True, False)
        self.assertEqual(
            tuple(bundle["model"][(partial_mask, False, False)].classes_),
            selected_classes,
        )
        feature_values = dataset.loc[dataset["label"] == "A", POSE_FEATURE_COLUMNS].iloc[0].to_numpy(
            dtype=np.float64
        )
        partial_label, partial_confidence = predict_letter(
            bundle["model"],
            feature_values,
            partial_mask,
            marker_signal=1,
        )
        self.assertEqual(partial_label, "A")
        self.assertGreaterEqual(partial_confidence, 0.60)
        b_features = dataset.loc[
            dataset["label"] == "B",
            POSE_FEATURE_COLUMNS,
        ].iloc[0].to_numpy(dtype=np.float64)
        other_signal_label, other_signal_confidence = predict_letter(
            bundle["model"],
            b_features,
            partial_mask,
            marker_signal=2,
        )
        self.assertEqual(other_signal_label, "B")
        self.assertGreaterEqual(other_signal_confidence, 0.0)
        repeat_features = dataset.loc[
            dataset["label"] == "~",
            POSE_FEATURE_COLUMNS,
        ].iloc[0].to_numpy(dtype=np.float64)
        repeat_label, repeat_confidence = predict_letter(
            bundle["model"],
            repeat_features,
            partial_mask,
            marker_signal=2,
        )
        self.assertEqual(repeat_label, "~")
        self.assertGreaterEqual(repeat_confidence, 0.0)
        space_features = dataset.loc[
            dataset["label"] == SPACE_LABEL,
            POSE_FEATURE_COLUMNS,
        ].iloc[0].to_numpy(dtype=np.float64)
        space_label, _ = predict_letter(
            bundle["model"],
            space_features,
            partial_mask,
            marker_signal=2,
        )
        self.assertEqual(space_label, SPACE_LABEL)
        self.assertIn((partial_mask, False, True), bundle["model"])
        self.assertIn((partial_mask, True, False), bundle["model"])
        self.assertIn((partial_mask, True, True), bundle["model"])
        self.assertIn(
            (partial_mask, False, False),
            bundle["model"],
        )


if __name__ == "__main__":
    unittest.main()
