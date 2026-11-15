import csv
from io import StringIO
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from main import PredictionSmoother, predict_letter
from robot_vision.robot_pose_features import (
    LETTER_DATASET_COLUMNS,
    LETTER_LABELS,
    MARKER_IDS,
    MARKER_SIZE_METERS,
    POSE_FEATURE_GROUPS,
    POSE_FEATURE_COLUMNS,
    POSE_MASK_COLUMNS,
    estimate_marker_poses,
    extract_features,
    extract_pose_observation,
    extract_pose_features,
)
from robot_vision.collect_robot_data import label_from_key, write_sample
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
        self.assertEqual(row[POSE_FEATURE_COLUMNS[-1]], "nan")

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
            bundle["model"], features, partial_mask
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
        self.assertIn(uncollected_mask, bundle["model"])

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
            tuple(bundle["model"][partial_mask].classes_),
            ("A", "B"),
        )
        feature_values = dataset.loc[0, POSE_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )
        partial_label, partial_confidence = predict_letter(
            bundle["model"], feature_values, partial_mask
        )
        self.assertIn(partial_label, ("A", "B"))
        self.assertGreaterEqual(partial_confidence, 0.60)


if __name__ == "__main__":
    unittest.main()
