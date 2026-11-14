import unittest

import cv2
import numpy as np
import pandas as pd

from main import PredictionSmoother, predict_letter
from robot_vision.robot_pose_features import (
    LETTER_DATASET_COLUMNS,
    LETTER_LABELS,
    MARKER_IDS,
    MARKER_SIZE_METERS,
    POSE_FEATURE_COLUMNS,
    estimate_marker_poses,
    extract_features,
    extract_pose_features,
)
from robot_vision.collect_robot_data import label_from_key
from train_classifier import split_by_session, train_model


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
    def test_base_marker_is_required(self):
        poses = pose_set()
        del poses[MARKER_IDS["base"]]
        self.assertIsNone(extract_pose_features(poses))

    def test_each_hand_marker_is_required_for_whole_pose_features(self):
        poses = pose_set()
        del poses[MARKER_IDS["left_hand"]]
        self.assertIsNone(extract_pose_features(poses))

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
    def test_every_letter_key_is_a_label_and_escape_is_not(self):
        for label in LETTER_LABELS:
            self.assertEqual(label_from_key(ord(label.lower())), label)
            self.assertEqual(label_from_key(ord(label)), label)
        self.assertIsNone(label_from_key(27))
        self.assertEqual(label_from_key(ord("q")), "Q")

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
        for label_index, label in enumerate(LETTER_LABELS):
            for session_index in range(4):
                capture_id = f"{label}-{session_index}"
                for sample_index in range(4):
                    bits = np.array(
                        [(label_index >> bit) & 1 for bit in range(5)] * 5
                    )[: len(POSE_FEATURE_COLUMNS)]
                    values = bits * 4.0
                    values = values + session_index * 0.002 + sample_index * 0.0001
                    rows.append(
                        {
                            "session_id": f"session-{session_index}",
                            "capture_id": capture_id,
                            "timestamp_utc": f"2026-01-01T00:00:{sample_index:02d}+00:00",
                            "label": label,
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
                    rows.append(
                        {
                            "session_id": f"session-{session_index}",
                            "capture_id": capture_id,
                            "timestamp_utc": f"2026-01-01T00:00:{sample_index:02d}+00:00",
                            "label": label,
                            **dict(zip(POSE_FEATURE_COLUMNS, values)),
                        }
                    )

        dataset = pd.DataFrame(rows, columns=LETTER_DATASET_COLUMNS)
        bundle = train_model(dataset, ("a", "b"))
        self.assertEqual(bundle["classes"], ("A", "B"))
        self.assertEqual(tuple(bundle["model"].classes_), ("A", "B"))


if __name__ == "__main__":
    unittest.main()
