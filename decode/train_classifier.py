import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from robot_vision.robot_pose_features import (
    LETTER_DATASET_COLUMNS,
    LETTER_LABELS,
    MODEL_SCHEMA_VERSION,
    POSE_FEATURE_COLUMNS,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_FILE = SCRIPT_DIR / "robot_vision" / "robot_pose_letter_dataset.csv"
MODEL_FILE = SCRIPT_DIR / "models" / "pose_letter_classifier.joblib"


def split_by_session(features, labels, groups, selected_letters):
    session_counts = pd.DataFrame(
        {"label": labels.to_numpy(), "session": groups.to_numpy()}
    ).groupby("label")["session"].nunique()
    missing_sessions = [
        label for label in selected_letters if session_counts.get(label, 0) < 2
    ]
    if missing_sessions:
        raise ValueError(
            "Each letter needs samples in at least two separate collector "
            f"sessions. Insufficient sessions for: {', '.join(missing_sessions)}."
        )

    expected_labels = set(selected_letters)
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
    for train_indices, test_indices in splitter.split(features, labels, groups=groups):
        if (
            set(labels.iloc[train_indices]) == expected_labels
            and set(labels.iloc[test_indices]) == expected_labels
        ):
            return train_indices, test_indices

    raise ValueError(
        "Could not create a session-independent split containing every letter in "
        "both partitions. Record more letters in multiple separate sessions."
    )


def train_model(dataset, selected_letters=LETTER_LABELS):
    if tuple(dataset.columns) != LETTER_DATASET_COLUMNS:
        raise ValueError(
            "Dataset schema does not match the A-Z pose pipeline. "
            "Record data with robot_vision/collect_robot_data.py."
        )

    requested_letters = tuple(
        dict.fromkeys(str(letter).strip().upper() for letter in selected_letters)
    )
    if not requested_letters:
        raise ValueError("Select at least one letter to train.")
    invalid_letters = [
        letter for letter in requested_letters if letter not in LETTER_LABELS
    ]
    if invalid_letters:
        raise ValueError(
            f"Invalid letter selection: {', '.join(invalid_letters)}."
        )
    selected_set = set(requested_letters)
    selected_letters = tuple(
        letter for letter in LETTER_LABELS if letter in selected_set
    )

    dataset = dataset.copy()
    dataset["label"] = dataset["label"].astype(str).str.upper()
    dataset = dataset.loc[dataset["label"].isin(selected_letters)]
    dataset.loc[:, POSE_FEATURE_COLUMNS] = dataset.loc[
        :, POSE_FEATURE_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")
    dataset = dataset.dropna(
        subset=[*POSE_FEATURE_COLUMNS, "capture_id", "session_id"]
    )

    observed_labels = set(dataset["label"])
    missing_labels = [
        label for label in selected_letters if label not in observed_labels
    ]
    if missing_labels:
        raise ValueError(
            f"No valid samples recorded for: {', '.join(missing_labels)}."
        )

    features = dataset.loc[:, POSE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    labels = dataset["label"]
    groups = dataset["session_id"]
    train_indices, test_indices = split_by_session(
        features,
        labels,
        groups,
        selected_letters,
    )

    training_counts = labels.iloc[train_indices].value_counts()
    insufficient_training = [
        label for label in selected_letters if training_counts.get(label, 0) < 3
    ]
    if insufficient_training:
        raise ValueError(
            "Each training letter needs at least three valid samples for "
            f"probability calibration. Insufficient samples for: {', '.join(insufficient_training)}."
        )

    model = CalibratedClassifierCV(
        estimator=make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
            ),
        ),
        method="sigmoid",
        cv=3,
    )
    model.fit(features[train_indices], labels.iloc[train_indices])
    predictions = model.predict(features[test_indices])

    print(
        f"Samples: {len(dataset)} across {groups.nunique()} recording sessions"
    )
    print(
        f"Session-held-out accuracy: "
        f"{accuracy_score(labels.iloc[test_indices], predictions):.3f}"
    )
    print(
        classification_report(
            labels.iloc[test_indices],
            predictions,
            labels=selected_letters,
            zero_division=0,
        )
    )

    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "classes": selected_letters,
        "model": model,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CPU-friendly whole-body pose letter classifier."
    )
    parser.add_argument(
        "--letters",
        nargs="+",
        type=str.upper,
        choices=LETTER_LABELS,
        default=LETTER_LABELS,
        metavar="LETTER",
        help="Letters to include (default: all A-Z), e.g. --letters A B.",
    )
    return parser.parse_args()


def main():
    if not DATASET_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_FILE}. Record A-Z poses first."
        )
    args = parse_args()
    bundle = train_model(pd.read_csv(DATASET_FILE), args.letters)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(
        f"Saved pose model for {', '.join(bundle['classes'])} to {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()
