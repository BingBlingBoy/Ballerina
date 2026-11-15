import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from robot_vision.robot_pose_features import (
    LETTER_DATASET_COLUMNS,
    LETTER_LABELS,
    LED_DETECTOR_COLUMN,
    LED_DETECTOR_VERSION,
    LED_FEATURE_COLUMNS,
    LED_VALID_COLUMN,
    LEGACY_LETTER_DATASET_COLUMNS,
    PREVIOUS_LETTER_DATASET_COLUMNS,
    MIN_CLASSIFICATION_PARTS,
    MODEL_SCHEMA_VERSION,
    MARKER_IDS,
    POSE_FEATURE_COLUMNS,
    POSE_FEATURE_GROUPS,
    POSE_MASK_COLUMNS,
    SUPPORTED_GROUP_MASKS,
    feature_indices_for_mask,
    feature_group_mask_for_marker_ids,
    observable_part_count,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_FILE = SCRIPT_DIR / "robot_vision" / "robot_pose_letter_dataset.csv"
MODEL_FILE = SCRIPT_DIR / "models" / "pose_letter_classifier.joblib"


def _fit_classifier(features, labels):
    classifier = CalibratedClassifierCV(
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
    classifier.fit(features, labels)
    return classifier


def _format_group_mask(group_mask):
    observed = [
        name
        for name, enabled in zip(POSE_FEATURE_GROUPS, group_mask)
        if enabled
    ]
    return ", ".join(observed) if observed else "no observable features"


def _contains_feature_groups(observed_mask, required_mask):
    return all(
        not required or observed
        for observed, required in zip(observed_mask, required_mask)
    )


def _normalize_dataset_schema(dataset):
    dataset = dataset.copy()
    if tuple(dataset.columns) == LEGACY_LETTER_DATASET_COLUMNS:
        dataset[LED_FEATURE_COLUMNS[0]] = np.nan
        dataset[LED_FEATURE_COLUMNS[1]] = np.nan
        dataset[LED_VALID_COLUMN] = 0
        dataset[LED_DETECTOR_COLUMN] = ""
        dataset = dataset.loc[:, LETTER_DATASET_COLUMNS]
    elif tuple(dataset.columns) == PREVIOUS_LETTER_DATASET_COLUMNS:
        dataset[LED_DETECTOR_COLUMN] = ""
        dataset = dataset.loc[:, LETTER_DATASET_COLUMNS]
    elif tuple(dataset.columns) != LETTER_DATASET_COLUMNS:
        raise ValueError(
            "Dataset schema does not match the A-Z pose pipeline. "
            "Record data with robot_vision/collect_robot_data.py."
        )
    return dataset


def _validate_observation_rows(dataset, features):
    marker_id_values = set(MARKER_IDS.values())
    group_columns = tuple(POSE_MASK_COLUMNS.values())
    raw_masks = dataset.loc[:, group_columns].apply(pd.to_numeric, errors="coerce")
    if raw_masks.isna().any().any() or not raw_masks.isin((0, 1)).all().all():
        raise ValueError("Observation mask columns must contain only 0 or 1.")
    group_masks = tuple(
        tuple(bool(value) for value in row)
        for row in raw_masks.to_numpy()
    )

    for row_index, (raw_marker_ids, values, group_mask) in enumerate(
        zip(dataset["visible_marker_ids"], features, group_masks)
    ):
        marker_text = str(raw_marker_ids).strip()
        try:
            marker_ids = (
                tuple(int(value) for value in marker_text.split(","))
                if marker_text
                else ()
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid visible_marker_ids at dataset row {row_index}."
            ) from error
        if (
            len(marker_ids) != len(set(marker_ids))
            or not set(marker_ids).issubset(marker_id_values)
        ):
            raise ValueError(
                f"Invalid visible_marker_ids at dataset row {row_index}."
            )
        possible_mask = feature_group_mask_for_marker_ids(marker_ids)
        if any(
            enabled and not possible
            for enabled, possible in zip(group_mask, possible_mask)
        ):
            raise ValueError(
                f"Observed feature mask requires an absent marker at dataset "
                f"row {row_index}."
            )
        if np.isinf(values).any():
            raise ValueError(f"Infinite feature value at dataset row {row_index}.")

        for enabled, indices in zip(group_mask, POSE_FEATURE_GROUPS.values()):
            group_values = values[list(indices)]
            if enabled and not np.isfinite(group_values).all():
                raise ValueError(
                    f"Observed feature group is incomplete at dataset row "
                    f"{row_index}."
                )
            if not enabled and np.isfinite(group_values).any():
                raise ValueError(
                    f"Unobserved feature group contains values at dataset row "
                    f"{row_index}."
                )
        if observable_part_count(group_mask) < MIN_CLASSIFICATION_PARTS:
            raise ValueError(
                f"Dataset row {row_index} has fewer than two observable body parts."
            )

    return group_masks


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
    dataset = _normalize_dataset_schema(dataset)

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
    dataset.loc[:, LED_FEATURE_COLUMNS] = dataset.loc[
        :, LED_FEATURE_COLUMNS
    ].apply(pd.to_numeric, errors="coerce")
    dataset[LED_VALID_COLUMN] = pd.to_numeric(
        dataset[LED_VALID_COLUMN],
        errors="coerce",
    )
    dataset = dataset.dropna(
        subset=["capture_id", "session_id", "visible_marker_ids"]
    ).reset_index(drop=True)

    observed_labels = set(dataset["label"])
    missing_labels = [
        label for label in selected_letters if label not in observed_labels
    ]
    if missing_labels:
        raise ValueError(
            f"No valid samples recorded for: {', '.join(missing_labels)}."
        )

    features = dataset.loc[:, POSE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    led_features = dataset.loc[:, LED_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    group_masks = _validate_observation_rows(dataset, features)
    led_valid = dataset[LED_VALID_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(led_valid).all() or not np.isin(led_valid, (0, 1)).all():
        raise ValueError("LED validity column must contain only 0 or 1.")
    stored_valid_led_rows = led_valid.astype(bool)
    detector_versions = (
        dataset[LED_DETECTOR_COLUMN].fillna("").astype(str).to_numpy()
    )
    valid_led_rows = stored_valid_led_rows & (
        detector_versions == LED_DETECTOR_VERSION
    )
    if np.isinf(led_features).any():
        raise ValueError("LED feature values cannot be infinite.")
    if not np.isfinite(led_features[stored_valid_led_rows]).all() or not np.isin(
        led_features[stored_valid_led_rows], (0, 1)
    ).all():
        raise ValueError("Valid LED states must contain red/blue binary values.")
    if np.isfinite(led_features[~stored_valid_led_rows]).any():
        raise ValueError("LED values must be blank when the LED state is invalid.")

    labels = dataset["label"]
    groups = dataset["session_id"]
    models = {}
    pattern_reports = []
    for group_mask in SUPPORTED_GROUP_MASKS:
        if observable_part_count(group_mask) < MIN_CLASSIFICATION_PARTS:
            continue
        for use_leds in (False, True):
            matching_rows = np.array(
                [
                    _contains_feature_groups(observed_mask, group_mask)
                    and (not use_leds or is_led_valid)
                    for observed_mask, is_led_valid in zip(
                        group_masks,
                        valid_led_rows,
                    )
                ],
                dtype=bool,
            )
            pattern_features = features[matching_rows]
            if use_leds and len(pattern_features):
                pattern_features = np.column_stack(
                    (pattern_features, led_features[matching_rows])
                )
            pattern_labels = labels.loc[matching_rows].reset_index(drop=True)
            pattern_groups = groups.loc[matching_rows].reset_index(drop=True)
            pattern_name = (
                f"{_format_group_mask(group_mask)} + red/blue LEDs"
                if use_leds
                else _format_group_mask(group_mask)
            )
            if not len(pattern_features):
                pattern_reports.append(
                    f"Skipped [{pattern_name}]: no compatible observations."
                )
                continue

            try:
                train_indices, test_indices = split_by_session(
                    pattern_features,
                    pattern_labels,
                    pattern_groups,
                    selected_letters,
                )
            except ValueError as error:
                pattern_reports.append(
                    f"Skipped [{pattern_name}]: {error}"
                )
                continue

            training_counts = pattern_labels.iloc[train_indices].value_counts()
            insufficient_training = [
                label
                for label in selected_letters
                if training_counts.get(label, 0) < 3
            ]
            if insufficient_training:
                pattern_reports.append(
                    f"Skipped [{pattern_name}]: fewer than three "
                    "training samples for "
                    f"{', '.join(insufficient_training)}."
                )
                continue

            feature_indices = feature_indices_for_mask(group_mask)
            if use_leds:
                feature_indices = (*feature_indices, *range(
                    len(POSE_FEATURE_COLUMNS),
                    len(POSE_FEATURE_COLUMNS) + len(LED_FEATURE_COLUMNS),
                ))
            mask_model = _fit_classifier(
                pattern_features[train_indices][:, feature_indices],
                pattern_labels.iloc[train_indices],
            )
            models[(group_mask, use_leds)] = mask_model
            predictions = mask_model.predict(
                pattern_features[test_indices][:, feature_indices]
            )
            score = accuracy_score(pattern_labels.iloc[test_indices], predictions)
            pattern_reports.append(
                f"Trained [{pattern_name}]: "
                f"{len(pattern_features)} compatible observations, "
                f"session-held-out accuracy {score:.3f}."
            )

    if not models:
        print(
            "No visibility pattern met the training requirements for all "
            "selected labels:"
        )
        for report in pattern_reports:
            print(report)
        raise ValueError(
            "No visibility pattern has enough data to train a classifier for "
            "every selected label. For at least one set of common feature "
            "groups, each selected label needs at least three training samples "
            "and samples in at least two collector sessions. Marker visibility "
            "patterns do not need to be identical."
        )

    print(f"Samples: {len(dataset)} across {groups.nunique()} recording sessions")
    print(
        f"Feature combinations: {len(models)} trained of "
        f"{2 * len(SUPPORTED_GROUP_MASKS)} supported"
    )
    for report in pattern_reports:
        print(report)

    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "classes": selected_letters,
        "model": models,
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
        default=None,
        metavar="LETTER",
        help="Letters to train (default: all A-Z), e.g. --letters A B.",
    )
    parser.add_argument(
        "--a",
        action="store_true",
        help="Train the A class (combine with --b and/or --c).",
    )
    parser.add_argument(
        "--b",
        action="store_true",
        help="Train the B class (combine with --a and/or --c).",
    )
    parser.add_argument(
        "--c",
        action="store_true",
        help="Train the C class (combine with --a and/or --b).",
    )
    args = parser.parse_args()
    if args.letters is not None and (args.a or args.b or args.c):
        parser.error("use either --letters or the individual --a/--b/--c flags")
    return args


def main():
    if not DATASET_FILE.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_FILE}. Record A-Z poses first."
        )
    args = parse_args()
    if args.letters is not None:
        selected_letters = args.letters
    else:
        selected_letters = [
            label
            for label in ("A", "B", "C")
            if getattr(args, label.lower())
        ]
        if not selected_letters:
            selected_letters = LETTER_LABELS
    bundle = train_model(pd.read_csv(DATASET_FILE), selected_letters)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(
        f"Saved pose model for {', '.join(bundle['classes'])} to {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()
