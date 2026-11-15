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
    MARKER_SIGNAL_COLUMN,
    MARKER_SIGNAL_FEATURE_COLUMNS,
    MIN_CLASSIFICATION_PARTS,
    MODEL_SCHEMA_VERSION,
    MARKER_IDS,
    POSE_FEATURE_COLUMNS,
    POSE_FEATURE_GROUPS,
    POSE_MASK_COLUMNS,
    POSE_CLASS_LABELS,
    REPEAT_LABEL,
    SPACE_LABEL,
    SUPPORTED_GROUP_MASKS,
    feature_indices_for_mask,
    feature_group_mask_for_marker_ids,
    observable_part_count,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_FILE = (
    SCRIPT_DIR / "robot_vision" / "robot_pose_letter_dataset_no_shoulders.csv"
)
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


def _validate_dataset_schema(dataset):
    if tuple(dataset.columns) != LETTER_DATASET_COLUMNS:
        raise ValueError(
            "Dataset schema does not match the no-shoulder A-Z pose pipeline. "
            "Collect fresh data with robot_vision/collect_robot_data.py."
        )


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
                f"Dataset row {row_index} has fewer than two observable features."
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
    _validate_dataset_schema(dataset)

    requested_letters = tuple(
        dict.fromkeys(str(letter).strip().upper() for letter in selected_letters)
    )
    if not requested_letters:
        raise ValueError("Select at least one letter to train.")
    invalid_letters = [
        letter for letter in requested_letters if letter not in POSE_CLASS_LABELS
    ]
    if invalid_letters:
        raise ValueError(
            f"Invalid letter selection: {', '.join(invalid_letters)}."
        )
    selected_set = set(requested_letters)
    selected_letters = tuple(
        letter for letter in POSE_CLASS_LABELS if letter in selected_set
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
    marker_signals = (
        dataset[MARKER_SIGNAL_COLUMN].astype("string").fillna("").str.strip()
    )
    if not marker_signals.isin(("", "id1", "id2")).all():
        raise ValueError("Marker signal values must be blank, id1, or id2.")
    marker_signal_features = np.column_stack(
        (
            marker_signals.eq("id1").to_numpy(dtype=np.float64),
            marker_signals.eq("id2").to_numpy(dtype=np.float64),
        )
    )
    marker_signal_valid = marker_signals.isin(("id1", "id2")).to_numpy()
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
            for use_marker_signal in (False, True):
                matching_rows = np.array(
                    [
                        _contains_feature_groups(observed_mask, group_mask)
                        and (not use_leds or is_led_valid)
                        and (not use_marker_signal or has_marker_signal)
                        for observed_mask, is_led_valid, has_marker_signal in zip(
                            group_masks,
                            valid_led_rows,
                            marker_signal_valid,
                        )
                    ],
                    dtype=bool,
                )
                pattern_features = np.column_stack(
                    (
                        features[matching_rows],
                        led_features[matching_rows],
                        marker_signal_features[matching_rows],
                    )
                )
                pattern_labels = labels.loc[matching_rows].reset_index(drop=True)
                pattern_groups = groups.loc[matching_rows].reset_index(drop=True)
                pattern_components = [_format_group_mask(group_mask)]
                if use_leds:
                    pattern_components.append("red/blue LEDs")
                if use_marker_signal:
                    pattern_components.append("ID 1/2 signal")
                pattern_name = " + ".join(pattern_components)
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
                    label for label in selected_letters
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
                    feature_indices = (
                        *feature_indices,
                        *range(
                            len(POSE_FEATURE_COLUMNS),
                            len(POSE_FEATURE_COLUMNS) + len(LED_FEATURE_COLUMNS),
                        ),
                    )
                if use_marker_signal:
                    feature_start = (
                        len(POSE_FEATURE_COLUMNS) + len(LED_FEATURE_COLUMNS)
                    )
                    feature_indices = (
                        *feature_indices,
                        *range(
                            feature_start,
                            feature_start + len(MARKER_SIGNAL_FEATURE_COLUMNS),
                        ),
                    )
                mask_model = _fit_classifier(
                    pattern_features[train_indices][:, feature_indices],
                    pattern_labels.iloc[train_indices],
                )
                models[(group_mask, use_leds, use_marker_signal)] = mask_model
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
        f"{4 * len(SUPPORTED_GROUP_MASKS)} supported"
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
        description="Train the CPU-friendly hand/head pose letter classifier."
    )
    parser.add_argument(
        "--letters",
        nargs="+",
        type=str.upper,
        choices=POSE_CLASS_LABELS,
        default=None,
        metavar="LETTER",
        help="Classes to train, including ~ for repeat (default: A-Z), e.g. --letters A B.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Include the dedicated ~ repeat-pose class with selected letters.",
    )
    parser.add_argument(
        "--space",
        action="store_true",
        help="Include the dedicated Space word-boundary pose class.",
    )
    for label in "ABCDEFGH":
        parser.add_argument(
            f"--{label.lower()}",
            action="store_true",
            help=f"Include the {label} class when training (combine with other letter flags).",
        )
    args = parser.parse_args()
    if args.letters is not None and any(
        getattr(args, label.lower()) for label in "ABCDEFGH"
    ):
        parser.error("use either --letters or individual A-H flags")
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
            for label in "ABCDEFGH"
            if getattr(args, label.lower())
        ]
        if not selected_letters:
            selected_letters = LETTER_LABELS
    if args.repeat:
        selected_letters = (*selected_letters, REPEAT_LABEL)
    if args.space:
        selected_letters = (*selected_letters, SPACE_LABEL)
    bundle = train_model(pd.read_csv(DATASET_FILE), selected_letters)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    print(
        f"Saved pose model for {', '.join(bundle['classes'])} to {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()
