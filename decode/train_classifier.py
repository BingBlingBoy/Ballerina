# import os
# import joblib
# import pandas as pd

# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     accuracy_score,
# )

# # =====================================
# # LOAD DATASET
# # =====================================


# DATASET_FILE = "./robot_vision/collect_robot_data.py"

# df = pd.read_csv(DATASET_FILE)

# print(f"Loaded {len(df)} samples")

# # =====================================
# # SPLIT FEATURES / LABELS
# # =====================================

# X = df.drop(columns=["label"])
# y = df["label"]

# print("Feature shape:", X.shape)
# print("Classes:", sorted(y.unique()))

# # =====================================
# # TRAIN / TEST SPLIT
# # =====================================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # =====================================
# # CREATE MODEL
# # =====================================

# model = make_pipeline(
#     StandardScaler(),
#     SVC(
#         kernel="rbf",
#         C=10,
#         gamma="scale",
#         probability=True
#     )
# )

# # =====================================
# # TRAIN
# # =====================================

# print("Training...")

# model.fit(
#     X_train,
#     y_train
# )

# print("Training complete")

# # =====================================
# # EVALUATE
# # =====================================

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(
#     y_test,
#     y_pred
# )

# print()
# print("Accuracy:")
# print(f"{accuracy:.4f}")

# print()
# print("Classification Report:")
# print(
#     classification_report(
#         y_test,
#         y_pred
#     )
# )

# print()
# print("Confusion Matrix:")
# print(
#     confusion_matrix(
#         y_test,
#         y_pred
#     )
# )

# # =====================================
# # SAVE MODEL
# # =====================================

# os.makedirs(
#     "models",
#     exist_ok=True
# )

# joblib.dump(
#     model,
#     "models/letter_classifier.joblib"
# )

# print()
# print(
#     "Model saved to:"
# )
# print(
#     "models/letter_classifier.joblib"
# )

# # =====================================
# # QUICK TEST
# # =====================================

# sample = X.iloc[[0]]

# prediction = model.predict(
#     sample
# )

# probabilities = model.predict_proba(
#     sample
# )

# confidence = probabilities.max()

# print()
# print(
#     f"Example prediction: "
#     f"{prediction[0]}"
# )

# print(
#     f"Confidence: "
#     f"{confidence:.2%}"
# )

import os
import joblib
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("./robot_vision/robot_pose_dataset.csv")

# Keep only A and B

df = df[df["label"].isin(["A", "B"])]

print("Samples:", len(df))
print("Classes:", df["label"].unique())

# =====================================
# FEATURES / LABELS
# =====================================

X = df.drop(columns=["label"])
y = df["label"]

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# MODEL
# =====================================

model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )
)

# =====================================
# TRAIN
# =====================================

print("Training...")

model.fit(
    X_train,
    y_train
)

print("Training complete")

# =====================================
# EVALUATE
# =====================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy:.4f}")

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================
# SAVE
# =====================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/letter_classifier.joblib"
)

print(
    "\nSaved model to "
    "models/letter_classifier.joblib"
)

# =====================================
# TEST ON FIRST SAMPLE
# =====================================

sample = X.iloc[[0]]

prediction = model.predict(sample)[0]

confidence = max(
    model.predict_proba(sample)[0]
)

print(
    f"\nPrediction: "
    f"{prediction}"
)

print(
    f"Confidence: "
    f"{confidence:.2%}"
)