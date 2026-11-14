
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Example robot configurations.
# Values are joint angles in radians.

POSES = {
    "A": [0.0, 0.0, 0.0],
    "B": [1.0, -0.8, 0.6],
    "C": [-1.0, 0.8, -0.6],
}

X = []
y = []

# Generate example training data
rng = np.random.default_rng(42)

for letter, pose in POSES.items():

    pose = np.array(pose)

    for _ in range(400):

        noise = rng.normal(0, 0.1, 3)

        sample = pose + noise

        X.append(sample)
        y.append(letter)

X = np.array(X)
y = np.array(y)

# Create classifier
model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )
)

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "models/letter_classifier.joblib")

print("Classifier trained successfully!")

# Test
test_pose = [[0.05, -0.02, 0.01]]

prediction = model.predict(test_pose)

print("Predicted letter:", prediction[0])
