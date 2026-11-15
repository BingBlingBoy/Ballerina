# CPU-only A-Z whole-body pose pipeline

The collector records **static whole-body poses** under labels A-Z. Pressing a
letter assigns that label to the current pose; it does not move the robot or
determine motion direction. For example, press `A` while holding the pose you
want to label A. `Q` is a valid class label. Press **Esc** to quit.

The camera uses the ArUco `DICT_4X4_50` dictionary and marker IDs 0-5:

| ID | Mounting |
|---|---|
| 0 | Head |
| 1 | Left shoulder |
| 2 | Right shoulder |
| 3 | Left hand |
| 4 | Right hand |
| 5 | Stationary base/reference; must not move with the torso |

Use 40 mm square markers, mounted rigidly and with consistent printed-face
orientation. Collect samples with the marker visibility patterns you expect in
operation. Complete-visibility samples are useful as a baseline, but are not
required to train a supported partial-visibility model.
For a cardboard prototype, use a stationary base, a torso that rotates relative
to it, and independently movable head and arms. The camera must see all markers
for complete captures. Live decoding and collection can use partial
observations; both report valid marker poses and observable feature groups.
This verifies the software path; poses collected from a mock-up may not
generalize to a finished robot.

## Record, train, and run

Run commands from the `decode` directory.

1. Calibrate the webcam using the checkerboard. The pictured board has 9 by 6
   squares, or 8 by 5 inner corners; each square is 1.5 cm:

   ```powershell
   python robot_vision\calibrate_camera.py --columns 8 --rows 5 --square-size 0.015 --samples 15
   ```

   Show the entire flat board at different positions and angles. Press `c` for
   each detected view. Capture at least 10 views; 15 or more varied views are
   recommended. Use the same camera, resolution, focus, and aspect ratio when
   collecting and running.

2. Arrange the mock robot/markers and start recording:

   ```powershell
   python robot_vision\collect_robot_data.py
   ```

   Hold a settled pose and press its A-Z label. Every letter is available,
   including `Q`; **Esc** exits. Each label capture records up to 20 samples at
   approximately 10 Hz when at least two body parts are observable. The CSV
   stores valid marker IDs and an explicit observed-feature mask for every row;
   unavailable feature values are left blank, never replaced with zero.

   For every letter, record captures under each visibility pattern you want the
   decoder to support, across multiple collector runs. Complete-visibility
   captures are useful as a baseline but are optional. Keep the pose label
   meaningful from the visible evidence; an occluded, independently moving
   joint cannot be learned from unrelated markers. For each classifier, the
   trainer can combine samples with different visibility masks as long as each
   sample directly contains the classifier's required feature groups. Every
   selected label still needs enough compatible samples across at least two
   sessions, and a session-held-out split must contain every label. Feature
   combinations without sufficient data are reported as skipped and the live
   decoder returns unknown for them. Prefer at least five sessions per label.
   Repeated frames of one pose are not substitutes for varied captures.

   Each key represents a static pose class. If the target task is dynamic sign
   recognition or motion recognition, one frame and one letter key are not
   enough; collect time-stamped sequences with explicit movement labels and use
   a temporal model instead.

3. Train the CPU-friendly classifier:

   ```powershell
   python train_classifier.py
   ```

   By default, training includes all A-Z. To train only on A and B:

   ```powershell
   python train_classifier.py --letters A B
   ```

   For a quick A/B/C subset, use the individual flags:

   ```powershell
   python train_classifier.py --a --b --c
   ```

   Any combination is supported, such as `python train_classifier.py --a
   --c`. Use either the letter flags or `--letters`, not both.

   The selected subset is stored with the model, and live inference uses exactly
   those classes. Training prints session-held-out accuracy separately for each
   visibility pattern it can train, and reports which patterns were skipped.
   At least one visibility pattern must have enough examples for every selected
   label; complete visibility is not required. The dataset is written to
   `robot_vision\robot_pose_letter_dataset.csv`; it is
   separate from any pre-existing joint-state dataset. A dataset from the
   previous schema must be moved aside or recollected; the collector will not
   overwrite it.

4. Run live whole-body pose inference:

   ```powershell
   python main.py
   ```

   The live result uses a visibility-specific classifier, a five-frame voting
   window, and a confidence threshold. It can classify from two or more
   observable body parts (arms, torso, or head); a missing feature group does
   not invalidate the others. Torso geometry and orientation count as one body
   part. Frames with fewer than two observable body parts show as unknown.
   Confidence is reduced for fewer observed body parts, and the window reports
   marker IDs with valid poses and the feature groups used. Press **Esc** to
   quit.

## Implementation notes

Collection and live inference share feature extraction in
[`robot_vision/robot_pose_features.py`](./robot_vision/robot_pose_features.py).
The 24-value vector keeps the independently estimated arm, torso, and head
features separate. Missing features are represented as unavailable, not zero:
arm features need that arm's shoulder and hand; torso orientation needs the
base and at least one shoulder; torso geometry needs both shoulders and the
base; head features need the head and both shoulders. The trainer fits a small
calibrated CPU classifier for feature combinations supported by the collected
data. It can pool samples from different visibility masks when they all
directly observe the model's required features; it never imputes a missing
feature. One absent joint therefore does not discard other visible evidence,
while unsupported feature combinations remain unknown.
Pose estimates with duplicate marker IDs, failed solves, or excessive
reprojection error are rejected. Camera calibration is stored locally and
ignored by Git.

The model bundle contains calibrated RBF SVMs for the supported feature
combinations. Inference selects one model per frame; image-based ArUco
detection remains the dominant CPU cost. Confidence combines the selected
model's calibrated
probability with a conservative feature-coverage factor; it does not replace
validation against occlusion and image-quality conditions. Measure frame rate
and latency on the target machine, and validate classifications before using
them to control hardware. A-Z labels remain static-pose classes, not motion
direction labels.
