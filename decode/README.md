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
orientation. All six markers must be visible to collect a whole-body sample.
For a cardboard prototype, use a stationary base, a torso that rotates relative
to it, and independently movable head and arms. The camera must see all markers
throughout the test. This verifies the software path; poses collected from a
mock-up may not generalize to a finished robot.

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

   Hold one complete pose still and press its A-Z label. Every letter is
   available, including `Q`; **Esc** exits. Each label capture records up to 20
   valid observations at approximately 10 Hz. If the screen says to show all
   markers, adjust the setup until markers 0-5 are visible and their poses pass
   validation.

   For every letter, record several distinct poses/captures across multiple
   collector runs. The trainer requires that **each of the 26 labels appears in
   at least two separate sessions** and holds out entire sessions for evaluation.
   Prefer at least five sessions per label. Repeated frames of one pose are not
   substitutes for varied examples.

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

   The selected subset is stored with the model, and live inference uses exactly
   those classes. Training prints session-held-out metrics and fails with a list
   of selected labels needing more data. Each selected letter must appear in at
   least two separate collector sessions. The dataset is written to
   `robot_vision\robot_pose_letter_dataset.csv`; it is separate from any
   pre-existing joint-state dataset.

4. Run live whole-body pose inference:

   ```powershell
   python main.py
   ```

   The live result uses a five-frame voting window and a confidence threshold.
   Missing or invalid marker poses show as unknown. Press **Esc** to quit.

## Implementation notes

Collection and live inference share feature extraction in
[`robot_vision/robot_pose_features.py`](./robot_vision/robot_pose_features.py).
The 24-value vector combines the independently estimated arm, torso, and head
features. Torso orientation is measured relative to fixed base marker 5; head
orientation is relative to the torso. Pose estimates with duplicate marker IDs,
failed solves, or excessive reprojection error are rejected. Camera calibration
is stored locally and ignored by Git.

The model is a calibrated RBF SVM on 24 features. Its inference and memory costs
are modest on CPU; image-based ArUco detection is the dominant cost. Measure
frame rate and latency on the target machine. Validate classifications against
ground truth before using them to control hardware.
