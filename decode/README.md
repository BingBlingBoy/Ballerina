# CPU-only A-Z hand/head pose pipeline

The collector records **static hand/head poses** under labels A-Z. Pressing a
letter assigns that label to the current pose; it does not move the robot or
determine motion direction. For example, press `A` while holding the pose you
want to label A. `Q` is a valid class label. Press **Esc** to quit.

The camera uses the ArUco `DICT_4X4_50` dictionary. The decoder uses these
markers:

| ID | Mounting |
|---|---|
| 0 | Head |
| 1 or 2 | Alternative binary signal markers; not pose markers |
| 3 | Left hand |
| 4 | Right hand |
| 5 | Stationary base/reference |

Use 40 mm square markers, mounted rigidly and with consistent printed-face
orientation. Collect samples with the marker visibility patterns you expect in
operation. IDs 1 and 2 are treated only as an alternative signal: ID 1 alone or
ID 2 alone is a valid signal state. If both or neither is detected, that signal
is unknown. Their positions are not used to estimate robot pose. Complete-pose
visibility is useful as a baseline, but is not required to train a supported
partial-visibility model.
Mount the red and blue LEDs at distinct, visible positions on the robot head.
Switch each LED on/off to provide two binary features alongside the pose.
Before recording, calibrate the LED locations relative to the head marker as
described below. The LEDs must be on the same flat surface as that marker. The
software reads LED states but does not control the hardware.
The decoder estimates hand and head marker poses relative to the stationary
base. Without a torso marker, torso rotation cannot be isolated from arm motion,
and head pitch cannot be separated from torso rotation. These states are not
classified independently. Partial observations can still classify when at
least two of the left hand, right hand, and head features are available relative
to the base.

## Record, train, and run

Run commands from the `decode` directory.
Camera commands default to camera index `0`; select another camera with
`--camera <index>` (for example, `--camera 1`). Use the same camera index for
camera calibration, LED calibration, collection, and live inference. OpenCV
indices are device numbers, not camera names; try `0`, `1`, and so on if you are
unsure which number belongs to your camera.

1. Calibrate the webcam using the checkerboard. The pictured board has 9 by 6
   squares, or 8 by 5 inner corners; each square is 1.5 cm:

   ```powershell
   python robot_vision\calibrate_camera.py --camera 1 --columns 8 --rows 5 --square-size 0.015 --samples 15
   ```

   Show the entire flat board at different positions and angles. Press `c` for
   each detected view. Capture at least 10 views; 15 or more varied views are
   recommended. Use the same camera, resolution, focus, and aspect ratio when
   collecting and running.

2. Calibrate the LED locations, then start recording:

   ```powershell
   python robot_vision\calibrate_leds.py --camera 1
   ```

   Keep head marker ID 0 visible. Turn both LEDs on, click the center of the red
   LED followed by the blue LED, then collect camera-specific reference samples:
   turn both LEDs off and press **O**, turn only red on and press **R**, turn
   only blue on and press **B**, then turn both on and press **A**. Keep the
   robot still during each 20-frame capture. The calibration learns each LED's
   appearance on this camera, including both individual and simultaneous
   operation, instead of relying on fixed hue thresholds. Press **Enter** to
   save, **X** to redo all steps, or **Esc** to cancel. The saved calibration
   tracks the sample regions as the head marker moves and tilts. Repeat
   calibration if either LED, camera, or head marker is moved, or if camera
   exposure/white balance changes substantially.

   Start the collector:

   ```powershell
   python robot_vision\collect_robot_data.py --camera 1
   ```

   Hold a settled pose and press its A-Z label. To teach the repeat command,
   hold its own distinct pose and press `~`; to teach the word boundary, hold
   another distinct pose and press **Space**. These train the special `~` and
   `SPACE` classes. Every letter is available, including `Q`; **Esc** exits.
   Each label capture records up to 20 samples at approximately 10 Hz when at
   least two hand/head features are observable. The CSV
   stores valid marker IDs and an explicit observed-feature mask for every row;
   unavailable pose feature values are left blank, never replaced with zero.
   The `marker_signal` field records `id1` or `id2` only when exactly one of
   those markers is detected; both/neither is stored as unknown. Samples are
   still retained based on pose visibility when the signal is unknown.
   It also stores red and blue LED states when the calibrated head-marker
   regions are available. Confirm that both on-screen states change as you
   switch each light. If the head marker is missing, too small, or uncalibrated,
   LED status is **UNKNOWN** and the sample is retained as pose-only data.

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

   To train a selected subset from A through H, use the corresponding flags.
   For example, to train A-H:

   ```powershell
   python train_classifier.py --a --b --c --d --e --f --g --h
   ```

   Flags can be combined in any subset, such as
   `python train_classifier.py --a --d --h`. Without flags, training still
   includes all A-Z. Add `--repeat` to include the `~` class and `--space` to
   include the word-boundary pose, for example
   `python train_classifier.py --a --b --repeat --space`. These options can
   also be combined with `--letters`, such as
   `python train_classifier.py --letters A B --repeat --space`. Use either
   A-H flags or `--letters`, not both.

   The selected subset is stored with the model, and live inference uses exactly
   those classes. Training prints session-held-out accuracy separately for each
   pose-visibility pattern and optional LED/ID 1-or-2 signal combination it can
   train, and reports which patterns were skipped.
   At least one visibility pattern must have enough examples for every selected
   label; complete visibility is not required. The dataset is written to
   `robot_vision\robot_pose_letter_dataset_no_shoulders.csv`. It is separate
   from the previous shoulder-based dataset and any pre-existing joint-state
   dataset. Older datasets without the `marker_signal` column are incompatible;
   the collector leaves them unchanged and asks you to move them aside before
   starting a fresh collection.

4. Run live hand/head pose inference:

   ```powershell
   python main.py --camera 1
   ```

   The live result uses a visibility-specific classifier and a 50% confidence
   threshold. Predictions above 50% appear green;
   lower-confidence predictions still show their best-guess letter as uncertain.
   The decoded word is not drawn on screen. Focus the camera window and press
   `#` (usually **Shift+3**) to record the currently predicted class. Pressing
   `#` again records another copy of a letter. The trained `~` pose records a
   repeat marker after a letter (`A~` represents `AA`); it must also be captured
   with `#`. Press **Enter** to print the completed word to the console and
   clear the current buffer for the next word. Press `!` to print the current
   word with `!` and end the message. The `SPACE` pose no longer ends a word.
   The decoder can classify from two or more
   observable features (left hand, right hand, or head relative to the base); a
   missing feature does not invalidate the others. When exactly one of ID 1 or
   ID 2 is visible, the decoder uses a classifier trained with that signal when
   one is available; if both or neither is visible, it can fall back to a
   pose/LED-only classifier. Frames with fewer than two observable pose
   features show as unknown.
   Confidence is reduced for fewer observed features, and the window reports
   marker IDs with valid poses and the feature groups used. Press **Esc** to
   quit.

## Implementation notes

Collection and live inference share feature extraction in
[`robot_vision/robot_pose_features.py`](./robot_vision/robot_pose_features.py).
The 18-value vector contains left hand, right hand, and head poses relative to
the stationary base marker. Each feature needs its marker and the base marker.
Missing features are represented as unavailable, not zero. The trainer fits a
small calibrated CPU classifier for feature combinations supported by the
collected data. It can pool samples from different visibility masks when they
all directly observe the model's required features; it never imputes a missing
feature. One absent marker therefore does not discard other visible evidence,
while unsupported feature combinations remain unknown.
IDs 1 and 2 bypass pose estimation and contribute a categorical signal feature
only when exactly one ID is detected. If both or neither is detected, that
feature is unknown; inference can still use a model trained without it.
Pose estimates with duplicate marker IDs, failed solves, or excessive
reprojection error are rejected. Camera calibration is stored locally and
ignored by Git.

The model bundle contains calibrated RBF SVMs for supported pose-feature
combinations, with LED-aware, ID-signal-aware, and pose-only variants when data
supports them. Both/neither ID signal observations are excluded from
ID-signal-aware training. Inference prefers models that use available LED and
ID signal evidence and falls back to trained models without those optional
signals. Only rows recorded by the calibrated head-marker ROI detector are
used to train LED-aware models; older LED readings remain available in the CSV
but are ignored for LED training. An unlit LED and an occluded or undetected
LED still look alike within its expected region, so keep the LEDs unobstructed
while recording and running. Inference selects one model per frame; image-based
ArUco detection remains the dominant CPU cost.
Confidence combines the selected model's calibrated
probability with a conservative feature-coverage factor; it does not replace
validation against occlusion and image-quality conditions. Measure frame rate
and latency on the target machine, and validate classifications before using
them to control hardware. A-Z labels remain static-pose classes, not motion
direction labels.
