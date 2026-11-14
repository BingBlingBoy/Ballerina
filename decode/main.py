
import pybullet as p
import pybullet_data
import numpy as np
import joblib
import time

# Load trained classifier
model = joblib.load(
    "models/letter_classifier.joblib"
)

# Start simulation
physics = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

p.loadURDF("plane.urdf")

# Camera
p.resetDebugVisualizerCamera(
    cameraDistance=2.5,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0.5, 0, 0.2]
)

# Load robot
robot = p.loadURDF(
    "robot/robot.urdf",
    useFixedBase=True
)

# Find movable joints
joint_indices = []

for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)

    if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        joint_indices.append(i)

print("Robot loaded!")
print("Movable joints:", joint_indices)

# Joint configuration
current_pose = np.zeros(3)

# Joint limits from URDF
lower_limits = np.array([-2.8, -2.8, -2.8])
upper_limits = np.array([2.8, 2.8, 2.8])

STEP = 0.05

# Example poses
POSES = {
    ord("1"): np.array([0.0, 0.0, 0.0]),
    ord("2"): np.array([1.0, -0.8, 0.6]),
    ord("3"): np.array([-1.0, 0.8, -0.6])
}

# Keyboard joint controls
JOINT_KEYS = {
    ord("q"): (0, STEP),
    ord("a"): (0, -STEP),

    ord("w"): (1, STEP),
    ord("s"): (1, -STEP),

    ord("e"): (2, STEP),
    ord("d"): (2, -STEP)
}

print("""
CONTROLS
----------------------
1 = Pose A
2 = Pose B
3 = Pose C

Q/A = Joint 1 +/- 
W/S = Joint 2 +/-
E/D = Joint 3 +/-

R = Reset
ESC = Quit
----------------------
""")

def update_robot():
    global current_pose

    for index, angle in zip(joint_indices, current_pose):
        p.resetJointState(
            robot,
            index,
            float(angle)
        )

def get_skeleton():
    skeleton = []

    # Base position
    base = p.getBasePositionAndOrientation(robot)[0]
    skeleton.append(base)

    # Link positions
    for link_index in range(4):
        state = p.getLinkState(
            robot,
            link_index,
            computeForwardKinematics=True
        )

        skeleton.append(state[4])

    return np.asarray(skeleton)

last_prediction_time = 0

while True:

    keys = p.getKeyboardEvents()

    for key, state in keys.items():

        if not (state & p.KEY_WAS_TRIGGERED):
            continue

        # Preset poses
        if key in POSES:
            current_pose = POSES[key].copy()

        # Individual joint control
        elif key in JOINT_KEYS:
            joint, change = JOINT_KEYS[key]
            current_pose[joint] += change

        # Reset
        elif key == ord("r"):
            current_pose = np.zeros(3)

        # Quit
        elif key == 27:
            p.disconnect()
            raise SystemExit

    # Enforce joint limits
    current_pose = np.clip(
        current_pose,
        lower_limits,
        upper_limits
    )

    # Apply joint positions
    update_robot()

    p.stepSimulation()

    # Extract actual joint angles
    joint_angles = []

    for index in joint_indices:
        state = p.getJointState(robot, index)
        joint_angles.append(state[0])

    # Reconstruct skeleton
    skeleton = get_skeleton()

    # Classification
    prediction = model.predict([joint_angles])[0]

    probabilities = model.predict_proba([joint_angles])[0]

    confidence = np.max(probabilities)

    # Print status
    print(
        "\rAngles:",
        np.round(joint_angles, 2),
        "| Prediction:",
        prediction,
        "| Confidence:",
        round(confidence, 3),
        end="          "
    )

    # Print skeleton occasionally
    if time.time() - last_prediction_time > 2:
        print("\nSkeleton coordinates:")
        print(np.round(skeleton, 3))
        last_prediction_time = time.time()

    time.sleep(1 / 30)
