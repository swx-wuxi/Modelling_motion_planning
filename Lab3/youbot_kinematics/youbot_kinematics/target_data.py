import numpy as np

target_joint_pos_1 = [4.622576563079859, 1.4919956782391877, -2.4435069071962348, 2.0936591207981596, 0.9990974803253598]
target_joint_pos_2 = [1.4387632277949232, 0.7054776193039478, -2.512352196859098, 1.3866665704382877, 1.6211613228469561]
target_joint_pos_3 = [4.708484383458099, 1.3752175535418365, -3.214261824370574, 1.7887614596312122, 1.7302036080836274]
target_joint_pos_4 = [4.054210505693871, 0.7992699204278928, -2.288749607899941, 1.2692020100313526, 2.645584347531603]

# rows are target joint positions, columns are the joints
TARGET_JOINT_POSITIONS_ORIGINAL = np.array([
    target_joint_pos_1, target_joint_pos_2, target_joint_pos_3, target_joint_pos_4
])

# Q1(d): fixed, varied test data, in radians, ordered as arm_joint_1 ... arm_joint_5.
# Each supplied configuration is inside the joint limits in youbot.urdf.
# This alone does not guarantee that IK will solve every connecting line segment.

# Set 1: four alternative checkpoints, for comparison with the original task.
TARGET_JOINT_POSITIONS_SET_1 = np.array([
    [4.50, 1.42, -2.55, 2.00, 1.15],
    [1.55, 0.82, -2.60, 1.50, 1.80],
    [4.60, 1.28, -3.10, 1.90, 1.55],
    [3.95, 0.92, -2.40, 1.38, 2.50],
])

# Set 2: five checkpoints, also checking that planning supports more targets.
TARGET_JOINT_POSITIONS_SET_2 = np.array([
    [4.72, 1.55, -2.52, 2.15, 1.05],
    [1.62, 0.88, -2.65, 1.55, 1.90],
    [4.48, 1.30, -3.05, 1.85, 1.45],
    [3.85, 1.00, -2.45, 1.48, 2.35],
    [4.20, 1.10, -2.70, 1.65, 2.10],
])

# Set 3: six checkpoints, with further joint-angle and target-count changes.
TARGET_JOINT_POSITIONS_SET_3 = np.array([
    [4.58, 1.48, -2.62, 2.08, 0.95],
    [1.72, 0.90, -2.72, 1.62, 1.72],
    [4.82, 1.32, -3.22, 1.92, 1.85],
    [3.70, 0.96, -2.38, 1.42, 2.72],
    [4.08, 1.18, -2.82, 1.78, 2.25],
    [1.92, 1.02, -2.90, 1.82, 1.35],
])

# Change only this selector, rebuild the package, and restart the planner.
# Supported values: "original", "set_1", "set_2", "set_3".
ACTIVE_TARGET_SET = "set_3"
TARGET_DATASETS = {
    "original": TARGET_JOINT_POSITIONS_ORIGINAL,
    "set_1": TARGET_JOINT_POSITIONS_SET_1,
    "set_2": TARGET_JOINT_POSITIONS_SET_2,
    "set_3": TARGET_JOINT_POSITIONS_SET_3,
}

# Keep the public variable used by the existing planner and kinematics nodes.
TARGET_JOINT_POSITIONS = TARGET_DATASETS[ACTIVE_TARGET_SET].copy()
