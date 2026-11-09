## KUKA YouBot — Kinematics and Trajectory Planning (Lab3 Q1)
The code repository: 
(https://github.com/swx-wuxi/Modelling_motion_planning/tree/main/Lab3)

Pull the latest code and supporting packages from the repository above:

```bash
git pull 
```

### 1. Overview

This project implements forward kinematics, Jacobian calculation, singularity detection, inverse kinematics and trajectory planning for the KUKA YouBot using ROS 2.

The main implementations are:

Q1(a): Forward kinematics, geometric Jacobian and singularity detection.

Q1(b): Shortest Cartesian path planning, waypoint interpolation and Jacobian-based inverse kinematics.

Q1(c): ROS 2 node for publishing the end-effector transform based on joint-state feedback.

Q1(d): Trajectory planning with modified target datasets.

### 2. Requirements
- Ubuntu 22.04

- ROS 2 Humble

- Python 3

- NumPy and SciPy

- The required YouBot simulation and ROS packages from the coursework repository

#### ROS Node for EE Tracking
To display the end-effector frame computed by the custom FK node for Q1(c),
open another terminal at the workspace root and run:

```bash
source install/setup.bash
ros2 run youbot_kinematics follow_trajectory
```

In RViz, add or enable the TF display and select
`arm_end_effector_follower` under Frames. This frame updates from
`/joint_states` as the robot moves.

### 3. Running and Testing

Follow the build and launch instructions in `Lab3/README.md`. 

To test different target points, edit `ACTIVE_TARGET_SET` in
`Lab3/youbot_kinematics/youbot_kinematics/target_data.py`:

- `"original"`: four original targets
- `"set_1"`: four alternative targets
- `"set_2"`: five targets
- `"set_3"`: six targets

After changing the selection or editing target values, stop the running
trajectory planner. From the workspace root, rebuild the package,
reload the environment, and restart the planner:
```
    colcon build --packages-select youbot_kinematics
    source install/setup.bash
    ros2 run youbot_kinematics plan_trajectory
```

Keep the robot launch running with `use_jspg:=false`.
If using a new terminal, `source install/setup.bash` there before running
the planner.
