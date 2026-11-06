import rclpy
from rclpy.node import Node
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
import os
# BEGIN ADDITION: signal handling for shutdown.
# Reason: temporarily ignore repeated Ctrl+C signals while ROS resources close.
import signal
# END ADDITION: signal handling for shutdown.

from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
import matplotlib.pyplot as plt


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import PyKDL
from youbot_kinematics.urdf import treeFromUrdfModel
from youbot_kinematics.urdf_parser import URDF

from ament_index_python.packages import get_package_share_directory

import yaml

# BEGIN ADDITION: recognise an externally stopped ROS executor.
# Reason: ROS shutdown should reach the same cleanup and plotting path as Ctrl+C.
from rclpy.executors import ExternalShutdownException
# END ADDITION: recognise an externally stopped ROS executor.

class TrajDemo(Node):
    def __init__(self):
        super().__init__("traj_demo")

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 5
        )
        self.joint_state_sub  # prevent unused variable warning
        self.traj_publisher = self.create_publisher(
            JointTrajectory, "/franka_arm_controller/joint_trajectory", 5
        )
        self.traj_publisher

        # load from urdf file
        self.declare_parameter("urdf_package", "franka_description")
        self.urdf_package = (
            self.get_parameter("urdf_package").get_parameter_value().string_value
        )
        self.urdf_package_path = get_package_share_directory(self.urdf_package)
        self.declare_parameter("urdf_path_in_package", "urdfs/fr3.urdf")
        self.urdf_path_in_package = (
            self.get_parameter("urdf_path_in_package")
            .get_parameter_value()
            .string_value
        )
        self.urdf_name_path = os.path.join(
            self.urdf_package_path, self.urdf_path_in_package
        )

        self.get_logger().info(
            f"loading robot into KDL from urdf: {self.urdf_name_path}"
        )

        robot = URDF.from_xml_file(self.urdf_name_path)

        (ok, self.kine_tree) = treeFromUrdfModel(robot)

        if not ok:
            raise RuntimeError("couldn't load URDF into KDL tree succesfully")

        self.declare_parameter("base_link", "base")
        self.declare_parameter("ee_link", "fr3_link8")
        self.base_link = (
            self.get_parameter("base_link").get_parameter_value().string_value
        )
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value

        self.kine_chain = self.kine_tree.getChain(self.base_link, self.ee_link)
        self.NJoints = self.kine_chain.getNrOfJoints()
        self.current_joint_position = PyKDL.JntArray(self.NJoints)
        self.current_joint_velocity = PyKDL.JntArray(self.NJoints)
        # KDL solvers
        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kine_chain)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # loading trajectory structure

        self.declare_parameter("traj_cfg_pkg", "trajectory_demo")
        self.traj_cfg_pkg = (
            self.get_parameter("traj_cfg_pkg").get_parameter_value().string_value
        )
        self.traj_cfg_pkg_path = get_package_share_directory(self.traj_cfg_pkg)
        self.declare_parameter("traj_cfg_path_within_pkg", "cfg/traj_waypoints.yaml")
        self.traj_cfg_path_within_pkg = (
            self.get_parameter("traj_cfg_path_within_pkg")
            .get_parameter_value()
            .string_value
        )

        self.traj_cfg_path = os.path.join(
            self.traj_cfg_pkg_path, self.traj_cfg_path_within_pkg
        )

        with open(self.traj_cfg_path) as stream:
            self.traj_cfg = yaml.safe_load(stream)

        assert type(self.traj_cfg["joint_names"]) == list, type(
            self.traj_cfg["joint_names"]
        )
        assert len(self.traj_cfg["joint_names"]) == self.NJoints

        self.created_traj = False
        self.waiting_for_controller_logged = False

        self.get_logger().info(f"got traj cfg:\n{self.traj_cfg}")

        # BEGIN Q6(a) ADDITION: storage and time origin for measured samples.
        # Reason: retain feedback for plotting after the node stops; this fulfils
        # the starter's measurement-storage TODO. Recording starts at the first
        # valid sample, which can precede trajectory planning and publication.
        # Store measurements collected while the robot moves.
        self.samples = []

        # Set this when the first valid measurement is recorded.
        self.recording_start = None
        # END Q6(a) ADDITION: storage and time origin for measured samples.

        # BEGIN Q6(b) ADDITION: execution timing and completion-check state.
        # Reason: measure from publication separately from recording, and retain
        # the final target and tolerances needed to detect a settled finish.
        # Track trajectory execution separately from data recording.
        self.execution_start = None
        self.execution_time = None
        self.final_joint_target = None
        self.planned_duration = None

        self.joint_position_tolerance = 0.01  # Radians, for every joint.
        self.joint_velocity_tolerance = 0.01  # Radians/second, for every joint.
        self.settle_duration = 0.20  # Seconds continuously within both tolerances.
        self.goal_reached_since = None
        # END Q6(b) ADDITION: execution timing and completion-check state.

    def joint_state_callback(self, msg: JointState):
        """Callback for the joint states of the robot arm. It will get joint positions and velocities, and eventually the cartesian position of the end-effector and save this. It allows initializes the trajectory once the first joint state message comes through.

        Args:
            msg (JointState): ROS Joint State Message.
        """
        # JointState ordering can differ from the trajectory/KDL joint ordering.
        # Match names so IK seeds and completion errors use the correct joints.
        state_indices = {name: index for index, name in enumerate(msg.name)}
        joint_names = self.traj_cfg["joint_names"]
        if any(
            name not in state_indices
            or state_indices[name] >= len(msg.position)
            for name in joint_names
        ):
            return

        for i, name in enumerate(joint_names):
            state_index = state_indices[name]
            self.current_joint_position[i] = msg.position[state_index]
            self.current_joint_velocity[i] = (
                msg.velocity[state_index]
                if state_index < len(msg.velocity)
                else np.nan
            )

        joint_pos, joint_vel = self.kdl_to_np(
            self.current_joint_position
        ), self.kdl_to_np(self.current_joint_velocity)

        # BEGIN Q6(a) ADDITION: record timestamped joint and tool measurements.
        # Reason: fulfil the starter's feedback-recording TODOs with observed
        # values rather than requested targets. Copy the joint arrays so later
        # callbacks cannot overwrite measurements already stored for plotting.

        # BEGIN Q6(c) ADDITION: read position and orientation from the same TF.
        # Reason: both parts of the pose must refer to one transform. TF supplies
        # its latest available value; it is not timestamp-matched to JointState.
        ee_pose = self.get_ee_pose_ros()
        if ee_pose is None:
            return
        ee_pos, ee_orientation = ee_pose
        # END Q6(c) ADDITION: read position and orientation from the same TF.

        # Measure time relative to our first valid sample.
        now = self.get_clock().now()

        if self.recording_start is None:
            self.recording_start = now

        elapsed = (now - self.recording_start).nanoseconds / 1e9

        # Store one complete measurement record.
        self.samples.append({
            "time": elapsed,
            "joint_positions": joint_pos.copy(),
            "joint_velocities": joint_vel.copy(),
            "ee_position": ee_pos,
            # BEGIN Q6(c) ADDITION: record measured roll, pitch and yaw in radians.
            # Reason: compare observed orientation with the YAML targets later.
            "ee_orientation": ee_orientation,
            # END Q6(c) ADDITION: record measured roll, pitch and yaw in radians.
        })
        # END Q6(a) ADDITION: record timestamped joint and tool measurements.

        # we do this after we have our first callback to have current joint positions
        
        # BEGIN Q6(b) MODIFICATION: time the starter's publish-once operation.
        # Reason: start timing after IK planning and just before publication.
        # The message stamp uses that same instant; the starter's one-time
        # publish behaviour is retained inside this extended block.
        if not self.created_traj:
            # A volatile, publish-once command can be lost during DDS discovery.
            # Receiving JointState/TF does not imply this publisher is connected.
            if self.traj_publisher.get_subscription_count() == 0:
                if not self.waiting_for_controller_logged:
                    self.get_logger().info(
                        "Waiting for a subscriber on "
                        "/franka_arm_controller/joint_trajectory before planning."
                    )
                    self.waiting_for_controller_logged = True
                return

            joint_traj = self.create_traj()

            # Remember the final joint angles for the completion check.
            final_point = joint_traj.points[-1]
            self.final_joint_target = np.asarray(
                final_point.positions,
                dtype=float,
            )

            # Convert the final arrival time into seconds.
            self.planned_duration = (
                final_point.time_from_start.sec
                + final_point.time_from_start.nanosec * 1e-9
            )

            # Start timing after planning, just before publishing.
            self.execution_start = self.get_clock().now()
            joint_traj.header.stamp = self.execution_start.to_msg()

            self.traj_publisher.publish(joint_traj)
            self.created_traj = True

            self.get_logger().info(
                f"Trajectory sent. Planned duration: "
                f"{self.planned_duration:.2f} s"
            )

            return
        # END Q6(b) MODIFICATION: time the starter's publish-once operation.

        # BEGIN Q6(b) ADDITION: detect completion and stop the execution timer.
        # Reason: a scheduled end time alone does not confirm the arm has settled.
        # Record completion once, after all joints pass both tolerances for the
        # full hold interval. This includes approach and settling, excludes IK
        # planning, and does not stop the controller or further sample recording.
        if self.execution_start is not None and self.execution_time is None:
            execution_elapsed = (now - self.execution_start).nanoseconds / 1e9

            # Taking the maximum checks that EVERY joint meets the tolerance.
            max_joint_error = float(
                np.max(np.abs(joint_pos - self.final_joint_target))
            )
            max_joint_speed = float(np.max(np.abs(joint_vel)))

            # The path may pass near its final pose before the scheduled end.
            at_goal = (
                execution_elapsed >= self.planned_duration
                and max_joint_error <= self.joint_position_tolerance
                and max_joint_speed <= self.joint_velocity_tolerance
            )

            if at_goal:
                if self.goal_reached_since is None:
                    self.goal_reached_since = now

                # Require a short hold so one passing sample is not enough.
                settled_for = (now - self.goal_reached_since).nanoseconds / 1e9
                if settled_for >= self.settle_duration:
                    self.execution_time = execution_elapsed
                    self.get_logger().info(
                        f"Trajectory complete. Measured execution time: "
                        f"{self.execution_time:.3f} s "
                        f"(includes approach and "
                        f"{self.settle_duration:.2f} s settling confirmation)."
                    )
            else:
                # Leaving either tolerance restarts the settling check.
                self.goal_reached_since = None
        # END Q6(b) ADDITION: detect completion and stop the execution timer.

    def kdl_to_np(self, data: PyKDL.JntArray) -> NDArray:
        """Helper Function to go from KDL arrays to numpy arrays

        Args:
            data (PyKDL.JntArray): desired KDL array to convert

        Returns:
            NDArray: converted NP Array
        """
        is_1d = data.columns() == 1
        np_shape = (data.rows(), data.columns()) if not is_1d else (data.rows(),)
        mat = np.zeros(np_shape)
        for i in range(data.rows()):
            if not is_1d:
                for j in range(data.columns()):
                    mat[i, j] = data[i, j]
            else:
                mat[i] = data[i]
        return mat

    # BEGIN Q6(c) ADDITION: obtain the full end-effector pose in the base frame.
    # Reason: extend Q6(a)'s TF position lookup to include the measured rotation.
    # Convert TF's quaternion into the same RPY convention used for IK targets.
    # Return None when TF is unavailable instead of recording fictitious zeros.
    def get_ee_pose_ros(
        self,
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Return position in metres and RPY orientation in radians from one TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
            )
        except TransformException:
            return None

        translation = transform.transform.translation
        quaternion = transform.transform.rotation
        rotation = PyKDL.Rotation.Quaternion(
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        )
        position = (
            translation.x,
            translation.y,
            translation.z,
        )
        # GetRPY uses the same convention as Rotation.RPY in the IK helper.
        return position, rotation.GetRPY()
    # END Q6(c) ADDITION: obtain the full end-effector pose in the base frame.

    # BEGIN Q6(a,c) MODIFICATION: replace the starter's dummy position helper.
    # Reason: the original returned [0, 0, 0]. Preserve its position-only interface
    # while obtaining real TF data through the shared pose helper added above.
    def get_ee_pos_ros(self) -> Optional[Tuple[float, float, float]]:
        """Keep the original position-only helper available."""
        pose = self.get_ee_pose_ros()
        return None if pose is None else pose[0]
    # END Q6(a,c) MODIFICATION: replace the starter's dummy position helper.

    def get_joint_pos(
        self,
        current_angles: Tuple[float],
        target_position: Tuple[float],
        target_orientation: Optional[Tuple[float]] = None,
    ) -> Tuple[float]:
        """Helper function to solve inverse kinematics using the KDL library

        Args:
            current_angles (Tuple[float]): current joint angles of the robot arm
            target_position (Tuple[float]): target cartesian position, 3 dimensional, of the end effector in world coordinates
            target_orientation (Optional[Tuple[float]], optional): target orientation in RPY format, optional. Defaults to None.

        Raises:
            RuntimeError: raises error if IK fails to solve. Check the workspace of the robot if this happens.

        Returns:
            Tuple[float]: joint angles that were found to satisfy targets
        """

        assert len(target_position) == 3
        assert target_orientation is None or len(target_orientation) == 3
        assert len(current_angles) == self.NJoints

        pos = PyKDL.Vector(target_position[0], target_position[1], target_position[2])
        if target_orientation is not None:
            # Constructs a rotation by first applying a rotation of r around the x-axis, then a rotation of p around the original y-axis, and finally a rotation of y around the original z-axis
            rot = PyKDL.Rotation.RPY(
                target_orientation[0], target_orientation[1], target_orientation[2]
            )

        seed_array = PyKDL.JntArray(self.NJoints)
        for i in range(self.NJoints):
            seed_array[i] = current_angles[i]

        if target_orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self.NJoints)

        if self.ik_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = list(result_angles)
            return result
        else:
            raise RuntimeError(
                f"Did not solve for goal_pose: {goal_pose} with initial seed {seed_array}"
            )


    # BEGIN Q6(a,b) MODIFICATION: complete the starter's trajectory builder.
    # Reason: the starter returned an empty JointTrajectory. Populate its joint
    # names, frame, and timed joint targets so the controller can execute it.
    # Sum each segment duration once: the starter added an already cumulative
    # arrival time back into its total, which gave incorrect multi-point timing.
    # Retain the previous IK solution as the next seed, and stamp after planning.
    def create_traj(self) -> JointTrajectory:
        """Convert Cartesian waypoints into a timed joint trajectory."""
        waypoints = self.traj_cfg["waypoints"]["cartesion"]

        joint_traj = JointTrajectory()
        joint_traj.joint_names = list(self.traj_cfg["joint_names"])
        joint_traj.header.frame_id = self.base_link

        # Start IK from the robot's current joint configuration.
        current_angles = self.kdl_to_np(self.current_joint_position)

        elapsed = 0.0

        for waypoint in waypoints:
            segment_duration = float(waypoint["time"])

            if not np.isfinite(segment_duration) or segment_duration <= 0:
                raise ValueError("Waypoint durations must be finite and positive.")

            # BEGIN Q6(c) MODIFICATION: validate and use optional RPY targets.
            # Reason: get_joint_pos() already accepts orientation, but the starter
            # never passed it. Reject malformed angles before solving IK.
            # Missing orientation preserves the behaviour of the Q6(a/b) YAMLs.
            # That behaviour requests identity rotation, not a free orientation.
            target_orientation = waypoint.get("orientation")
            if target_orientation is not None:
                target_orientation = np.asarray(target_orientation, dtype=float)
                if (
                    target_orientation.shape != (3,)
                    or not np.all(np.isfinite(target_orientation))
                ):
                    raise ValueError(
                        "Waypoint orientation must contain three finite "
                        "[roll, pitch, yaw] values in radians."
                    )

            # Find joint angles for both position and, when supplied, orientation.
            target_angles = self.get_joint_pos(
                current_angles,
                waypoint["position"],
                target_orientation,
            )
            # END Q6(c) MODIFICATION: validate and use optional RPY targets.

            elapsed += segment_duration

            # Describe one timed target for all seven joints.
            point = JointTrajectoryPoint()
            point.positions = [float(angle) for angle in target_angles]
            point.time_from_start = Duration(seconds=elapsed).to_msg()

            joint_traj.points.append(point)

            # Use this solution as the starting guess for the next target.
            current_angles = target_angles

        # Timestamp after completing the IK calculations.
        joint_traj.header.stamp = self.get_clock().now().to_msg()

        return joint_traj
    # END Q6(a,b) MODIFICATION: complete the starter's trajectory builder.


    # BEGIN Q6(a) MODIFICATION: replace the starter's empty plot with measurements.
    # Reason: show the executed XYZ path, its recorded start and requested targets.
    # Equal axis scales preserve shape; saving before plt.show() keeps a report
    # image even after the windows close. Later Q6(b/c) extensions are marked below.
    def plot_joint_traj(self):
        """Save the full 3D trajectory and a face-on yz view of the drawing."""
        if not self.samples:
            print("No valid position samples were recorded.")
            return

        positions = np.asarray(
            [sample["ee_position"] for sample in self.samples],
            dtype=float,
        )

        targets = np.asarray(
            [
                waypoint["position"]
                for waypoint in self.traj_cfg["waypoints"]["cartesion"]
            ],
            dtype=float,
        )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label="Recorded trajectory",
        )

        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            color="green",
            s=60,
            label="Recorded start",
        )

        ax.scatter(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            color="red",
            marker="x",
            s=80,
            label="Requested waypoint",
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        # Use the configuration name to distinguish default, letter and orientation
        # runs. Repeating the same configuration replaces its previous plot files.
        trajectory_name = os.path.splitext(
            os.path.basename(self.traj_cfg_path)
        )[0]
        ax.set_title(
            f"{trajectory_name} trajectory — position in the base frame"
        )
        ax.legend()

        # Use the same distance scale on all three axes.
        all_points = np.vstack((positions, targets))
        lower = all_points.min(axis=0)
        upper = all_points.max(axis=0)
        centre = (lower + upper) / 2
        half_width = max(float((upper - lower).max()) / 2, 0.01) * 1.1

        ax.set_xlim(centre[0] - half_width, centre[0] + half_width)
        ax.set_ylim(centre[1] - half_width, centre[1] + half_width)
        ax.set_zlim(centre[2] - half_width, centre[2] + half_width)
        ax.set_box_aspect((1, 1, 1))

        fig.tight_layout()

        # Existing local output destination, retained here without changing it.
        output_dir = (
            "/home/swx/modelling_output"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"{trajectory_name}_trajectory.png",
        )

        fig.savefig(output_path, dpi=300)
        print(f"Saved trajectory plot to: {output_path}")

        # BEGIN Q6(b) ADDITION: save a face-on yz plot of the recorded drawing.
        # Reason: the 3D view can hide the letter behind the approach movement.
        # Use measured y/z coordinates at equal scale, excluding the scheduled
        # approach and samples after confirmed completion. This projection does
        # not measure departures from the drawing plane in the x direction.
        if (
            len(targets) > 1
            and self.execution_start is not None
            and self.recording_start is not None
        ):
            sample_times = np.asarray(
                [sample["time"] for sample in self.samples],
                dtype=float,
            )

            # Sample times begin at recording_start, before planning finishes.
            execution_offset = (
                self.execution_start - self.recording_start
            ).nanoseconds / 1e9

            # The first waypoint's duration is the scheduled approach interval.
            approach_duration = float(
                self.traj_cfg["waypoints"]["cartesion"][0]["time"]
            )
            drawing_start_time = execution_offset + approach_duration
            drawing_mask = sample_times >= drawing_start_time

            # Exclude idle samples collected after completion was confirmed.
            if self.execution_time is not None:
                drawing_end_time = execution_offset + self.execution_time
                drawing_mask &= sample_times <= drawing_end_time

            drawing_positions = positions[drawing_mask]
            if len(drawing_positions) > 0:
                fig_yz, ax_yz = plt.subplots(figsize=(6, 7))

                # Project the measured path: column 1 is y, column 2 is z.
                ax_yz.plot(
                    drawing_positions[:, 1],
                    drawing_positions[:, 2],
                    linewidth=2,
                    label="Recorded drawing",
                )
                ax_yz.scatter(
                    targets[:, 1],
                    targets[:, 2],
                    color="red",
                    marker="x",
                    s=45,
                    label="Requested waypoints",
                )
                ax_yz.scatter(
                    drawing_positions[0, 1],
                    drawing_positions[0, 2],
                    color="green",
                    s=40,
                    zorder=3,
                    label="Recorded drawing start",
                )

                incomplete = " (incomplete)" if self.execution_time is None else ""
                ax_yz.set_title(
                    f"{trajectory_name} drawing — yz projection{incomplete}\n"
                    "Base frame; scheduled approach excluded"
                )
                ax_yz.set_xlabel("y [m]")
                ax_yz.set_ylabel("z [m]")
                ax_yz.set_aspect("equal", adjustable="box")
                ax_yz.margins(x=0.15, y=0.08)
                ax_yz.grid(True, alpha=0.3)
                ax_yz.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13))
                fig_yz.tight_layout()

                yz_output_path = os.path.join(
                    output_dir,
                    f"{trajectory_name}_drawing_yz.png",
                )
                fig_yz.savefig(yz_output_path, dpi=300)
                print(f"Saved yz drawing plot to: {yz_output_path}")
            else:
                print("YZ plot skipped: no samples after the scheduled approach.")
        # END Q6(b) ADDITION: save a face-on yz plot of the recorded drawing.

        # BEGIN Q6(c) ADDITION: save position and orientation against time.
        # Reason: generate both required time plots before opening the figures.
        self.plot_pose_time_series(trajectory_name, output_dir)
        # END Q6(c) ADDITION: save position and orientation against time.

        plt.show()
    # END Q6(a) MODIFICATION: replace the starter's empty plot with measurements.

    # BEGIN Q6(c) ADDITION: plot measured pose and the requested waypoint targets.
    # Reason: compare recorded XYZ/RPY with targets at their scheduled times.
    # Use publication as time zero, show the approach, and exclude later idle
    # samples. Old configs without orientation retain their existing plot set.
    def plot_pose_time_series(self, trajectory_name, output_dir):
        """Save separate XYZ and RPY time plots for configs with orientation targets."""
        waypoints = self.traj_cfg["waypoints"]["cartesion"]
        if not any(w.get("orientation") is not None for w in waypoints):
            return
        if self.execution_start is None or self.recording_start is None:
            print("Pose time plots skipped: trajectory execution has not started.")
            return

        # Use trajectory publication as t=0, matching the execution timer.
        execution_offset = (
            self.execution_start - self.recording_start
        ).nanoseconds / 1e9
        times = np.asarray([s["time"] for s in self.samples]) - execution_offset
        mask = times >= 0.0
        if self.execution_time is not None:
            mask &= times <= self.execution_time
        if not np.any(mask):
            print("Pose time plots skipped: no samples after trajectory publication.")
            return

        times = times[mask]
        positions = np.asarray([s["ee_position"] for s in self.samples])[mask]
        orientations = np.asarray([s["ee_orientation"] for s in self.samples])[mask]
        target_times = np.cumsum([float(w["time"]) for w in waypoints])
        target_positions = np.asarray([w["position"] for w in waypoints])
        # NaNs omit markers for any waypoint without an explicit RPY target.
        target_orientations = np.asarray([
            w["orientation"] if w.get("orientation") is not None else [np.nan] * 3
            for w in waypoints
        ], dtype=float)
        approach_duration = target_times[0]
        end_time = max(float(times[-1]), 0.1)
        incomplete = " (incomplete)" if self.execution_time is None else ""

        plots = (
            ("position", positions, target_positions, ("x", "y", "z"), "m"),
            ("orientation", orientations, target_orientations,
             ("roll", "pitch", "yaw"), "rad"),
        )
        for quantity, measured, requested, labels, unit in plots:
            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            for index, ax in enumerate(axes):
                ax.plot(times, measured[:, index], label="Recorded")
                ax.scatter(
                    target_times, requested[:, index], color="red", marker="x",
                    zorder=3, label="Requested waypoints",
                )
                ax.axvspan(
                    0.0, min(approach_duration, end_time), color="grey",
                    alpha=0.12, label="Scheduled approach",
                )
                # A constant target applies to the drawing, after the approach.
                if (
                    quantity == "orientation"
                    and np.all(np.isfinite(requested[:, index]))
                    and np.allclose(requested[:, index], requested[0, index])
                    and end_time >= approach_duration
                ):
                    ax.hlines(
                        requested[0, index], approach_duration, end_time,
                        colors="black", linestyles="--", linewidth=1,
                        label="Constant orientation target",
                    )
                ax.set_ylabel(f"{labels[index]} [{unit}]")
                ax.set_xlim(0.0, end_time)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time since trajectory publication [s]")
            representation = "Cartesian position" if quantity == "position" else "RPY orientation"
            fig.suptitle(
                f"{trajectory_name} — {representation}{incomplete}\n"
                f"{self.ee_link} relative to {self.base_link}"
            )
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, legend_labels, loc="upper center",
                bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=9,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            output_path = os.path.join(
                output_dir, f"{trajectory_name}_{quantity}_time.png"
            )
            fig.savefig(output_path, dpi=300)
            print(f"Saved {quantity} time plot to: {output_path}")
    # END Q6(c) ADDITION: plot measured pose and the requested waypoint targets.

# BEGIN MODIFICATION: clean shutdown followed by plotting.
# Reason: handle Ctrl+C and external ROS shutdown, including interruption during
# node construction. Release ROS resources even when shutdown is interrupted,
# ignore duplicate Ctrl+C during cleanup, then restore normal signal handling
# before opening plots. Retained starter startup/spin calls provide context.
def main(args=None):
    rclpy.init(args=args)
    traj_demo = None

    try:
        traj_demo = TrajDemo()
        rclpy.spin(traj_demo)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        # ros2 run can forward Ctrl+C after the terminal has already sent it.
        # Ignore duplicate interrupts while releasing ROS resources.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                if traj_demo is not None:
                    traj_demo.destroy_node()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)

    if traj_demo is not None:
        traj_demo.plot_joint_traj()
# END MODIFICATION: clean shutdown followed by plotting.


if __name__ == "__main__":

    main()
