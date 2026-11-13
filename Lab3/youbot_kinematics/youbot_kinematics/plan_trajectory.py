import rclpy
from rclpy.node import Node
from scipy.linalg import expm
from scipy.linalg import logm
from itertools import permutations
import time
import threading
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker

import numpy as np
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS


class YoubotTrajectoryPlanning(Node):
    def __init__(self):
        # Initialize node
        super().__init__("youbot_trajectory_planner")

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicStudent()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/youbot_arm_controller/joint_trajectory", 5
        )
        self.checkpoint_pub = self.create_publisher(Marker, "checkpoint_positions", 100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        self.get_logger().info("Waiting 5 seconds for everything to load up.")
        time.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = [
            "arm_joint_1",
            "arm_joint_2",
            "arm_joint_3",
            "arm_joint_4",
            "arm_joint_5",
        ]
        self.traj_pub.publish(traj)

    def q6(self):
        """This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        
        ##### Detailed Steps to solving Q1.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------

        # 1. Read the five arm joints by name, ignoring gripper joints and message
        # ordering. Process callbacks until feedback and the controller are ready.
        from sensor_msgs.msg import JointState
        from rclpy.duration import Duration
        from geometry_msgs.msg import Point

        joint_names = [f"arm_joint_{i}" for i in range(1, 6)]
        measured = {"joints": None}

        # (a) read current joint states （5 arm joints)
        def read_arm_state(msg):
            positions = dict(zip(msg.name, msg.position))
            if all(name in positions for name in joint_names):
                joints = np.array([positions[name] for name in joint_names])
                if np.all(np.isfinite(joints)):
                    measured["joints"] = joints

        # (b) wait for the joint-states callback 
        # Get the current joint positions and store it in the KDL youbot object.
        subscription = self.create_subscription(
            JointState, "/joint_states", read_arm_state, 10
        )
        self.get_logger().info("Waiting for arm feedback and trajectory controller.")
        deadline = time.monotonic() + 30.0
        try:
            while measured["joints"] is None or self.traj_pub.get_subscription_count() == 0:
                if not rclpy.ok() or time.monotonic() >= deadline:
                    raise RuntimeError("Start YouBot with use_jspg:=false before planning.")
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(subscription)
        self.kdl_youbot.current_joint_position = measured["joints"]

        # (c) Load all targets and compute the shortest path through them. 
        # (The first target is the measured start.)
        target_tfs, target_joints = self.load_targets()
        order, distance = self.get_shortest_path(target_tfs)
        self.get_logger().info(
            f"Shortest checkpoint order: {order.tolist()}; length: {distance:.6f} m"
        )

        # (d) Insert 30 interior samples per straight segment. 
        # Solve position IK
        # sequentially, using the previous solution as the next initial guess.
        num_intermediate_points = 30
        full_tfs = self.intermediate_tfs(
            order, target_tfs, num_points=num_intermediate_points
        )
        try:
            joint_path = self.full_checkpoints_to_joints(full_tfs, target_joints[:, 0])
        except RuntimeError as direct_error:
            # The base has two position-IK branches. They meet only on its
            # axis; joining them away from the axis cannot follow a straight
            # Cartesian segment. Replan through the axis instead of splicing
            # unrelated IK solutions or changing the measured starting state.
            self.get_logger().info(f"Direct route needs replanning: {direct_error}")
            start = target_tfs[:3, 3, 0]
            first = target_tfs[:3, 3, order[1]]
            radius_start = np.linalg.norm(start[:2])
            radius_first = np.linalg.norm(first[:2])
            axis_tf = target_tfs[:, :, 0].copy()
            axis_tf[:2, 3] = 0.0
            radius_sum = radius_start + radius_first
            if radius_sum > 1e-12:
                # Minimize the length of the two straight segments through
                # the axis by unfolding their radial/height coordinates.
                axis_tf[2, 3] = (
                    radius_first * start[2] + radius_start * first[2]
                ) / radius_sum

            # A repeated axis checkpoint provides a stationary Cartesian
            # segment for a continuous base rotation before leaving the axis.
            target_tfs = np.concatenate(
                (target_tfs[:, :, :1], axis_tf[:, :, None],
                 axis_tf[:, :, None], target_tfs[:, :, 1:]), axis=2,
            )
            order = np.r_[0, 1, 2, order[1:] + 2]
            self.get_logger().info(
                f"Replanning through base-axis waypoint {axis_tf[:3, 3].tolist()}; "
                "preserving the original target visit order."
            )
            for density in (30, 60, 120, 240):
                num_intermediate_points = density
                full_tfs = self.intermediate_tfs(order, target_tfs, num_points=density)
                try:
                    joint_path = self.full_checkpoints_to_joints(
                        full_tfs, target_joints[:, 0]
                    )
                    break
                except RuntimeError as replan_error:
                    if density == 240:
                        raise RuntimeError(
                            "Could not validate a continuous replanned trajectory; "
                            "no command sent."
                        ) from replan_error
                    self.get_logger().info(
                        f"Refining the replanned Cartesian segments: {replan_error}"
                    )
        self.get_logger().info(
            f"Interpolation: {num_intermediate_points} interior points per segment; "
            f"{full_tfs.shape[2]} total poses including checkpoints."
        )

        # Show the complete interpolated route on the README's existing Marker
        # topic. Keep the real measured start; no manual initialization is needed.
        path_marker = Marker()
        path_marker.header.frame_id = "base_link"
        path_marker.ns = "planned_trajectory"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 0.002
        path_marker.color.a = 1.0
        path_marker.color.g = 0.8
        path_marker.color.b = 1.0
        path_marker.points = [
            Point(x=float(full_tfs[0, 3, i]), y=float(full_tfs[1, 3, i]),
                  z=float(full_tfs[2, 3, i]))
            for i in range(full_tfs.shape[2])
        ]
        samples_marker = Marker()
        samples_marker.header.frame_id = "base_link"
        samples_marker.ns = "trajectory_samples"
        samples_marker.id = 0
        samples_marker.type = Marker.SPHERE_LIST
        samples_marker.action = Marker.ADD
        samples_marker.pose.orientation.w = 1.0
        samples_marker.scale.x = 0.012
        samples_marker.scale.y = 0.012
        samples_marker.scale.z = 0.012
        samples_marker.color.a = 1.0
        samples_marker.color.r = 0.1
        samples_marker.color.g = 1.0
        samples_marker.color.b = 0.1

        # Draw genuine interpolation samples, excluding the large checkpoints.
        # Keep visible balls about 1.8 cm apart to avoid a merged tube on short
        # segments. This filters visualization only: IK still uses every sample.
        visible_indices = []
        for segment in range(len(order) - 1):
            first = segment * (num_intermediate_points + 1)
            segment_indices = []
            for offset in range(1, num_intermediate_points + 1):
                index = first + offset
                point = full_tfs[:3, 3, index]
                start = full_tfs[:3, 3, first]
                end = full_tfs[:3, 3, first + num_intermediate_points + 1]
                if min(np.linalg.norm(point - start), np.linalg.norm(point - end)) < 0.018:
                    continue
                if not segment_indices or np.linalg.norm(
                    point - full_tfs[:3, 3, segment_indices[-1]]
                ) >= 0.018:
                    segment_indices.append(index)
            if not segment_indices and np.linalg.norm(end - start) > 1e-9:
                segment_indices.append(first + (num_intermediate_points + 1) // 2)
            visible_indices.extend(segment_indices)
        samples_marker.points = [path_marker.points[i] for i in visible_indices]
        goals_marker = Marker()
        goals_marker.header.frame_id = "base_link"
        goals_marker.ns = "trajectory_checkpoints"
        goals_marker.id = 0
        goals_marker.type = Marker.SPHERE_LIST
        goals_marker.action = Marker.ADD
        goals_marker.pose.orientation.w = 1.0
        goals_marker.scale.x = 0.025
        goals_marker.scale.y = 0.025
        goals_marker.scale.z = 0.025
        goals_marker.color.a = 1.0
        goals_marker.color.r = 1.0
        goals_marker.color.g = 0.5
        goals_marker.points = [
            Point(x=float(target_tfs[0, 3, i]), y=float(target_tfs[1, 3, i]),
                  z=float(target_tfs[2, 3, i])) for i in order
        ]

        # Refresh markers so a late RViz subscriber also sees the complete
        # validated route, including any axis waypoint introduced by replanning.
        def publish_path():
            stamp = self.get_clock().now().to_msg()
            for marker in (path_marker, samples_marker, goals_marker):
                marker.header.stamp = stamp
                self.checkpoint_pub.publish(marker)

        publish_path()
        self.checkpoint_timer = self.create_timer(1.0, publish_path)

        # Keep the same nominal time per checkpoint-to-checkpoint segment when
        # increasing sampling density. Limit average joint speed to 1.0 rad/s.
        min_sample_duration = 0.15 * 21 / (num_intermediate_points + 1)
        traj = JointTrajectory()
        traj.header.frame_id = "base_link"
        traj.joint_names = joint_names
        elapsed = 0.5
        for i in range(joint_path.shape[1]):
            if i > 0:
                largest_move = float(np.max(np.abs(joint_path[:, i] - joint_path[:, i - 1])))
                elapsed += max(min_sample_duration, largest_move / 1.0)
            point = JointTrajectoryPoint()
            point.positions = joint_path[:, i].tolist()
            point.time_from_start = Duration(seconds=elapsed).to_msg()
            traj.points.append(point)

        self.get_logger().info(
            f"Trajectory ready: {len(traj.points)} points; duration: {elapsed:.2f} s."
        )

        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the TARGET_JOINT_POSITIONS variable. In this variable you will find each
        row has target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        num_target_positions = len(TARGET_JOINT_POSITIONS)
        self.get_logger().info(f"{num_target_positions} target positions")
        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, num_target_positions + 1))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(
            np.identity(4), num_target_positions + 1, axis=1
        ).reshape((4, 4, num_target_positions + 1))

        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.current_joint_position
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(
            target_joint_positions[:, 0].tolist()
        )

        # TODO: populate the transforms in the target_cart_tf object
        # populate the joint positions in the target_joint_positions object
        # Your code starts here ------------------------------

        # Keep the measured start in column zero. Store each supplied joint
        # target and compute its end-effector transform with the student FK.
        for i, joints in enumerate(TARGET_JOINT_POSITIONS, start=1):
            target_joint_positions[:, i] = joints
            target_cart_tf[:, :, i] = self.kdl_youbot.forward_kinematics(joints.tolist())

        # Your code ends here ------------------------------

        self.get_logger().info(f"{target_cart_tf.shape} target poses")
        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, num_target_positions + 1)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, num_target_positions + 1)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """
        num_checkpoints = checkpoints_tf.shape[2]
        # TODO: implement this method. Make it flexible to accomodate different numbers of targets.
        # Your code starts here ------------------------------

        # Fix checkpoint zero as the start and enumerate every permutation of
        # the remaining targets. Minimize the sum of Euclidean segment lengths.
        if num_checkpoints < 1:
            raise ValueError("At least the starting checkpoint is required")
        positions = checkpoints_tf[:3, 3, :].T
        min_dist = float("inf")
        sorted_order = None
        for remaining in permutations(range(1, num_checkpoints)):
            order = np.array((0,) + remaining, dtype=int)
            distance = float(np.linalg.norm(np.diff(positions[order], axis=0), axis=1).sum())
            if distance < min_dist:
                min_dist = distance
                sorted_order = order

        # Your code ends here ------------------------------

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (num_checkpoints,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(
        self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points
    ):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting,
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """
        # TODO: implement this
        # Your code starts here ------------------------------

        # Join the segments in route order, including each shared checkpoint
        # only once. For M checkpoints and N interior samples: (M-1)*(N+1)+1.
        order = np.asarray(sorted_checkpoint_idx, dtype=int)
        if order.ndim != 1 or order.size == 0:
            raise ValueError("The checkpoint order must be a non-empty sequence")
        poses = [target_checkpoint_tfs[:, :, order[0]]]
        for start, end in zip(order[:-1], order[1:]):
            interior = self.decoupled_rot_and_trans(
                target_checkpoint_tfs[:, :, start],
                target_checkpoint_tfs[:, :, end], num_points,
            )
            poses.extend(interior[:, :, i] for i in range(interior.shape[2]))
            poses.append(target_checkpoint_tfs[:, :, end])
        full_checkpoint_tfs = np.stack(poses, axis=2)

        # Your code ends here ------------------------------

        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """
        self.get_logger().info("checkpoint a")
        self.get_logger().info(str(checkpoint_a_tf))
        self.get_logger().info("checkpoint b")
        self.get_logger().info(str(checkpoint_b_tf))
        # TODO: implement this
        # Your code starts here ------------------------------

        # Interpolate position on a straight line and rotation independently:
        # p(s)=(1-s)*p_a+s*p_b, R(s)=R_a*exp(s*log(R_a.T*R_b)).
        # A rotation vector gives a real rotation logarithm even at 180 degrees.
        from scipy.spatial.transform import Rotation

        if not isinstance(num_points, (int, np.integer)) or num_points < 0:
            raise ValueError("num_points must be a non-negative integer")
        rotation_a = checkpoint_a_tf[:3, :3]
        relative_rotation = rotation_a.T @ checkpoint_b_tf[:3, :3]
        rx, ry, rz = Rotation.from_matrix(relative_rotation).as_rotvec()
        rotation_log = np.array([[0., -rz, ry], [rz, 0., -rx], [-ry, rx, 0.]])
        tfs = np.repeat(np.eye(4)[:, :, None], num_points, axis=2)
        for i, fraction in enumerate(np.linspace(0., 1., num_points + 2)[1:-1]):
            tfs[:3, 3, i] = (
                (1. - fraction) * checkpoint_a_tf[:3, 3]
                + fraction * checkpoint_b_tf[:3, 3]
            )
            tfs[:3, :3, i] = rotation_a @ expm(fraction * rotation_log)

        # Your code ends here ------------------------------
        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints,
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:(Input)
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        # TODO: Implement this
        # Your code starts here ------------------------------

        # First solve forward from the actual measured configuration, warming
        # each Jacobian IK solve with the previous result. Preserve sample zero.
        initial_joints = np.asarray(init_joint_position, dtype=float).reshape(5)
        num_samples = full_checkpoint_tfs.shape[2]
        # Waypoint IK accuracy alone does not constrain motion between samples.
        # Allow at most 1 mm error along the controller's position-only, linear
        # joint interpolation. This is separate from the 1e-5 m IK tolerance.
        max_path_deviation = 1e-3

        def interpolation_is_valid(left, right, start, end):
            delta = right - left
            rates = np.abs(delta * self.kdl_youbot.youbot_joint_readings_polarity)
            cumulative_rates = np.cumsum(rates)
            previous_rates = np.r_[0.0, cumulative_rates[:-1]]
            # Each DH translation is a rotated vector. Its second derivative
            # is bounded by its length times the squared sum of joint rates.
            # The a term rotates with this joint; d rotates with earlier joints.
            dh = self.kdl_youbot.dh_params
            curvature_bound = float(
                np.abs(dh['a']) @ cumulative_rates**2
                + np.abs(dh['d']) @ previous_rates**2
            )

            def deviation(fraction):
                actual = self.kdl_youbot.forward_kinematics(
                    (left + fraction * delta).tolist()
                )[:3, 3]
                expected = (1.0 - fraction) * start + fraction * end
                error = float(np.linalg.norm(actual - expected))
                return error if np.isfinite(error) else float('inf')

            pending = [(0.0, 1.0, deviation(0.0), deviation(1.0), 0)]
            max_bound = 0.0
            while pending:
                a, b, error_a, error_b, depth = pending.pop()
                endpoint_error = max(error_a, error_b)
                if endpoint_error > max_path_deviation:
                    return False, endpoint_error
                # Between samples, the deviation from the error-vector chord
                # is at most M*h^2/8. Refine until the entire interval is bounded,
                # so a peak between test samples cannot silently pass validation.
                bound = endpoint_error + curvature_bound * (b - a)**2 / 8.0
                if bound <= max_path_deviation:
                    max_bound = max(max_bound, bound)
                    continue
                if depth >= 12:
                    return False, bound
                midpoint = (a + b) / 2.0
                error_mid = deviation(midpoint)
                pending.append((midpoint, b, error_mid, error_b, depth + 1))
                pending.append((a, midpoint, error_a, error_mid, depth + 1))
            return True, max_bound

        q_checkpoints = np.full((5, num_samples), np.nan)
        seed = initial_joints.copy()
        positions = full_checkpoint_tfs[:3, 3, :]
        radii = np.linalg.norm(positions[:2], axis=0)
        on_axis = radii < 1e-10
        axis_start = axis_end = -1
        axis_rotation = 0.0
        axis_base = seed[0]
        dh = self.kdl_youbot.dh_params
        base_polarity = self.kdl_youbot.youbot_joint_readings_polarity[0]
        # In the YouBot arm plane the two arm links and the tool offset form
        # a three-link chain. Use its reach to select a base branch that can
        # reach the remaining route, rather than a distant target's joint seed.
        arm_reach = abs(dh['a'][1]) + abs(dh['a'][2]) + np.hypot(dh['a'][3], dh['d'][4])

        for i in range(num_samples):
            if on_axis[i] and (i == 0 or not on_axis[i - 1]):
                axis_start = i
                axis_end = i + 1
                while axis_end < num_samples and on_axis[axis_end]:
                    axis_end += 1
                axis_base = seed[0]
                axis_rotation = 0.0
                if axis_end < num_samples:
                    outgoing = positions[:, axis_end]
                    candidates = []
                    for side in (1, -1):
                        radial = side * radii[axis_end:] - dh['a'][0]
                        height = positions[2, axis_end:] - dh['d'][0]
                        if np.any(np.hypot(radial, height) > arm_reach + 1e-8):
                            continue
                        heading = np.arctan2(outgoing[1], outgoing[0])
                        if side < 0:
                            heading += np.pi
                        base_target = (heading - dh['theta'][0]) / base_polarity
                        delta = base_target - axis_base
                        candidates.append(float(np.arctan2(np.sin(delta), np.cos(delta))))
                    if not candidates:
                        raise RuntimeError(
                            "No continuous base branch reaches the remaining route; "
                            "Cartesian replanning is required."
                        )
                    axis_rotation = min(candidates, key=abs)
                    if axis_end - axis_start < 2 and abs(axis_rotation) > 1e-4:
                        raise RuntimeError("A stationary axis segment is required to rotate the base.")

            if on_axis[i] and axis_end - axis_start > 1:
                # At zero tool radius, base rotation is a position-Jacobian
                # null-space motion. Spread it across the stationary segment.
                fraction = (i - axis_start) / (axis_end - axis_start - 1)
                seed[0] = axis_base + fraction * axis_rotation
            if i == 0:
                # Preserve the actual initial configuration exactly.
                joints = initial_joints.reshape(5, 1)
                actual = self.kdl_youbot.forward_kinematics(initial_joints.tolist())[:3, 3]
                error = float(np.linalg.norm(actual - positions[:, 0]))
            else:
                joints, error = self.ik_position_only(full_checkpoint_tfs[:, :, i], seed, num=2000)
            if not np.isfinite(error) or error >= 1e-5:
                raise RuntimeError(
                    f"Continuous IK failed at sample {i}; Cartesian replanning is required."
                )
            seed = joints.reshape(5).copy()
            # Tool roll does not affect position and need not move for position IK.
            seed[4] = initial_joints[4]
            q_checkpoints[:, i] = seed

        # Remove equivalent full turns without changing any waypoint position.
        # These are simulator joint targets; this position IK does not enforce
        # hardware joint limits, as documented by the template's IK method.
        for i in range(1, num_samples):
            delta = q_checkpoints[:, i] - q_checkpoints[:, i - 1]
            q_checkpoints[:, i] = q_checkpoints[:, i - 1] + np.arctan2(
                np.sin(delta), np.cos(delta)
            )

        # Validate every returned waypoint, including stationary axis samples,
        # before the template run() method publishes the single trajectory.
        max_error = 0.0
        for i in range(num_samples):
            actual = self.kdl_youbot.forward_kinematics(q_checkpoints[:, i].tolist())[:3, 3]
            error = float(np.linalg.norm(actual - full_checkpoint_tfs[:3, 3, i]))
            if not np.isfinite(error) or error >= 1e-5:
                raise RuntimeError(f"IK validation failed at sample {i}; no command sent.")
            max_error = max(max_error, error)

        # Validate every interval after angle unwrapping, not only a fallback
        # join. The controller receives these exact positions and no velocities.
        max_interpolation_bound = 0.0
        for i in range(num_samples - 1):
            valid, bound = interpolation_is_valid(
                q_checkpoints[:, i], q_checkpoints[:, i + 1],
                full_checkpoint_tfs[:3, 3, i], full_checkpoint_tfs[:3, 3, i + 1],
            )
            if not valid:
                raise RuntimeError(
                    f"Joint interpolation at samples {i}-{i + 1} could not be "
                    f"validated within {max_path_deviation:.6f} m of the Cartesian "
                    "path; no command sent."
                )
            max_interpolation_bound = max(max_interpolation_bound, bound)
        self.get_logger().info(
            f"IK solved {q_checkpoints.shape[1]} samples; max error: {max_error:.3e} m; "
            f"interpolation deviation bound: {max_interpolation_bound:.3e} m."
        )

        # Your code ends here ------------------------------

        return q_checkpoints

    def ik_position_only(self, pose, q0, lam=0.25, num=500):
        """Solve one position target by iterative Jacobian pseudoinverse updates.

        Args:
            pose (np.ndarray, 4x4): Target end-effector pose in the base frame.
                Only its translation is used; orientation is not constrained.
            q0 (np.ndarray): Five initial joint angles in radians, shape (5,) or (5, 1).
            lam (float): Positive step size for each joint update, default 0.25.
            num (int): Maximum number of updates, default 500.
        Returns:
            q (np.ndarray, 5x1): Final joint estimate in radians.
            error (float): Final Euclidean position error in metres. An error below
                1e-5 indicates convergence; otherwise the target has not been solved.

        This local solver may not converge from a distant or singular initial guess.
        It does not enforce joint limits or collision constraints.
        """
        pose = np.asarray(pose, dtype=float)
        initial_joints = np.asarray(q0, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("pose must be a finite 4x4 transformation matrix")
        if initial_joints.shape not in ((5,), (5, 1)) or not np.all(np.isfinite(initial_joints)):
            raise ValueError("q0 must contain five finite joint angles, shape (5,) or (5, 1)")
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("lam must be a positive finite step size")
        if not isinstance(num, (int, np.integer)) or num < 0:
            raise ValueError("num must be a non-negative integer")

        q = initial_joints.reshape(5).copy()
        target_position = pose[:3, 3]
        tolerance = 1e-5
        max_step = 0.1  # Limit the joint-update vector length to 0.1 radians.

        for _ in range(num):
            current_pose = self.kdl_youbot.forward_kinematics(q.tolist())
            position_error = target_position - current_pose[:3, 3]
            if np.linalg.norm(position_error) < tolerance:
                break

            position_jacobian = self.kdl_youbot.get_jacobian(q.tolist())[:3, :]
            # Delta p = J_p Delta q; use the pseudoinverse because J_p is 3x5.
            joint_step = lam * (np.linalg.pinv(position_jacobian) @ position_error)
            step_norm = np.linalg.norm(joint_step)
            if step_norm > max_step:
                joint_step *= max_step / step_norm
            q += joint_step

        # Recompute after the final update so error always describes the returned q.
        final_pose = self.kdl_youbot.forward_kinematics(q.tolist())
        error = float(np.linalg.norm(target_position - final_pose[:3, 3]))
        return q.reshape(5, 1), error


def main(args=None):
    rclpy.init(args=args)

    youbot_planner = YoubotTrajectoryPlanning()

    youbot_planner.run()

    rclpy.spin(youbot_planner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    youbot_planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
