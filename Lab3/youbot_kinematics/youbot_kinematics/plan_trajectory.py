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

    def _joint_position_limits(self):
        """Read limits in the same joint coordinates as JointState and the URDF."""
        if not hasattr(self, '_position_limits'):
            from pathlib import Path
            import xml.etree.ElementTree as ET
            from ament_index_python.packages import (
                PackageNotFoundError, get_package_share_directory,
            )

            try:
                description = Path(get_package_share_directory('youbot_description'))
            except PackageNotFoundError:
                description = Path(__file__).resolve().parents[2] / 'youbot_description'
            robot = ET.parse(description / 'urdfs' / 'youbot.urdf').getroot()
            joints = {joint.attrib['name']: joint for joint in robot.findall('joint')}
            limits = [joints[f'arm_joint_{i}'].find('limit') for i in range(1, 6)]
            lower = np.array([float(limit.attrib['lower']) for limit in limits])
            upper = np.array([float(limit.attrib['upper']) for limit in limits])
            self._position_limits = lower, upper
        return self._position_limits

    def _bounded_checkpoint_route(self, target_tfs, target_joints):
        """Find the shortest route in the direct/axis transition graph.

        Each target may have two base-angle branches. Direct edges preserve a
        branch; an axis transfer permits a continuous, bounded change of branch.
        Dynamic programming chooses both the target order and these branches.
        """
        lower, upper = self._joint_position_limits()
        initial = target_joints[:, 0].copy()
        if np.any(initial < lower) or np.any(initial > upper):
            raise RuntimeError('The measured starting joints exceed URDF limits.')
        dh = self.kdl_youbot.dh_params
        polarity = self.kdl_youbot.youbot_joint_readings_polarity[0]
        reach = abs(dh['a'][1]) + abs(dh['a'][2]) + np.hypot(dh['a'][3], dh['d'][4])

        def side_of(position, base):
            theta = dh['theta'][0] + polarity * base
            radial = position[:2] @ np.array([np.cos(theta), np.sin(theta)])
            return 1 if radial >= 0.0 else -1

        states = []
        count = target_tfs.shape[2] - 1
        for target in range(count):
            pose = target_tfs[:, :, target + 1]
            supplied = target_joints[:, target + 1].copy()
            supplied[4] = initial[4]
            if np.any(supplied < lower) or np.any(supplied > upper):
                raise RuntimeError(f'Target {target + 1} is outside URDF joint limits.')
            radius = np.linalg.norm(pose[:2, 3])
            heading = np.arctan2(pose[1, 3], pose[0, 3])
            for side in (1, -1):
                if np.hypot(side * radius - dh['a'][0], pose[2, 3] - dh['d'][0]) > reach + 1e-9:
                    continue
                raw = (heading + (np.pi if side < 0 else 0.0) - dh['theta'][0]) / polarity
                first = int(np.ceil((lower[0] - raw) / (2.0 * np.pi)))
                last = int(np.floor((upper[0] - raw) / (2.0 * np.pi)))
                for turn in range(first, last + 1):
                    base = raw + turn * 2.0 * np.pi
                    guesses = [supplied.copy()]
                    guesses += [np.array([base, a, b, c, initial[4]]) for a, b, c in (
                        (1.2, -2.0, 1.5), (0.9, -1.5, 1.0), (1.5, -2.5, 1.8))]
                    witness = None
                    for guess in guesses:
                        guess[0] = base
                        solution, error = self.ik_position_only(pose, guess, num=600, base_angle=base)
                        if error < 1e-5:
                            witness = solution.reshape(5)
                            break
                    if witness is not None:
                        states.append({'target': target, 'pose': pose, 'base': float(base),
                                       'side': side, 'joints': witness})
            if not any(state['target'] == target for state in states):
                raise RuntimeError(f'No bounded position-IK branch for target {target + 1}.')

        start = {'target': -1, 'pose': target_tfs[:, :, 0], 'base': float(initial[0]),
                 'side': side_of(target_tfs[:3, 3, 0], initial[0]), 'joints': initial}

        def edge(left, right):
            a, b = left['pose'][:3, 3], right['pose'][:3, 3]
            fractions = np.linspace(0.0, 1.0, 17)
            positions = a[:, None] * (1.0 - fractions) + b[:, None] * fractions
            radii = np.linalg.norm(positions[:2], axis=0)
            direct = left['side'] == right['side'] and abs(right['base'] - left['base']) < np.pi - 1e-8
            direct = direct and np.min(radii) > 1e-7
            radial = left['side'] * radii - dh['a'][0]
            direct = direct and np.all(np.hypot(radial, positions[2] - dh['d'][0]) <= reach + 1e-9)
            if direct:
                return float(np.linalg.norm(b - a)), False
            radius_a, radius_b = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
            return float(np.hypot(radius_a + radius_b, b[2] - a[2])), True

        size = len(states)
        costs = np.full((1 << count, size), np.inf)
        parents = {}
        edges = {(i, j): edge(states[i], states[j]) for i in range(size) for j in range(size)
                 if states[i]['target'] != states[j]['target']}
        for i, state in enumerate(states):
            mask = 1 << state['target']
            costs[mask, i] = edge(start, state)[0]
            parents[mask, i] = None
        for mask in range(1, 1 << count):
            for i, state in enumerate(states):
                if not np.isfinite(costs[mask, i]):
                    continue
                for j, following in enumerate(states):
                    bit = 1 << following['target']
                    if mask & bit:
                        continue
                    next_mask = mask | bit
                    score = costs[mask, i] + edges[i, j][0]
                    if score < costs[next_mask, j]:
                        costs[next_mask, j] = score
                        parents[next_mask, j] = (mask, i)
        mask = (1 << count) - 1
        index = int(np.argmin(costs[mask]))
        route = []
        while True:
            route.append(states[index])
            previous = parents[mask, index]
            if previous is None:
                break
            mask, index = previous
        route.reverse()

        poses = [start['pose']]
        bases = [start['base']]
        sides = []
        left = start
        for right in route:
            if edge(left, right)[1]:
                a, b = left['pose'][:3, 3], right['pose'][:3, 3]
                ra, rb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
                axis = left['pose'].copy()
                axis[:2, 3] = 0.0
                axis[2, 3] = (rb * a[2] + ra * b[2]) / (ra + rb) if ra + rb > 1e-12 else a[2]
                poses.extend([axis, axis.copy()])
                bases.extend([left['base'], right['base']])
                sides.extend([left['side'], 0])
            poses.append(right['pose'])
            bases.append(right['base'])
            sides.append(right['side'])
            left = right
        self.get_logger().info(
            f'Bounded checkpoint order: {[0] + [state["target"] + 1 for state in route]}; '
            f'{len(poses) - count - 1} axis transition checkpoints.'
        )
        return np.stack(poses, axis=2), np.asarray(bases), sides

    def _base_angles_for_route(self, full_tfs, checkpoint_bases, segment_sides, density):
        """Choose the bounded base angle for every Cartesian interpolation sample."""
        dh = self.kdl_youbot.dh_params
        polarity = self.kdl_youbot.youbot_joint_readings_polarity[0]
        lower, upper = self._joint_position_limits()
        values = np.empty(full_tfs.shape[2])
        stride = density + 1
        for segment, side in enumerate(segment_sides):
            for offset in range(stride + 1):
                index = segment * stride + offset
                fraction = offset / stride
                reference = (1.0 - fraction) * checkpoint_bases[segment] + fraction * checkpoint_bases[segment + 1]
                position = full_tfs[:3, 3, index]
                if side == 0 or np.linalg.norm(position[:2]) < 1e-10:
                    base = reference
                else:
                    heading = np.arctan2(position[1], position[0]) + (np.pi if side < 0 else 0.0)
                    base = (heading - dh['theta'][0]) / polarity
                    base += 2.0 * np.pi * np.round((reference - base) / (2.0 * np.pi))
                if base < lower[0] - 1e-9 or base > upper[0] + 1e-9:
                    raise RuntimeError('A Cartesian segment crosses the base-joint limit.')
                values[index] = np.clip(base, lower[0], upper[0])
        return values

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
            self.get_logger().info(f"Direct route needs bounded replanning: {direct_error}")
            target_tfs, checkpoint_bases, segment_sides = self._bounded_checkpoint_route(
                target_tfs, target_joints
            )
            order = np.arange(target_tfs.shape[2], dtype=int)
            for density in (30, 60, 120, 240):
                num_intermediate_points = density
                full_tfs = self.intermediate_tfs(order, target_tfs, num_points=density)
                base_angles = self._base_angles_for_route(
                    full_tfs, checkpoint_bases, segment_sides, density
                )
                try:
                    joint_path = self.full_checkpoints_to_joints(
                        full_tfs, target_joints[:, 0], base_angles=base_angles
                    )
                    break
                except RuntimeError as replan_error:
                    if density == 240:
                        raise RuntimeError(
                            "Could not validate a continuous bounded trajectory; no command sent."
                        ) from replan_error
                    self.get_logger().info(f"Refining bounded Cartesian segments: {replan_error}")
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

        # Preserve the nominal duration budget when inserting safety transitions.
        # Every interval still respects the existing 1 rad/s joint-speed bound.
        joint_moves = np.max(np.abs(np.diff(joint_path, axis=1)), axis=0)
        durations = np.maximum(joint_moves / 1.0, 1e-3)
        nominal_duration = 0.15 * 21 * (len(TARGET_JOINT_POSITIONS) + 2)
        if durations.size:
            durations *= max(1.0, nominal_duration / float(durations.sum()))
        traj = JointTrajectory()
        traj.header.frame_id = "base_link"
        traj.joint_names = joint_names
        elapsed = 0.5
        for i in range(joint_path.shape[1]):
            if i > 0:
                elapsed += float(durations[i - 1])
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

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position, base_angles=None):
        """Solve and validate a continuous path with strict URDF position limits."""
        lower, upper = self._joint_position_limits()
        initial = np.asarray(init_joint_position, dtype=float).reshape(5)
        if not np.all(np.isfinite(initial)) or np.any(initial < lower) or np.any(initial > upper):
            raise RuntimeError('The measured starting configuration exceeds URDF limits.')
        count = full_checkpoint_tfs.shape[2]
        if base_angles is not None and np.shape(base_angles) != (count,):
            raise ValueError('One base angle is required for each Cartesian sample.')
        path = np.empty((5, count))
        seed = initial.copy()
        for i in range(count):
            pose = full_checkpoint_tfs[:, :, i]
            if i == 0:
                solution = initial.copy()
                error = float(np.linalg.norm(self.kdl_youbot.forward_kinematics(initial.tolist())[:3, 3] - pose[:3, 3]))
            else:
                base = None if base_angles is None else float(base_angles[i])
                solution, error = self.ik_position_only(pose, seed, num=1500, base_angle=base)
                solution = solution.reshape(5)
            if not np.isfinite(error) or error >= 1e-5:
                raise RuntimeError(f'Bounded continuous IK failed at sample {i}; no command sent.')
            solution[4] = initial[4]
            path[:, i] = solution
            seed = solution.copy()
        if np.any(path < lower[:, None]) or np.any(path > upper[:, None]):
            raise RuntimeError('A trajectory point exceeds URDF position limits; no command sent.')

        positions = full_checkpoint_tfs[:3, 3, :]
        errors = [float(np.linalg.norm(self.kdl_youbot.forward_kinematics(path[:, i].tolist())[:3, 3] - positions[:, i]))
                  for i in range(count)]
        if not np.all(np.isfinite(errors)) or max(errors) >= 1e-5:
            raise RuntimeError('Waypoint FK validation failed; no command sent.')
        maximum_bound = 0.0
        dh = self.kdl_youbot.dh_params
        for i in range(count - 1):
            left, right = path[:, i], path[:, i + 1]
            delta = right - left
            cumulative = np.cumsum(np.abs(delta * self.kdl_youbot.youbot_joint_readings_polarity))
            previous = np.r_[0.0, cumulative[:-1]]
            curvature = float(np.abs(dh['a']) @ cumulative**2 + np.abs(dh['d']) @ previous**2)
            def deviation(fraction):
                actual = self.kdl_youbot.forward_kinematics((left + fraction * delta).tolist())[:3, 3]
                expected = (1.0 - fraction) * positions[:, i] + fraction * positions[:, i + 1]
                return float(np.linalg.norm(actual - expected))
            pending = [(0.0, 1.0, errors[i], errors[i + 1], 0)]
            while pending:
                a, b, error_a, error_b, depth = pending.pop()
                endpoint = max(error_a, error_b)
                bound = endpoint + curvature * (b - a)**2 / 8.0
                if not np.isfinite(bound) or endpoint > 1e-3:
                    raise RuntimeError(f'Cartesian interpolation error exceeds 1 mm at interval {i}.')
                if bound <= 1e-3:
                    maximum_bound = max(maximum_bound, bound)
                    continue
                if depth >= 12:
                    raise RuntimeError(f'Cartesian interpolation could not be validated at interval {i}.')
                middle = (a + b) / 2.0
                error_middle = deviation(middle)
                pending.extend([(a, middle, error_a, error_middle, depth + 1),
                                (middle, b, error_middle, error_b, depth + 1)])
        self.get_logger().info(
            f'IK solved {count} samples; max error: {max(errors):.3e} m; '
            f'interpolation deviation bound: {maximum_bound:.3e} m; all joints within URDF limits.'
        )
        return path

    def ik_position_only(self, pose, q0, lam=0.25, num=500, base_angle=None):
        """Position IK using bounded, active-set Jacobian pseudoinverse steps.

        The optional base angle selects a continuous admissible branch. Only
        position is constrained; collision checking is outside this lab solver.
        """
        pose = np.asarray(pose, dtype=float)
        initial = np.asarray(q0, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError('pose must be a finite 4x4 transformation matrix')
        if initial.shape not in ((5,), (5, 1)) or not np.all(np.isfinite(initial)):
            raise ValueError('q0 must contain five finite joint angles')
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError('lam must be positive and finite')
        if not isinstance(num, (int, np.integer)) or num < 0:
            raise ValueError('num must be a non-negative integer')
        lower, upper = self._joint_position_limits()
        q = initial.reshape(5).copy()
        if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
            raise ValueError('Initial joints exceed URDF position limits')
        q = np.clip(q, lower, upper)
        if base_angle is not None:
            if not np.isfinite(base_angle) or base_angle < lower[0] or base_angle > upper[0]:
                raise ValueError('Selected base angle exceeds URDF position limits')
            q[0] = base_angle
        target = pose[:3, 3]
        for _ in range(num):
            error = target - self.kdl_youbot.forward_kinematics(q.tolist())[:3, 3]
            norm = float(np.linalg.norm(error))
            if norm < 1e-5:
                break
            jacobian = self.kdl_youbot.get_jacobian(q.tolist())[:3, :]
            free = np.ones(5, dtype=bool)
            free[4] = False
            if base_angle is not None:
                free[0] = False
            step = np.zeros(5)
            for _active in range(5):
                step[:] = 0.0
                if not np.any(free):
                    break
                step[free] = lam * (np.linalg.pinv(jacobian[:, free], rcond=1e-7) @ error)
                outward = free & (((q <= lower + 1e-10) & (step < 0.0)) |
                                  ((q >= upper - 1e-10) & (step > 0.0)))
                if not np.any(outward):
                    break
                free[outward] = False
            length = float(np.linalg.norm(step))
            if length < 1e-12:
                break
            if length > 0.1:
                step *= 0.1 / length
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                candidate = np.clip(q + fraction * step, lower, upper)
                candidate_error = float(np.linalg.norm(target - self.kdl_youbot.forward_kinematics(candidate.tolist())[:3, 3]))
                if candidate_error < norm:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                break
        error = float(np.linalg.norm(target - self.kdl_youbot.forward_kinematics(q.tolist())[:3, 3]))
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
