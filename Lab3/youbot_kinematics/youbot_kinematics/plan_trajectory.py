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
        # TODO: implement this
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------

        # Read the five arm joints by name, ignoring gripper joints and message
        # ordering. Process callbacks until feedback and the controller are ready.
        from sensor_msgs.msg import JointState
        from rclpy.duration import Duration

        joint_names = [f"arm_joint_{i}" for i in range(1, 6)]
        measured = {"joints": None}

        def read_arm_state(msg):
            positions = dict(zip(msg.name, msg.position))
            if all(name in positions for name in joint_names):
                joints = np.array([positions[name] for name in joint_names])
                if np.all(np.isfinite(joints)):
                    measured["joints"] = joints

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

        # Convert joint targets to Cartesian checkpoints and find the shortest
        # open tour from the measured start; the route does not return home.
        target_tfs, target_joints = self.load_targets()
        order, distance = self.get_shortest_path(target_tfs)
        self.get_logger().info(
            f"Shortest checkpoint order: {order.tolist()}; length: {distance:.6f} m"
        )

        # Insert 20 interior samples per straight segment. Solve position IK
        # sequentially, using the previous solution as the next initial guess.
        full_tfs = self.intermediate_tfs(order, target_tfs, num_points=20)
        joint_path = self.full_checkpoints_to_joints(full_tfs, target_joints[:, 0])

        # Build timed joint targets, as in LAB2. Allow at least 0.15 seconds per
        # segment and limit its average joint speed to 1.0 rad/s (URDF: 1.57 rad/s).
        traj = JointTrajectory()
        traj.header.frame_id = "base_link"
        traj.joint_names = joint_names
        elapsed = 0.5
        for i in range(joint_path.shape[1]):
            if i > 0:
                largest_move = float(np.max(np.abs(joint_path[:, i] - joint_path[:, i - 1])))
                elapsed += max(0.15, largest_move / 1.0)
            point = JointTrajectoryPoint()
            point.positions = joint_path[:, i].tolist()
            point.time_from_start = Duration(seconds=elapsed).to_msg()
            traj.points.append(point)

        # Display only the original checkpoints with the template's large sphere
        # markers. Refresh them so RViz can also connect after planning finishes.
        ordered_tfs = target_tfs[:, :, order]
        self.publish_traj_tfs(ordered_tfs)
        self.checkpoint_timer = self.create_timer(
            1.0, lambda: self.publish_traj_tfs(ordered_tfs)
        )
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
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        # TODO: Implement this
        # Your code starts here ------------------------------

        # Use the existing Jacobian position IK, warming each solve with the
        # previous solution. Reject failed targets before any command is sent.
        seed = np.asarray(init_joint_position, dtype=float).reshape(5)
        q_checkpoints = np.empty((5, full_checkpoint_tfs.shape[2]))
        max_error = 0.0
        for i in range(full_checkpoint_tfs.shape[2]):
            joints, error = self.ik_position_only(full_checkpoint_tfs[:, :, i], seed, num=2000)
            if not np.isfinite(error) or error >= 1e-5:
                raise RuntimeError(
                    f"IK failed at sample {i}: position error {error:.6f} m. "
                    "No trajectory was sent. Try the bent initial joint pose "
                    "[4.0, 1.5, -2.8, 1.8, 2.9] before running the planner."
                )
            seed = joints.reshape(5)
            q_checkpoints[:, i] = seed
            max_error = max(max_error, error)
        self.get_logger().info(
            f"IK solved {q_checkpoints.shape[1]} samples; max error: {max_error:.3e} m."
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
