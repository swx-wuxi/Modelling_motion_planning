import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from math import isfinite

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent


class YoubotTrajectoryFollower(Node):
    def __init__(self):
        super().__init__('youbot_trajectory_follower')
        self.joint_names = [f'arm_joint_{i}' for i in range(1, 6)]

        # Reuse the student FK implementation from Q1(a), just as the planner
        # does. Only this follower node is spun; the helper supplies FK methods.
        self.kinematics = YoubotKinematicStudent()

        # INPUT: the broadcaster publishes measured JointState messages here.
        # Each incoming message schedules joint_state_callback(msg).
        self.subscription_ = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # OUTPUT: TransformBroadcaster creates the /tf publisher internally.
        # It wraps each TransformStamped in the TFMessage expected by TF/RViz.
        self.tf_broadcaster = TransformBroadcaster(self)
        self.received_first_state = False
        self.get_logger().info(
            'Listening on /joint_states; broadcasting base_link -> arm_end_effector_follower.'
        )

    def joint_state_callback(self, msg):
        # 1. Extract the five arm joints by name. Message ordering may change,
        # and gripper joint states must not be included in the arm FK input.
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in self.joint_names):
            return
        joints = [float(positions[name]) for name in self.joint_names]
        if not all(isfinite(angle) for angle in joints):
            return

        # 2. Convert actual joint angles to the current tool pose using FK.
        # The 4x4 matrix contains rotation in [:3, :3] and position in [:3, 3].
        pose = self.kinematics.forward_kinematics(joints)

        # 3. Fill a transform from the base to a separate computed tool frame.
        # Use the measurement timestamp so TF describes the sampled joint state.
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'arm_end_effector_follower'
        transform.transform.translation.x = float(pose[0, 3])
        transform.transform.translation.y = float(pose[1, 3])
        transform.transform.translation.z = float(pose[2, 3])

        # ROS stores orientation as quaternion [x, y, z, w], not a 3x3 matrix.
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])

        # 4. Publish the computed pose. A new joint message produces a new TF;
        # no timer, IK calculation, or robot motion command is needed here.
        self.tf_broadcaster.sendTransform(transform)
        if not self.received_first_state:
            self.received_first_state = True
            self.get_logger().info(
                f'First FK position: x={pose[0, 3]:.4f}, '
                f'y={pose[1, 3]:.4f}, z={pose[2, 3]:.4f} m.'
            )


def main(args=None):
    rclpy.init(args=args)
    follower = YoubotTrajectoryFollower()
    try:
        rclpy.spin(follower)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        follower.kinematics.destroy_node()
        follower.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
