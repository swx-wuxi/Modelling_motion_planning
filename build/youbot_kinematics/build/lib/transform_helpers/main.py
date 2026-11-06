import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import numpy as np
from numpy.typing import NDArray

from transform_helpers.utils import rotmat2q

# Modified DH parameters for the FR3 arm; the last row is the fixed flange.
# a and d are in meters.
a_list = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
d_list = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]

# radians
alpha_list = [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0]
theta_list = [0] * len(alpha_list)

DH_PARAMS = np.array([a_list, d_list, alpha_list, theta_list]).T

BASE_FRAME = "base"
FRAMES = ["fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7", "fr3_link8"]
# Modified DH Params for the Franka FR3 robot arm
# Reference: 
# https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters
def get_transform_n_to_n_minus_one(n: int, theta: float) -> NDArray:
    # n starts at 1; array indices start at 0.
    n_minus_one = n - 1

    # Read the modified DH parameters a, d, and alpha.
    # Use the input joint angle for theta.
    a = DH_PARAMS[n_minus_one, 0]
    d = DH_PARAMS[n_minus_one, 1]
    alpha = DH_PARAMS[n_minus_one, 2]

    # Angle unit: rad
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Modified DH transform from frame n to its parent frame n-1.
    transform_matrix = np.array([
        [
            cos_theta,
            -sin_theta,
            0.0,
            a
        ],
        [
            sin_theta * cos_alpha,
            cos_theta * cos_alpha,
            -sin_alpha,
            -d * sin_alpha
        ],
        [
            sin_theta * sin_alpha,
            cos_theta * sin_alpha,
            cos_alpha,
            d * cos_alpha
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ], dtype=float)
    return transform_matrix

class ForwardKinematicCalculator(Node):

    def __init__(self):
        super().__init__('fk_calculator')
         
        # Publish our calculated transforms on /tf.
        self.tf_broadcaster = TransformBroadcaster(self)

        # Keep our frame names separate from the official robot frames.
        self.prefix = "my_robot/"

        # Recalculate the transforms whenever new joint angles arrive.
        self.joint_sub = self.create_subscription(
            JointState,               # Message type to receive
            '/joint_states',          # Joint state topic
            self.publish_transforms,  # Call the function when a message arrives
            10                        # Message queue depth
        )

    def publish_transforms(self, msg: JointState):
        self.get_logger().debug(str(msg))

        transforms = []
        # Note our frames list is longer than the number of joints,
        # So some special handling is required

        # Total 10 links
        # For base_link and the flange_link(fr_link8),theta = 0
        for i in range(len(FRAMES) - 1, -1, -1):
            frame_id = self.prefix + FRAMES[i]
            if i != 0:
                parent_id = self.prefix + FRAMES[i - 1]
            else:
                parent_id = self.prefix + BASE_FRAME
            theta = None
            if i != len(FRAMES) - 1 and i != 0:
                # Assumes msg.position follows fr3_joint1 through fr3_joint7.
                theta = msg.position[i - 1]
            elif i == len(FRAMES) - 1:
                # The flange has a fixed offset and no joint rotation.
                theta = 0
            else:
                theta = 0

            t = TransformStamped()     
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_id
            t.child_frame_id = frame_id

            if i != 0:
                transform = get_transform_n_to_n_minus_one(i, theta)
            else:
                transform = np.eye(4)

            # ROS stores orientation as a quaternion (x, y, z, w).
            quat = rotmat2q(transform[:3, :3])

            # Extract the position in meters from the last column of the 4x4 transform matrix.
            t.transform.translation.x = float(transform[0, 3])
            t.transform.translation.y = float(transform[1, 3])
            t.transform.translation.z = float(transform[2, 3])

            # Set the rotation quaternion with (x, y, z,w).
            t.transform.rotation = quat

            transforms.append(t)

        # Send the whole chain in one message so echo --once shows all frames.
        self.tf_broadcaster.sendTransform(transforms)

def main(args=None):
    # Start the node and process incoming joint states until it is stopped.
    rclpy.init(args=args)
    fk_calculator = ForwardKinematicCalculator()
    rclpy.spin(fk_calculator)

    # Destroy the node explicitly
    fk_calculator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



