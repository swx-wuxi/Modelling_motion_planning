from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

# ******For mathematical principles in this file, refer to:***********
# https://sjpaf79z13xy.jp.larksuite.com/wiki/NLrxw1R2vikHnekArZ3jf1kepRP
# ***********************************************

def rotmat2q(T: NDArray) -> Quaternion:
    # Convert a 3x3 rotation matrix to a ROS quaternion.
    if T.shape != (3, 3):
        raise ValueError

    trace = np.trace(T)   # Sum the diagonal elements.

    if trace > 0:
        w = 0.5 * np.sqrt(trace + 1.0)
        x = (T[2, 1] - T[1, 2]) / (4*w)
        y = (T[0, 2] - T[2, 0]) / (4*w)
        z = (T[1, 0] - T[0, 1]) / (4*w)

    elif T[0, 0] > T[1, 1] and T[0, 0] > T[2, 2]:
        x = 0.5 * np.sqrt(1.0 + T[0, 0] - T[1, 1] - T[2, 2])
        w = (T[2, 1] - T[1, 2]) / (4*x)
        y = (T[0, 1] + T[1, 0]) / (4*x)
        z = (T[0, 2] + T[2, 0]) / (4*x)

    elif T[1, 1] > T[2, 2]:
        y = 0.5 * np.sqrt(1.0 + T[1, 1] - T[0, 0] - T[2, 2])
        w = (T[0, 2] - T[2, 0]) / (4*y)
        x = (T[0, 1] + T[1, 0]) / (4*y)
        z = (T[1, 2] + T[2, 1]) / (4*y)

    else:
        z = 0.5 * np.sqrt(1.0 + T[2, 2] - T[0, 0] - T[1, 1])
        w = (T[1, 0] - T[0, 1]) / (4*z)
        x = (T[0, 2] + T[2, 0]) / (4*z)
        y = (T[1, 2] + T[2, 1]) / (4*z)

    # Normalize the quaternion to unit length.
    components = np.array([x, y, z, w], dtype=float)
    components /= np.linalg.norm(components)

    q = Quaternion()
    q.x = float(components[0])
    q.y = float(components[1])
    q.z = float(components[2])
    q.w = float(components[3])

    return q

def quat2rodrigues(q: Quaternion) -> NDArray:
    # Convert a ROS quaternion to a Rodrigues rotation vector.
    
    # Return a three-element vector along the rotation axis
    # with magnitude equal to the rotation angle in radians.

    # Normalize the ROS quaternion in (x, y, z, w) order.
    # Compute the quaternion norm.
    components = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(components)    # Quaternion norm.
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Quaternion must be finite and non-zero")
    components /= norm

    # q and -q represent the same rotation; choose an angle no greater than pi.
    if components[3] < 0.0:
        components = -components

    vector = components[:3]    # Extract the (x, y, z) vector part.
    vector_norm = np.linalg.norm(vector)
    # For small theta, sin(theta / 2) is approximately theta / 2.
    if vector_norm < 1e-8:
        return 2.0 * vector

    angle = 2.0 * np.arctan2(vector_norm, components[3])
    axis = vector / vector_norm
    return angle * axis

def quat2zyx(q: Quaternion) -> NDArray:
    # Convert a ROS quaternion to three Euler angles
    # using the intrinsic Tait-Bryan Z-Y-X convention.
    
    # Return a three-element array of Euler angles in Z-Y-X order.
    
    # ROS 2 quaternions use (x, y, z, w) order.
    components = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(components)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Quaternion must be finite and non-zero")
    x, y, z, w = components / norm

    # Extract the Z-Y-X angles in radians.
    alpha = np.arctan2(2.0 * (x * y + w * z),
                       1.0 - 2.0 * (y * y + z * z))
    theta = np.arcsin(np.clip(2.0 * (w * y - x * z), -1.0, 1.0))
    gamma = np.arctan2(2.0 * (y * z + w * x),
                       1.0 - 2.0 * (x * x + y * y))

    cos_theta = np.hypot(1.0 - 2.0 * (y * y + z * z),
                         2.0 * (x * y + w * z))
    
    # *******At theta = +/-pi/2, cos_theta = 0 *********
    # And gimbal lock prevents determining alpha and gamma separately.
    # So we need to write a new algorithm when cos_theta is very small. 
    if cos_theta < 1e-8:
        alpha = np.arctan2(2.0 * (w * z - x * y),
                           1.0 - 2.0 * (x * x + z * z))
        gamma = 0.0

    return np.array([alpha, theta, gamma], dtype=float)