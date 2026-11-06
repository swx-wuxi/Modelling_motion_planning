from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation
    

    if T.shape != (3, 3):
        raise ValueError

    # TODO: implement this
    raise NotImplementedError

    return q

def quat2rodrigues(q: Quaternion) -> NDArray:
    # Function that transforms a ROS quaternion representation to the Rodrigues
    
    # The returned variable should contain a 3 element array in the direction
    # of the rotation axis with magnitude equal to the rotation angle

    # TODO: implement this
    raise NotImplementedError

    return T

def quat2zyx(q: Quaternion) -> NDArray:
    # Function that transforms a ROS quaternion representation to three Euler
    # angles corresponding to the intrinsic Tait-Bryan Z-Y-X representation
    # 
    
    # The returned variable should contain a 3 element array of the Euler angles

    # TODO: implement this
    raise NotImplementedError

    return T