import numpy as np
from scipy.spatial.transform import Rotation as R

def get_orientation(normal):
    z_axis = np.array([0.0, 0.0, 1.0])
    rvec = np.cross(z_axis, normal)
    if np.linalg.norm(rvec) == 0:
        rvec = z_axis
    else:
        rvec = rvec / np.linalg.norm(rvec)
    theta = np.arccos(np.dot(z_axis, normal))
    return R.from_rotvec(rvec*theta).as_quat()