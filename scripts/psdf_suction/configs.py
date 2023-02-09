import os
import numpy as np
from scipy.spatial.transform import Rotation as R

DEVICE = "cuda:0"
EPSILON = 1e-6

class Config:
    def __init__(self):
        self.package_name = "psdf_suction"
        self.path = os.path.dirname(os.path.dirname(__file__))

        # vaccum cup
        self.gripper_radius = 0.01
        self.gripper_height = 0.02
        self.gripper_vertices = 8
        self.gripper_angle_threshold = 45
        self.vacuum_length = 0.125

        # PSDF
        # self.volume_origin = np.array([0.15, -0.20, 0.03])
        # self.volume_range = np.array([0.5, 0.5, 0.5])
        # self.volume_resolution = 0.002
        self.volume_origin = np.array([-0.5, -0.50, 0])
        self.volume_range = np.array([2, 2, 2])
        self.volume_resolution = 0.01
        self.volume_shape = np.ceil(self.volume_range / self.volume_resolution).astype(np.int32).tolist()
        self.T_volume_to_world = np.eye(4).astype(np.float32)
        self.T_volume_to_world[:3, 3] = self.volume_origin
        self.T_world_to_volume = np.eye(4).astype(np.float32)
        self.T_world_to_volume[:3, 3] = -self.T_volume_to_world[:3, 3]

        # setting init pose
        self.init_position = (2 * self.volume_origin + self.volume_range) / 2
        self.init_position[2] = self.volume_origin[2] + self.volume_range[2]
        self.init_orientation = R.from_euler("xyz", [180, 0., -90], degrees=True).as_quat()

config = Config()