import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import json
import time

from psdf_suction.ur5_commander import UR5Commander
from psdf_suction.psdf import PSDF
from psdf_suction.analyser.vacuum_cup_analyser import VacuumCupAnalyser
from psdf_suction.realsense_commander import RealSenseCommander
from psdf_suction.configs import config, EPSILON, DEVICE
from psdf_suction.utils import get_orientation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from psdf_suction.vaccum_cup import VaccumCup
from avt_grasp_tool.chart_utils import refine_grasp_normal, show_chart, analyse_chart
import os
from cv2 import imread
from pytools import P

from torch import frac
from xela.XELA import XELA
from avt_grasp_tool.chart_utils import *
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs


def compute_score(graspable_map, point_map, normal_map, position_pre,
                  dist_weight_pub=None, normal_weight_pub=None, range_weight_pub=None):
    # current first
    dist_sigma = 0.05
    dist_weight = np.exp(-0.5 * (
        ((point_map[..., :2] - position_pre[:2]) ** 2).sum(axis=-1) / dist_sigma ** 2))
    dist_weight = (dist_weight / (dist_weight.sum() + EPSILON)
                   ).astype(np.float32)

    # upward first
    normal_weight = normal_map[..., 2] + 1
    normal_weight = (normal_weight / (normal_weight.sum() +
                     EPSILON)).astype(np.float32)

    # face center first
    ksize = 21
    range_weight = graspable_map * \
        cv2.GaussianBlur(graspable_map, (ksize, ksize), 5)
    range_weight = (range_weight / (range_weight.sum() +
                    EPSILON)).astype(np.float32)

    # show topic
    if dist_weight_pub is not None:
        dist_weight_pub.publish(sensor_msgs.msg.Image(
            data=dist_weight.tobytes(),
            height=dist_weight.shape[0],
            width=dist_weight.shape[1],
            encoding="32FC1"
        ))
    if normal_weight_pub is not None:
        normal_weight_pub.publish(sensor_msgs.msg.Image(
            data=normal_weight.tobytes(),
            height=normal_weight.shape[0],
            width=normal_weight.shape[1],
            encoding="32FC1"
        ))
    if range_weight_pub is not None:
        range_weight_pub.publish(sensor_msgs.msg.Image(
            data=range_weight.tobytes(),
            height=range_weight.shape[0],
            width=range_weight.shape[1],
            encoding="32FC1"
        ))

    score = dist_weight * normal_weight * range_weight
    return score


def get_camera_pose(arm, T_cam_to_tool0):
    tool0_pose = arm.get_pose()
    T_tool0_to_world = np.eye(4)
    T_tool0_to_world[:3, :3] = R.from_quat(tool0_pose[3:]).as_matrix()
    T_tool0_to_world[:3, 3] = tool0_pose[:3]
    T_cam_to_world = T_tool0_to_world @ T_cam_to_tool0
    return T_cam_to_world


def main():
    import rospy
    rospy.init_node('avg_suction')

    # init analyser
    analyser = VacuumCupAnalyser(radius=config.gripper_radius,
                                 height=config.gripper_height,
                                 num_vertices=config.gripper_vertices,
                                 angle_threshold=config.gripper_angle_threshold)
    rospy.loginfo("Analyser initialized")

    # init arm control
    arm = UR5Commander()
    rospy.loginfo("Arm initialized")

    # suction cup init
    print("[ init suction cup ]")
    cup = VaccumCup()
    cup.release()

    # load camera config
    with open(os.path.join(os.path.dirname(__file__), "../config/cam_info_realsense.json"), 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)

    # init camera
    camera = RealSenseCommander()
    camera.start()
    rospy.loginfo("Camera initialized")

    # init xela
    w_list = [2.76749049, 2.50199062, 1.88931666, 1.66445914, 3.20192794, 2.37374989, 1.75607728, 1.5546019,
              3.01912242, 2.76639647, 1.68184047, 1.17119893, 2.76687492, 2.62543491, 1.70173676, 1.2056192]

    b_list = [1.89618919e-04, -2.11084252e-05, -4.35183739e-05, 9.51700947e-05, -2.64512283e-05, -1.49470146e-04, -7.97973529e-05, 1.45149973e-05, -
              3.58794390e-05, -1.47181813e-04, -8.89498016e-05, 1.56093658e-05, 7.57060221e-05, -5.67033388e-05, -3.06370707e-06, 4.16183831e-05]
    xela = XELA()
    xela.start()
    print('XELA initialized')

    # get xela data
    frame_fixed = xela.get()
    x_fixed = frame_fixed[:, :, 0]
    y_fixed = frame_fixed[:, :, 1]

    # touch T pose
    T_touch_to_tool0 = np.eye(4)
    T_touch_to_tool0[:3, 3] = [-(0.0805+0.0075), 0-(0.001+0.0075), 0.1711]
    T_touch_to_tool0[:3, :3] = R.from_euler(
        'xyz', [180, 0, 90], degrees=True).as_matrix()

    # VaccumCup Parameter
    cup_radius = 0.01
    cup_height = 0.02
    cup_samples = 8
    spring_threshold = 0.1

    # threshold
    cam_dist_threshold = 0.3

    # sequential grasping
    init_pose = config.init_position.tolist() + config.init_orientation.tolist()
    init_grasp_position = config.init_position.copy()
    init_grasp_position[2] = config.volume_origin[2]
    init_grasp_normal = np.array([0, 0, 1])
    
    # arm move loop
    while True:
        # init arm pose
        arm.set_pose(init_pose, wait=True)
        rospy.loginfo("Arm back to init pose")

        # pre-touch
        psdf = PSDF(config.volume_shape, config.volume_resolution,
                    device=DEVICE, with_color=True)
        rospy.loginfo("PSDF initialized")

        grasp_position = init_grasp_position
        grasp_normal = init_grasp_normal
        distance = 0.4
        
        while True:
            # get camera message
            color, depth = camera.get_image()

            T_cam_to_world = get_camera_pose(arm, T_cam_to_tool0)
            T_cam_to_volume = config.T_world_to_volume @ T_cam_to_world

            # fuse new data to psdf
            ts = rospy.rostime.get_time()
            psdf.fuse(np.copy(depth), cam_intr,
                      T_cam_to_volume, color=np.copy(color))
            rospy.loginfo("fuse time %f" % (rospy.rostime.get_time()-ts))

            # flatten to 2D point image
            ts = rospy.rostime.get_time()
            point_map, normal_map, variance_map, _ = psdf.flatten()
            point_map = point_map @ config.T_volume_to_world[:3,
                                                             :3].T + config.T_volume_to_world[:3, 3]

            # analysis
            normal_mask = normal_map[..., 2] > np.cos(
                config.gripper_angle_threshold/180*np.pi)
            variance_mask = variance_map < 1e-2
            # z_mask = point_map[:, :, 2] > 0.02
            final_mask = normal_mask * variance_mask  # * z_mask
            obj_ids = np.where(final_mask != 0)
            vision_dict = {"point_cloud": point_map,
                           "normal": -normal_map}
            graspable_map = analyser.analyse(
                vision_dict, obj_ids).astype(np.float32)

            # update grasp pose
            score = compute_score(graspable_map, point_map,
                                  normal_map, grasp_position)
            idx = np.argmax(score)
            i, j = idx // config.volume_shape[0], idx % config.volume_shape[1]
            grasp_position = point_map[i, j]
            grasp_normal = normal_map[i, j]

            # move
            grasp_orientation = get_orientation(grasp_normal)
            grasp_orientation = (R.from_quat(config.init_orientation) *
                                 R.from_quat(grasp_orientation)).as_quat()
            T_cam2world = np.eye(4)
            T_cam2world[:3, :3] = R.from_quat(grasp_orientation).as_matrix()
            T_cam2world[:3, 3] = grasp_position
            T_tool02world = T_cam2world @ np.linalg.inv(T_cam_to_tool0)
            move_position = T_tool02world[:3, 3] + grasp_normal * distance
            move_orientation = R.from_matrix(T_tool02world[:3, :3]).as_quat()
            move_pose = move_position.tolist() + move_orientation.tolist()
            arm.set_pose(move_pose, wait=False)
            distance -= 0.01
            if distance < cam_dist_threshold:
                break

        # touch grasp position
        rospy.loginfo("touch grasp position")
        T_touch_to_world = T_cam2world
        # R_cam2world @ R_touch2cam
        T_touch_to_world[:3, :3] = T_touch_to_world[:3,
                                                    :3] @ T_touch_to_tool0[:3, :3]
        T_tool0_to_world = T_touch_to_world @ np.linalg.inv(T_touch_to_tool0)

        # Approaching till touched
        touch_direction = T_tool02world[:3, 2]
        distance = 0.01
        while True:  # touch loop
            cup.ur5_touch()
            move_position = T_tool0_to_world[:3,
                                             3] - distance * touch_direction
            move_orientation = R.from_matrix(
                T_tool0_to_world[:3, :3]).as_quat()
            move_pose = move_position.tolist() + move_orientation.tolist()
            arm.set_pose(move_pose, wait=True)
            # cup.ur5_touch()

            frame = xela.get()
            pred_z = w_list * frame[:, :, 2].reshape(-1) + b_list
            pred_z = pred_z * 1000

            # 1.3 Touch Prediction
            pred_diff = pred_z.max() - pred_z.min()

            # print('The diff between z_min and z_max is:', pred_diff)
            if pred_diff < 0.5:
                print('Uskin is not touching.')
                distance -= 0.001
            elif pred_diff >= 0.5 and pred_diff < 1.8:
                print('Uskin is touching sth.')
                distance -= (2.0 - pred_diff) / 1000.0
            elif pred_diff >= 1.8:
                break

        # 1.4 Delete the Outlier Point
        frame = xela.get()
        pred_z = w_list * frame[:, :, 2].reshape(-1) + b_list
        pred_z = pred_z * 1000
        pred_z[pred_z > 2] = 2
        pred_z[pred_z < 0] = 0

        # Construct the initial chart
        vertices = np.stack(
            [x_fixed, y_fixed, pred_z.reshape(4, 4)/1000], axis=-1)
        print("vertices.shape", vertices.shape)
        vertices = vertices.reshape(-1, 3)
        faces = np.array([
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [4, 8, 9],
            [4, 9, 5],
            [5, 9, 10],
            [5, 10, 6],
            [6, 10, 11],
            [6, 11, 7],
            [8, 12, 13],
            [8, 13, 9],
            [9, 13, 14],
            [9, 14, 10],
            [10, 14, 11],
            [11, 14, 15]
        ])

        # analyse
        grasp_position = np.mean(vertices[[5, 6, 9, 10]], axis=0)
        grasp_normal = np.array([0, 0, 1])
        cup_radius = 0.01298 / 2
        cup_height = 0.00697
        cup_samples = 8
        spring_threshold = 0.1

        grasp_normal = refine_grasp_normal(
            vertices, faces, grasp_position, grasp_normal, cup_radius, cup_samples)
        rospy.loginfo("refined")
        is_graspable = analyse_chart(vertices, faces, grasp_position,
                                     grasp_normal, cup_radius, cup_height, cup_samples, spring_threshold)
        rospy.loginfo("analysed: result: {}".format(is_graspable))

        # conduct grasping
        if is_graspable:
            # touch frame to world frame
            # point_in_world = T_tool0_to_world @ T_touch_to_tool0 @ point_in_touch
            # normal_in_world = R_tool0_to_world @ R_touch_to_tool0 @ normal_in_touch
            current_pose = arm.get_pose()
            T_current_tool0 = np.eye(4)
            T_current_tool0[:3, :3] = R.from_quat(current_pose[3:]).as_matrix()
            T_current_tool0[:3, 3] = current_pose[:3]
            T_current_touch = T_current_tool0 @ T_touch_to_tool0
            print("Debug -2")
            print(grasp_position)
            print(grasp_normal)
            grasp_position = T_current_touch[:3,
                                             :3] @ grasp_position + T_current_touch[:3, 3]
            grasp_normal = T_current_touch[:3, :3] @ grasp_normal
            print("Debug -1")
            print(grasp_position)
            print(grasp_normal)

            cup.ur5_release()

            # move back from touch pose
            move_position = T_tool0_to_world[:3, 3] - 0.1 * touch_direction
            move_orientation = R.from_matrix(
                T_tool0_to_world[:3, :3]).as_quat()
            move_pose = move_position.tolist() + move_orientation.tolist()
            print('Debug0')
            arm.set_pose(move_pose, wait=True)
            print('Debug1')

            # grasp pose
            grasp_orientation = get_orientation(
                grasp_normal)  # R_grasp_to_world
            # R_tool0_to_world = R_grasp_to_world @ R_tool0_to_grasp
            grasp_orientation = (R.from_quat(
                grasp_orientation) * R.from_quat(config.init_orientation).inv()).as_quat()
            # grasp_orientation = ( R.from_quat(config.init_orientation) * \
            #                         R.from_quat(grasp_orientation)).as_quat()

            # pre suction pose
            move_position = grasp_position + \
                grasp_normal * (config.vacuum_length + 0.1)
            move_orientation = grasp_orientation
            move_pose = move_position.tolist() + move_orientation.tolist()
            print('Debug 2')
            arm.set_pose(move_pose, wait=True)
            # suction pose
            move_position = grasp_position + \
                grasp_normal * (config.vacuum_length)
            move_orientation = grasp_orientation
            move_pose = move_position.tolist() + move_orientation.tolist()
            print('Debug 3')
            arm.set_pose(move_pose, wait=True)

            # cup suction and place
            cup.ur5_grasp()
            grasp_postpose = arm.get_pose()
            grasp_postpose[2] = 0.4
            arm.set_pose(grasp_postpose, wait=True)
            time.sleep(3)
            # arm.set_positions(static_positions["place_pre_positions"], wait=True)
            # success, metric = recorder.check()
            # arm.set_positions(static_positions["place_positions"], wait=True)
            release_pose = np.array([0.5, -0.5, 0.25 + config.vacuum_length])
            release_rotation = R.from_euler(
                'xyz', [180, 0, -90], degrees=True).as_quat()
            final_release_pose = release_pose.tolist() + release_rotation.tolist()
            arm.set_pose(final_release_pose, wait=True)
            cup.ur5_release()


if __name__ == '__main__':
    main()