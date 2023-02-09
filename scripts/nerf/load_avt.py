import os
import torch
import numpy as np
import imageio 
import json
from scipy.spatial.transform import Rotation as R

def load_avt_data(basedir, need_fix_pose=True):

    trans_file = os.path.join(basedir, 'transforms.json')
    if not os.path.exists(trans_file):
        trans_file = os.path.join(basedir, 'transforms_train.json')

    with open(trans_file, 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_poses = []
    imgs = []
    poses = []
        
    for frame in meta['frames'][::1]:
        file_path = frame['file_path']
        if '.png' not in file_path:
            file_path = file_path + '.png'
        fname = os.path.join(basedir, file_path)
        img = imageio.imread(fname)
        imgs.append(img)
        T_cam_to_world = np.array(frame['transform_matrix'])
        if need_fix_pose:
            #fix poses
            T_cam_face_to_world = T_cam_to_world

            T_img_to_cam_face = np.eye(4) 
            T_img_to_cam_face[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            T_cam_to_world = T_cam_face_to_world @ T_img_to_cam_face

        poses.append(T_cam_to_world)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_poses.append(poses)
        
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
        
    fx = meta['fx']
    fy = meta['fy']
    cx = meta['cx']
    cy = meta['cy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    focal = fx

    return [H, W, focal], K, imgs, poses