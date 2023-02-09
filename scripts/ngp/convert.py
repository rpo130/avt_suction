import json
import os
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

def convert(basedir):
    with open(os.path.join(basedir, 'transforms.json')) as fp:
        meta = json.load(fp)

    img_path = os.path.join(basedir, meta['frames'][0]['file_path']+".png")
    img = cv.imread(img_path)
    H,W = img.shape[0], img.shape[1]
    fx,fy = meta['fx'],meta['fy']

    meta_ngp = {
        'fl_x':fx,
        'fl_y':fy,
        'camera_angle_x':np.arctan(.5*W/fx)*2,
        'camera_angle_y':np.arctan(.5*H/fy)*2,
        'cx':meta['cx'],
        'cy':meta['cy'],
        'w':W,
        'h':H,
        'aabb_scale':4,
        'frames':meta['frames']
    }

    for frame in meta_ngp['frames']:
        T_cf_2_world = np.array(frame['transform_matrix'])
        T_img_to_cam_face = np.eye(4)
        T_img_to_cam_face[:3,:3] = R.from_euler("xyz", [180,0,0], degrees=True).as_matrix()
        T_cam_to_world = T_cf_2_world @ T_img_to_cam_face
        frame['transform_matrix'] = T_cam_to_world.tolist()

    filepath = os.path.join(basedir, 'transforms_ngp.json')
    with open(filepath, 'w') as fp:
        fp.write(json.dumps(meta_ngp, indent=2))

def gen_test(basedir):
    with open(os.path.join(basedir, 'transforms_ngp.json')) as fp:
        meta_ngp = json.load(fp)

    frames = []
    T_ = np.array(meta_ngp['frames'][0]['transform_matrix'])
    
    T_tran = np.eye(4)
    T_tran[2,3] = 0.1
    T_ = T_ @ T_tran
    
    frames.append({"file_path": str(3), "transform_matrix": T_.tolist(), })
    
    meta_ngp['frames'] = frames
    filepath = os.path.join(basedir, 'transforms_ngp_test.json')
    with open(filepath, 'w') as fp:
        fp.write(json.dumps(meta_ngp, indent=2))

# convert('./data/nerf/avt_data_glass_light_20230115_1/')
# gen_test('./data/nerf/avt_data_glass_light_20230115_1/')

convert('./data/nerf/avt_data_glass_20230204_8/')