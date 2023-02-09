import json
import numpy as np
import pathlib
import os

config_file = '/home/amax_djh/code/ysl/nerf-pytorch/configs/avt_data_glass_20230204_8.txt'
# config_file = '/home/amax_djh/code/ysl/instant-DexNerf-main/data/nerf/canister/transforms.json'

def nerf_info(config_path=config_file):
    from nerf.run_nerf import config_parser, load_avt_data
    basedir = pathlib.Path(config_path).resolve().parent.parent

    parser = config_parser()
    args = parser.parse_args(f"--config {config_path}")

    if not pathlib.Path(args.datadir).is_absolute():
        args.datadir = os.path.join(basedir, args.datadir)

    if args.ft_path is not None and not pathlib.Path(args.ft_path).is_absolute():
        args.ft_path = os.path.join(basedir, args.ft_path)

    if not pathlib.Path(args.basedir).is_absolute():
        args.basedir = os.path.join(basedir, args.basedir)

    hwf, K, _, poses = load_avt_data(args.datadir, False)
    return hwf,K,poses
	
    # poses = []
    # with open(config_path, 'r') as tf:
    #     meta = json.load(tf)
    #     for frame in meta['frames']:
    #         poses.append(frame['transform_matrix'])
    # width = int(meta['w'])
    # height = int(meta['h'])
    # camera_angle_x = meta['camera_angle_x']
    # K = np.eye(3)
    # K[0,0] = meta['fl_x']
    # K[1,1] = meta['fl_y']
    # K[0,2] = meta['cx']
    # K[1,2] = meta['cy']
    # focal = K[0,0]

    # return [height, width, focal], K, poses

def nerf_camera(T_c2ws, config=config_file):
    from nerf.run_nerf import run
    color, depth, dexdepth, K = run(config, T_c2ws)
    print(color)
    print(depth)
    return color, dexdepth, K

    # from ngp_dex.scripts import eval
    # snapfile = '/home/amax_djh/code/avt_grasp_ws/src/avt_suction/scripts/ngp_dex/base.msgpack'
    # color,depth,K = eval.create_depthmaps(config, snapfile, T_c2ws, sigma_thrsh=15)
    # print(color)
    # print(depth)
    # return color, depth, K