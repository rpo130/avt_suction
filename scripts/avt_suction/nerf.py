import json
import numpy as np
import pathlib
import os

config_file = '/home/amax_djh/code/ysl/instant-DexNerf-main/data/nerf/canister/transforms.json'
# config_file = '/home/amax_djh/code/ysl/instant-ngp/data/nerf/avt_data_20230210_1/transforms_ngp.json'


class Nerf:
    def __init__(self, config=None):
        if config is None:
            config = '/home/amax_djh/code/ysl/nerf-pytorch/configs/avt_data_glass_20230204_7.txt'
            # config = '/home/amax_djh/code/ysl/nerf-pytorch/configs/lego.txt'

        self.config = config
    
    def nerf_info(self):
        from nerf.run_nerf import config_parser, load_avt_data
        basedir = pathlib.Path(self.config).resolve().parent.parent

        parser = config_parser()
        args = parser.parse_args(f"--config {self.config}")

        if not pathlib.Path(args.datadir).is_absolute():
            args.datadir = os.path.join(basedir, args.datadir)

        if args.ft_path is not None and not pathlib.Path(args.ft_path).is_absolute():
            args.ft_path = os.path.join(basedir, args.ft_path)

        if not pathlib.Path(args.basedir).is_absolute():
            args.basedir = os.path.join(basedir, args.basedir)

        hwf, K, _, poses = load_avt_data(args.datadir, False)
        return hwf,K,poses

    def nerf_camera(self, T_c2ws):
        from nerf.run_nerf import run
        color, depth, dexdepth, K = run(self.config, T_c2ws)
        return color, dexdepth, K

class NerfNgp:
    def __init__(self, config=None):
        if config is None:
            config = '/home/amax_djh/ysl/instant-ngp/data/nerf/avt_data_glass_20230218_5/transforms_ngp.json'
        self.config = config
        self.snapfile = '/home/amax_djh/ysl/instant-ngp/data/nerf/avt_data_glass_20230218_5/transforms_ngp_base.ingp'
    
    def nerf_info(self):
        poses = []
        with open(self.config, 'r') as tf:
            meta = json.load(tf)
            for frame in meta['frames']:
                poses.append(frame['transform_matrix'])
        width = int(meta['w'])
        height = int(meta['h'])
        camera_angle_x = meta['camera_angle_x']
        K = np.eye(3)
        K[0,0] = meta['fl_x']
        K[1,1] = meta['fl_y']
        K[0,2] = meta['cx']
        K[1,2] = meta['cy']
        focal = K[0,0]

        return [height, width, focal], K, poses

    def nerf_camera(self, T_c2ws):
        from ngp.scripts import eval
        snapfile = self.snapfile
        color,depth,K = eval.create_depthmaps(self.config, snapfile, T_c2ws, sigma_thrsh=15)
        return color, depth, K