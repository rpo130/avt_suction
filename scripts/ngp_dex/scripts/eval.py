import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from common import *
import matplotlib.pyplot as plt
import json
import os

import pyngp as ngp

def create_depthmaps(transforms_file, snapshot_file, poses, sigma_thrsh=15):
	with open(transforms_file, 'r') as tf:
		meta = json.load(tf)

	width = int(meta['w'])
	height = int(meta['h'])
	camera_angle_x = meta['camera_angle_x']
	camera_angle_y = meta['camera_angle_y']
	fx = meta['fl_x']
	fy = meta['fl_y']
	cx = meta['cx']
	cy = meta['cy']

	mode = ngp.TestbedMode.Nerf
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
	testbed = ngp.Testbed(mode)
	# testbed.nerf.sharpen = float(0)
	testbed.shall_train = False

	# Load a trained NeRF model
	print("Loading snapshot ", snapshot_file)
	testbed.load_snapshot(snapshot_file)
	testbed.nerf.render_with_camera_distortion = True
	testbed.snap_to_pixel_centers = True
	spp = 1
	testbed.nerf.rendering_min_transmittance = 1e-4
	testbed.fov_axis = 0
	testbed.fov = camera_angle_x * 180 / np.pi
	testbed.fov_axis = 1
	testbed.fov = camera_angle_y * 180 / np.pi


	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])


	# Set camera matrix
	colors = []
	depths = []
	for c2w_matrix in poses:
		testbed.set_nerf_camera_matrix(np.matrix(c2w_matrix)[:-1, :])
		
		# Set render mode
		testbed.render_mode = ngp.RenderMode.Depth
		# Adjust DeX threshold value
		testbed.dex_nerf = True
		testbed.sigma_thrsh = sigma_thrsh

		# Render estimated depth
		depth_raw = testbed.render(width, height, spp, True)  # raw depth values (float, in m)
		depth_raw = depth_raw[..., 0]
		# depth_int = 1000 * depth_raw  # transform depth to mm
		# depth_int = depth_int.astype(np.uint8)
		depths.append(depth_raw)

		testbed.render_mode = ngp.RenderMode.Shade
		rgb = testbed.render(width, height, spp, True)
		rgb = rgb[...,0:3]
		rgb = linear_to_srgb(rgb)
		rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
		colors.append(rgb)


	ngp.free_temporary_memory()
	return colors, depths, K

if __name__ == '__main__':
	p = np.array([
        [
          -0.15399402818711255,
          -0.1679341184813246,
          0.9736960363134933,
          -0.544839962479
        ],
        [
          -0.02392861174898336,
          -0.9845278462284572,
          -0.1735866974757061,
          -0.913826727587
        ],
        [
          0.9877819905335761,
          -0.050030509198451194,
          0.14759297858255477,
          -0.18107418152299995
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ])
	depths = create_depthmaps(transforms_file=os.path.join(ROOT_DIR, "data/nerf/canister/transforms.json"), 
					snapshot_file=os.path.join(SCRIPTS_FOLDER, "base.msgpack"),
					poses=[p])

	plt.imshow(depths[0])
	plt.show()
