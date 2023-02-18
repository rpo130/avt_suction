import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from common import *
import matplotlib.pyplot as plt
import json
import os

import pyngp as ngp

def create_depthmaps(transforms_file, snapshot_file, poses, sigma_thrsh=15):
	transforms_file = transforms_file

	# poses = []
	# img_names = []

	with open(transforms_file, 'r') as tf:
		meta = json.load(tf)
		for frame in meta['frames']:
			# poses.append(frame['transform_matrix'])
			basename = os.path.basename(frame['file_path'])
			if not os.path.splitext(basename)[1]:
				basename = basename + ".png"

	width = int(meta['w'])
	height = int(meta['h'])
	camera_angle_x = meta['camera_angle_x']

	testbed = ngp.Testbed()
	testbed.root_dir = ROOT_DIR

	testbed.load_file(transforms_file)

	# Load a trained NeRF model
	print("Loading snapshot ", snapshot_file)
	testbed.load_snapshot(snapshot_file)

	# testbed.nerf.sharpen = float(0)
	# testbed.exposure = float(0)
	testbed.shall_train = False

	testbed.nerf.render_with_camera_distortion = True
	testbed.snap_to_pixel_centers = True
	spp = 8
	testbed.nerf.rendering_min_transmittance = 1e-4
	testbed.fov_axis = 0
	testbed.fov = camera_angle_x * 180 / np.pi
	
	fx = meta['fl_x']
	fy = meta['fl_y']
	cx = meta['cx']
	cy = meta['cy']
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
		print(f'rendering dex')
		depth = testbed.render(width, height, spp, True)
		depth = depth[..., 0]
		depths.append(depth)

		testbed.render_mode = ngp.RenderMode.Shade
		print(f'rendering rgb')
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
