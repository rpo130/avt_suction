import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from chart_utils import ray_intersect

class import_obj(object):
	def __init__(self, file):
		self.vertices = []
		self.faces = []
		with open(file) as f :
			for line in f:
				line = line.replace('//', '/')
				line = line.replace('\n', '')
				if line[:2] == "v ":
					self.vertices.append([float(v) for v in line.split(" ")[1:]])
				elif line[0] == "f":
					self.faces.append([int(s.split('/')[0]) for s in line.split(' ')[1:]])

if __name__=="__main__":
    verts = np.array([
        [0.1, -0.1, 0],
        [0., 0.1, 0],
        [-0.1, -0.1, 0],
    ])
    faces = np.array([
        [0, 1, 2]
    ])
    
    point = np.array([0.0, 0, 0.1])
    direction = np.array([0.2, 0.2, -1])
    direction /= np.linalg.norm(direction)
    touch_point = ray_intersect(verts, faces, point, direction)
    if touch_point is None:
        print("no touch point")

    fig = plt.figure("ray intersection")
    plt.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, alpha=0.5)
    axis_range = [verts.min(), verts.max()]
    point_ = point + direction * 0.2
    ax.plot([point[0], point_[0]], [point[1], point_[1]], [point[2], point_[2]], c='r')
    if touch_point is not None:
        ax.scatter([point[0], touch_point[0]], [point[1], touch_point[1]], [point[2], touch_point[2]], c='black')
    ax.set_xlim(*axis_range)
    ax.set_ylim(*axis_range)
    ax.set_zlim(*axis_range)
    plt.show()