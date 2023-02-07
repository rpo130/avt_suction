import os
from turtle import position
import numpy as np
from chart_utils import analyse_chart, refine_grasp_normal

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
    # initialize using origin chart .obj file
    obj_file = os.path.join(os.path.dirname(__file__), "initial_sheet.obj")
    np.random.seed(777)

    obj = import_obj(obj_file)
    verts = np.array(obj.vertices)
    verts[:, 0] += (np.random.rand(verts.shape[0]) * 2 - 1) * 0.002
    faces = np.array(obj.faces) - 1
    
    grasp_position = np.array([0, 0, 0])
    # grasp_normal = np.array([-1, 0.0, 0.0])
    grasp_normal = np.array([-0.01, 0.005, 0.005])
    grasp_normal = grasp_normal / np.linalg.norm(grasp_normal)

    cup_radius = 0.005
    cup_height = 0.01
    cup_samples = 8
    spring_threshold = 0.1

    print(np.linalg.norm(grasp_normal))
    is_graspable = analyse_chart(verts, faces, grasp_position, grasp_normal, cup_radius, cup_height, cup_samples, spring_threshold, show=True)
    print(is_graspable)

    grasp_normal = refine_grasp_normal(verts, faces, grasp_position, grasp_normal, cup_radius, cup_samples)

    print(np.linalg.norm(grasp_normal))
    is_graspable = analyse_chart(verts, faces, grasp_position, grasp_normal, cup_radius, cup_height, cup_samples, spring_threshold, show=True)
    print(is_graspable)
