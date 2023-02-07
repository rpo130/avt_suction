import os
import numpy as np
from chart_utils import show_chart
import pdb

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

    obj = import_obj(obj_file)
    verts = np.array(obj.vertices)
    faces = np.array(obj.faces) - 1
    show_chart(verts, faces)
