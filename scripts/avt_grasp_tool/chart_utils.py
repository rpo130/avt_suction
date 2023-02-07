from tkinter.tix import Tree
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def show_chart(verts, faces):
    fig = plt.figure("chart")
    plt.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color = 'g', alpha = 0.1)
    axis_range = [verts.min(), verts.max()]
    ax.set_xlim(*axis_range)
    ax.set_ylim(*axis_range)
    ax.set_zlim(*axis_range)
    plt.show()

def triangle_intersect(p1, p2, p3, q, w):
    touch_point = None
    A1 = 0.5 * w @ np.cross(p2 - q, p3 - q).T 
    A2 = 0.5 * w @ np.cross(p3 - q, p1 - q).T
    A3 = 0.5 * w @ np.cross(p1 - q, p2 - q).T 
    # check in trangle
    if (A1 <= 0 and A2 <= 0 and A3 <= 0) \
        or (A1 >= 0 and A2 >= 0 and A3 >= 0):
        # interpolate
        touch_point = ((A1 * p1) + (A2 * p2) + (A3 * p3)) / (A1 + A2 + A3)
    return touch_point

def ray_intersect(verts, faces, point, direction):
    for i, face in enumerate(faces):
        touch_point = triangle_intersect(verts[face[0]], verts[face[1]], verts[face[2]], point, direction)
        if touch_point is not None:
            return touch_point
    return touch_point

def refine_grasp_normal(verts, faces, grasp_position, grasp_normal, cup_radius, cup_samples):
    """
    all inputs should be represented in world frame 
    """
    grasp_normal = grasp_normal / np.linalg.norm(grasp_normal)
    
    # get transformation from world to cup frame
    rvec = np.cross(grasp_normal, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rvec)
    if (sin_theta > 0):
        rvec /= sin_theta
    R_world_to_cup = R.from_rotvec(np.arcsin(sin_theta) * rvec).as_matrix()
    t_world_to_cup = - grasp_position

    # transform the verts to cup frame
    verts = verts @ R_world_to_cup.T + t_world_to_cup
    
    # loop for each point on cup ring
    touch_points = []
    for i in range(cup_samples):
        phi = (i / cup_samples) * 2 * np.pi
        x = np.cos(phi) * cup_radius
        y = np.sin(phi) * cup_radius
        touch_point = ray_intersect(verts, faces, np.array([x, y, 0]), np.array([0, 0, -1]))
        if touch_point is not None:
            touch_points.append(touch_point)
    touch_points = np.array(touch_points)

    if touch_points.shape[0] >= 3:
        # refine grasp normal by find minimum eigen value
        u, s, vh = np.linalg.svd(touch_points, full_matrices=True)
        grasp_normal = vh[-1]
        if grasp_normal[-1] < 0:
            grasp_normal *= -1
    
    return grasp_normal @ R_world_to_cup

def check_spring(verts, faces, cup_radius, cup_height, cup_samples, spring_threshold, show_fig=None):
    # spring model
    apex = np.array([0, 0, cup_height])
    sc = np.sqrt(cup_radius**2 + cup_height**2)
    sp = np.sqrt(2 * cup_radius**2 * (1 - np.cos(360 / cup_samples / 180 * np.pi)))
    sf = np.sqrt(2 * cup_radius**2 * (1 - np.cos(360 / cup_samples / 180 * np.pi * 2)))
    
    # loop for each point on cup ring
    direction = np.array([0, 0, -1])
    loss_length = max(sc, max(sp, sf)) * (1 + spring_threshold)
    touch_points_c = np.zeros((cup_samples, 3))
    for i in range(cup_samples):
        phi = (i / cup_samples) * 2 * np.pi
        x = np.cos(phi) * cup_radius
        y = np.sin(phi) * cup_radius
        point = np.array([x, y, 0])
        touch_point = ray_intersect(verts, faces, point, direction)
        if touch_point is not None:
            touch_points_c[i] = touch_point
        else:
            touch_points_c[i] = point + direction * loss_length
    
    # get all sample points
    touch_points_p = np.zeros((cup_samples, cup_samples+1, 3))
    touch_points_f = np.zeros((cup_samples, cup_samples+1, 3))
    for i in range(cup_samples):
        p = touch_points_c[i]
        p_n = touch_points_c[(i+1)%cup_samples] # next point
        p_nn = touch_points_c[(i+2)%cup_samples] # next next point
        
        touch_points_p[i, 0] = p
        touch_points_f[i, 0] = p
        for j in range(1, cup_samples+1):
            rate = j / cup_samples
            
            # perimeter spring
            p_p = (1-rate) * p + (rate) * p_n
            touch_point = ray_intersect(verts, faces, p_p, direction)
            if touch_point is not None:
                touch_points_p[i, j] = touch_point
            else:
                touch_points_p[i, j] = p_p
                touch_points_p[i, j, 2] = - loss_length

            # flexion spring
            p_f = (1-rate) * p + (rate) * p_nn
            touch_point = ray_intersect(verts, faces, p_f, direction)
            if touch_point is not None:
                touch_points_f[i, j] = touch_point
            else:
                touch_points_f[i, j] = p_f
                touch_points_f[i, j, 2] = - loss_length

    sc_ = np.linalg.norm(touch_points_c - apex, axis=-1)
    sp_ = np.linalg.norm(touch_points_p[:, :-1] - touch_points_p[:, 1:], axis=-1).sum(axis=-1)
    sf_ = np.linalg.norm(touch_points_f[:, :-1] - touch_points_f[:, 1:], axis=-1).sum(axis=-1)
    
    is_graspable = True
    is_graspable = is_graspable and np.all(np.abs(sc_ - sc) < sc * spring_threshold)
    # print(sc_)
    # print(sc)
    # print(np.abs(sc_ - sc))
    # print(sc * spring_threshold)
    is_graspable = is_graspable and np.all(np.abs(sp_ - sp) < sp * spring_threshold)
    is_graspable = is_graspable and np.all(np.abs(sf_ - sf) < sf * spring_threshold)

    # show chart and grasp
    if show_fig is not None:
        # fig = plt.figure("grasp")
        # plt.clf()
        
        # show chart
        ax = show_fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, alpha=0.2)
        
        # draw projected points
        ax.scatter(touch_points_p[:, :, 0], touch_points_p[:, :, 1], touch_points_p[:, :, 2], c='black')
        if is_graspable:
            ax.scatter(touch_points_f[:, :, 0], touch_points_f[:, :, 1], touch_points_f[:, :, 2], c='blue')
        else:
            ax.scatter(touch_points_f[:, :, 0], touch_points_f[:, :, 1], touch_points_f[:, :, 2], c='red')
        ax.plot([apex[0], 0], [apex[1], 0], [apex[2], 0], c='r')

        # axis_range = [verts.min(), verts.max()]
        # ax.set_xlim(*axis_range)
        # ax.set_ylim(*axis_range)
        # ax.set_zlim(*axis_range)

        ax.set_xlim(-2.5*0.0047, 2.5*0.0047)
        ax.set_ylim(-2.5*0.0047, 2.5*0.0047)
        # ax.set_zlim(0.0, 2*1.0)
        ax.set_zlim(-0.002, 0.006)
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')

        # plt.show()

    return is_graspable

def analyse_chart(verts, faces, 
        grasp_position, grasp_normal, 
        cup_radius, cup_height, cup_samples, spring_threshold,
        show_fig=None):
    """
    all inputs should be represented in world frame 
    """
    grasp_normal = grasp_normal / np.linalg.norm(grasp_normal)
    
    # get transformation from world to cup frame
    rvec = np.cross(grasp_normal, np.array([0, 0, 1]))
    sin_theta = np.linalg.norm(rvec)
    if sin_theta > 0:
        rvec /= sin_theta
        R_world_to_cup = R.from_rotvec(np.arcsin(sin_theta) * rvec).as_matrix()
    else:
        R_world_to_cup = np.eye(3)
    t_world_to_cup = - grasp_position @ R_world_to_cup.T

    # transform the verts to cup frame
    verts = verts @ R_world_to_cup.T + t_world_to_cup

    # analyse using srping model
    return check_spring(verts, faces, cup_radius, cup_height, cup_samples, spring_threshold, show_fig)
