import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import random
import trimesh
import numpy as np
import pyrender
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils import bbox, misc, kit
from pyrender.constants import RenderFlags
from scipy.spatial.transform import Rotation as R

LIMITS = [(-0.4, 0.4), (0.1, 0.9), (-0.4, 0.4)]   # skeleton

def save_mesh_vis(trimesh_obj, out_path):
    # print("[VIS] Visualizing mesh...")
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj, smooth=True)
    scene = pyrender.Scene(bg_color=[0,0,0,0],
                           ambient_light=[0.3,0.3,0.3,1.0])

    # squirtle blastoise rotated and cam looking down 27 deg
    # this is for rotating the mesh to render certain parts from the side
    rotation_mat_y = np.identity(4)
    rot = R.from_euler('y', -30, degrees=True).as_matrix()
    rotation_mat_y[:3, :3] = rot
    scene.add(mesh, pose=rotation_mat_y)

    mag = 0.65
    cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag)
    translation_mat = np.array([
        [1, 0, 0, -0.0],
        [0, 1, 0, 0.35],
        [0, 0, 1, 1.5],
        [0, 0, 0, 1]
    ])

    # squirtle blastoise rotated and cam looking down 27 deg
    rotation_mat2 = np.zeros((4, 4))
    rot2 = R.from_euler('x', -27, degrees=True).as_matrix()
    rotation_mat2[:3, :3] += rot2
    rotation_mat2[3, 3] = 1
    cam_pose = rotation_mat2 @ translation_mat
    
    # cam_pose = translation_mat
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=cam_pose)

    # pyrender.Viewer(scene)
    flags = RenderFlags.RGBA
    r = pyrender.OffscreenRenderer(viewport_width=1500,
                                   viewport_height=1500,
                                   point_size=1.0)
    color, _ = r.render(scene, flags=flags)
    r.delete()
    im = Image.fromarray(color)
    im.save(out_path)


def vis_voxels(voxels, a):
    ax = plt.axes(projection='3d')
    ax.voxels(voxels, alpha=a)
    # ax.view_init(30, 45)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    # plt.axis('off')
    return plt


def show_verts_to_faces(verts, joints_to_faces, num_joints,
                        limits=LIMITS, highlight_part_idx=-1):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])

    for i in tqdm(range(num_joints)):
        if highlight_part_idx != -1:
            alpha = 1 if i == highlight_part_idx else 0.01
        else:
            alpha = 1

        r = random.random()
        g = random.random()
        b = random.random()
        color = np.array([[r, g, b]])

        joint_faces = joints_to_faces[i]
        
        for f in joint_faces:
            tri_verts = verts[f]
            tri_face = Poly3DCollection([tri_verts])
            tri_face.set_facecolor(color)
            tri_face.set_alpha(alpha)
            tri_face.set_edgecolor(color)
            ax.add_collection3d(tri_face)

    ax.view_init(45, 0)
    plt.axis('off')
    return plt, ax


def show_corresponding_patches(verts, joints_to_faces,
                               src_or_dest: str, groups, colors,
                               limits=LIMITS):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])

    num_corres = len(groups)
    for i in range(num_corres):
        color = colors[i]
        pair = groups[i]
        j = 0 if src_or_dest == 'src' else 1
        nodes = pair[j]
        for n in nodes:
            if n == -1:
                continue
            # if n != 6 and n!= 12:
            #     continue
            joint_faces = joints_to_faces[n]
            for f in joint_faces:
                tri_verts = verts[f]
                tri_face = Poly3DCollection([tri_verts])
                tri_face.set_facecolor(color)
                tri_face.set_alpha(1)
                tri_face.set_edgecolor(color)
                ax.add_collection3d(tri_face)

    ax.view_init(45, 0)
    ax.set_proj_type('ortho')
    plt.axis('off')
    return plt, ax


def save_corresponding_patches_trimesh(verts, faces,
                                       joints_to_verts, 
                                       joints_to_faces,
                                       src_or_dest: str, groups, colors,
                                       outpath):
    # commented out to render certain parts
    # which_part = 3
    vert_colors = np.ones((len(verts), 4))
    vert_colors[:, 3] = 255
    for i in range(len(groups)):
        color = colors[i][0] * 255
        color = np.array(misc.increase_saturation(color, 35))
        pair = groups[i]
        j = 0 if src_or_dest == 'src' else 1
        nodes = pair[j]
        for n in nodes:
            if n == -1:
                continue
            # if n != which_part:
            #     continue
            # if n != which_part and n != 11:
            #     continue
            joint_verts = joints_to_verts[n]
            vert_colors[joint_verts, :3] = color
    # faces = joints_to_faces[3]
    # faces2 = joints_to_faces[11]
    # faces = np.concatenate((faces, faces2))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    mesh.visual.vertex_colors = vert_colors
    save_mesh_vis(mesh, outpath)


def show_verts_to_faces_and_bones(verts, joints_to_faces, num_joints,
                                  bones,
                                  limits=LIMITS,
                                  highlight_part_idx=-1,
                                  rot=45):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])

    for i in tqdm(range(num_joints)):
        if highlight_part_idx != -1:
            alpha = 0.8 if i == highlight_part_idx else 0.01
        else:
            alpha = 0.8

        r = random.random()
        g = random.random()
        b = random.random()
        color = np.array([[r, g, b]])

        joint_faces = joints_to_faces[i]
        
        for f in joint_faces:
            tri_verts = verts[f]
            tri_face = Poly3DCollection([tri_verts])
            tri_face.set_facecolor(color)
            tri_face.set_alpha(alpha)
            tri_face.set_edgecolor(color)
            ax.add_collection3d(tri_face)

    for i in range(bones.shape[0]):
        coordinates = np.transpose(bones[i])
        ax.scatter(coordinates[0], coordinates[1], coordinates[2],
                   c='grey', alpha=1)
        ax.plot(coordinates[0], coordinates[1], coordinates[2], c='r')

    ax.view_init(rot, 0)
    plt.axis('off')
    return plt, ax


def scatter(points, limits=LIMITS):
    arr = np.transpose(points)
    x = arr[0]
    y = arr[1]
    z = arr[2]
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])
    ax.scatter(x, y, z, alpha=1, c='grey')
    return plt, ax


def vis_points(points, limits=LIMITS):
    plt, _ = scatter(points, limits)
    return plt


def vis_sample_points(points, values, limits=LIMITS):
    model_points = points
    model_values = values.astype('bool').flatten()
    filtered_points = model_points[model_values]
    plt, ax = scatter(filtered_points, limits)
    ax.view_init(45, 0)
    # plt.axis('off')
    return plt


def vis_vector_field(starting_points, directions, limits=LIMITS):
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])
    ax.quiver(starting_points[:, 0],
              starting_points[:, 1],
              starting_points[:, 2],
              directions[:, 0],
              directions[:, 1],
              directions[:, 2])
    return plt


def show_bones(bones, limits=LIMITS, text=False):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])
    for i in range(bones.shape[0]):
        coordinates = np.transpose(bones[i])
        ax.scatter(coordinates[0], coordinates[1], coordinates[2],
                   c='grey', alpha=1)
        ax.plot(coordinates[0], coordinates[1], coordinates[2], c='r')
        if text:
            mids = np.mean(coordinates, axis=1)
            ax.text(mids[0], mids[1], mids[2], str(i), 'y', fontsize='x-small')
    ax.view_init(45, 0)
    plt.axis('off')
    return plt


def show_corresponding_bones(bones, groups, src_or_dest: str,
                             custom_colors=None, limits=LIMITS, text=False,
                             rot=45):
    # corresponds to camear parameters in blender render script
    r = R.from_euler('z', -27, degrees=True)
    # r = R.from_euler('z', -30, degrees=True)
    # r = R.from_euler('z', -3, degrees=True)
    mat = r.as_matrix()
    rot_mat = np.eye(4)
    rot_mat[:3, :3] = mat
    bones_all = np.reshape(bones, (-1, 3))
    bones_all = kit.transform_points(bones_all, rot_mat, 'zeros')
    bones = np.reshape(bones_all, (-1, 2, 3))

    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])
    num_corres = len(groups)
    
    show_corresp = True
    use_tableau = True

    if not use_tableau:
        random.seed(285714)
    else:
        ta = misc.get_tab_20()
        x = num_corres // len(ta)
        ta = ta * (x + 1)
        ta = ta[:num_corres]
        random.seed(2)
        random.shuffle(ta)

    colors = [] if custom_colors is None else custom_colors
    for i in range(num_corres):
        if not use_tableau:
            r = 0.6
            g = 0.6
            b = 0.6
        else:
            # use tableau 20
            r, g, b = ta[i]

            r /= 255; g /= 255; b /= 255

        if show_corresp:
            color = np.array([[r, g, b]])
        else:
            color = np.array([[0.6, 0.6, 0.6]])
        colors.append(color)
        final_color = color if custom_colors is None else custom_colors[i]

        pair = groups[i]
        j = 0 if src_or_dest == 'src' else 1
        nodes = pair[j]

        for n in nodes:
            if n == -1:
                continue
            coordinates = np.transpose(bones[n])
            if show_corresp:
                ax.scatter(coordinates[0], coordinates[1], coordinates[2],
                            c=[final_color*0.6], s=100, alpha=1)
                ax.plot(coordinates[0], coordinates[1], coordinates[2],
                        c=final_color, linewidth=8)
            else:
                ax.scatter(coordinates[0], coordinates[1], coordinates[2],
                        c='black', s=15, alpha=1)
                ax.plot(coordinates[0], coordinates[1], coordinates[2],
                        c=final_color, linewidth=8)
            if text:
                mids = np.mean(coordinates, axis=1)
                ax.text(mids[0], mids[1], mids[2], str(n), 'y',
                fontsize='x-small')

    ax.view_init(rot, 0)
    ax.set_proj_type('ortho')
    plt.axis('off')
    return plt, colors


def show_joints(joints, limits=LIMITS, text=False):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])
    for i in range(joints.shape[0]):
        ax.scatter(joints[i][0], joints[i][1], joints[i][2], c='r')
        if text:
            ax.text(joints[i][0], joints[i][1], joints[i][2], str(i), 'y')
    ax.view_init(45, 0)
    plt.axis('off')
    return plt


def show_joints_and_local_coors(joints, local_coords, limits=LIMITS):
    """Shows joints and their local coordinate frames.

    Args:
        joints ([type]): [description]
        start ([type]): [description]
        end ([type]): [description]
    """
    plt, ax = scatter(joints, limits)
    y = np.linspace(-0.5, 0.5, 10)
    z = np.linspace(-0.5, 0.5, 10)
    yy, zz = np.meshgrid(y, z)
    # ax.plot_surface(0, yy, zz, alpha=0.2)
    for i in range(local_coords.shape[0]):
        system = local_coords[i]
        orig_coor = system[0]
        # [r, g, b]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for j in range(2, 5):
            # 3 iterations for each of x, y, z axes
            ax.quiver(orig_coor[0], orig_coor[1], orig_coor[2],
                      system[j][0], system[j][1], system[j][2],
                      length=0.06, color=colors[j-2])
    ax.view_init(45, 0)
    plt.axis('off')
    # plt.show()
    return plt


def show_bones_and_local_coors(bones, local_coords, limits=LIMITS, text=False):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim3d(limits[0][0], limits[0][1])
    ax.set_ylim3d(limits[1][0], limits[1][1])
    ax.set_zlim3d(limits[2][0], limits[2][1])
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    for i in range(bones.shape[0]):
        coordinates = np.transpose(bones[i])
        ax.scatter(coordinates[0], coordinates[1], coordinates[2],
                   c='grey', s=10, alpha=0.7)
        ax.plot(coordinates[0], coordinates[1], coordinates[2],
                c='black', alpha=0.5, linewidth=4)
        if text:
            mids = np.mean(coordinates, axis=1)
            ax.text(mids[0], mids[1], mids[2], str(i), 'y',
            fontsize='x-small')
    y = np.linspace(-0.5, 0.5, 10)
    z = np.linspace(-0.5, 0.5, 10)
    yy, zz = np.meshgrid(y, z)
    # ax.plot_surface(0, yy, zz, alpha=0.2)
    for i in range(local_coords.shape[0]):
        system = local_coords[i]
        orig_coor = system[0]
        # [r, g, b]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for j in range(2, 5):
            # 3 iterations for each of x, y, z axes
            ax.quiver(orig_coor[0], orig_coor[1], orig_coor[2],
                      system[j][0], system[j][1], system[j][2],
                      length=0.06, color=colors[j-2])
    ax.view_init(45, 0)

    plt.axis('off')

    return plt


def show_coordinate_system(local_coords, start, end):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim3d(start, end)
    ax.set_ylim3d(start, end)
    ax.set_zlim3d(start, end)
    y = np.linspace(-0.5, 0.5, 10)
    z = np.linspace(-0.5, 0.5, 10)
    yy, zz = np.meshgrid(y, z)
    # ax.plot_surface(0, yy, zz, alpha=0.2)
    for i in range(local_coords.shape[0]):
        system = local_coords[i]
        orig_coor = system[0]
        # [r, g, b]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for j in range(2, 5):
            # 3 iterations for each of x, y, z axes
            ax.quiver(orig_coor[0], orig_coor[1], orig_coor[2],
                      system[j][0], system[j][1], system[j][2],
                      length=0.06, color=colors[j-2])
    ax.view_init(45, 0)
    plt.axis('off')
    plt.show()
    return plt


def samp_points(points, values, limits=LIMITS):
    """Scatter plot points according to their values

    Args:
        points (numpy.ndarray): sampled points from a single model
        values (numpy.ndarray): values associated with the sampled points
        start (int): an integer representing the starting point of an axis
        end (int): an integer representing the ending point of an axis

    Returns:
        [pyplot, Axes3D]: plot and axes containing data points
    """
    model_points = points
    model_values = values.astype('bool').flatten()
    filtered_points = model_points[model_values]
    plt, ax = scatter(filtered_points, limits)
    return plt, ax


def show_points_and_bones(points, values, bones, limits=LIMITS):
    """Shows skeleton overlayed with sampled points

    Args:
        points (numpy.ndarray): sampled points from a single model
        values (numpy.ndarray): values associated with the sampled points
        bones (numpy.ndarray): bones of a single model
        start (int): an integer representing the starting point of an axis
        end (int): an integer representing the ending point of an axis
    """
    plt, ax = samp_points(points, values, limits)
    for i in range(bones.shape[0]):
        coordinates = np.transpose(bones[i])
        ax.scatter(coordinates[0], coordinates[1], coordinates[2], c='r')
        ax.plot(coordinates[0], coordinates[1], coordinates[2], c='r')
        mids = np.mean(coordinates, axis=1)
        # ax.text(mids[0], mids[1], mids[2], str(i), 'z')
    ax.view_init(45, 0)
    plt.axis('off')
    plt.show()
    return plt


def show_matplotlib_mesh(verts, faces, start, end):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(start, end)
    ax.set_ylim3d(start, end)
    ax.set_zlim3d(start, end)

    color = np.array([[0, 0, 1]])
    edge_color = np.array([[0, 0, 0]])
    alpha = 1
    for f in faces:
        tri_verts = verts[f]
        tri_face = Poly3DCollection([tri_verts])
        tri_face.set_facecolor(color)
        tri_face.set_alpha(alpha)
        tri_face.set_edgecolor(edge_color)
        ax.add_collection3d(tri_face)

    # ax.view_init(45, 0)
    # plt.axis('off')
    return plt, ax


def show_bounding_box(verts, faces, bbox: bbox.Bbox, start, end):
    fig = plt.figure(figsize=(5, 5), dpi=80)
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    fig.add_axes(ax)
    ax.set_xlim3d(start, end)
    ax.set_ylim3d(start, end)
    ax.set_zlim3d(start, end)

    cube_definition = np.zeros((4, 3), dtype=np.float32)
    botleft = np.zeros((1, 3), dtype=np.float32)
    botleft[0, 0] = bbox.origin_in_local[0] - bbox.scaffold.xneg
    botleft[0, 1] = bbox.origin_in_local[1] - bbox.scaffold.yneg
    botleft[0, 2] = bbox.origin_in_local[2] - bbox.scaffold.zneg
    xlen = bbox.scaffold.xneg + bbox.scaffold.xpos
    ylen = bbox.scaffold.yneg + bbox.scaffold.ypos
    zlen = bbox.scaffold.zneg + bbox.scaffold.zpos
    cube_definition[0] = botleft
    cube_definition[1] = botleft + np.array([xlen, 0, 0])
    cube_definition[2] = botleft + np.array([0, ylen, 0])
    cube_definition[3] = botleft + np.array([0, 0, zlen])

    plot_cube(cube_definition, ax)

    # ax.view_init(45, 0)
    # plt.axis('off')
    return plt, ax


def show_bounding_box_and_points(points, bbox: bbox.Bbox, start, end):
    plt, ax = scatter(
        points,
        limits=[[start, end], [start, end], [start, end]])

    cube_definition = np.zeros((4, 3), dtype=np.float32)
    botleft = np.zeros((1, 3), dtype=np.float32)
    botleft[0, 0] = bbox.origin_in_local[0] - bbox.scaffold.xneg
    botleft[0, 1] = bbox.origin_in_local[1] - bbox.scaffold.yneg
    botleft[0, 2] = bbox.origin_in_local[2] - bbox.scaffold.zneg
    xlen = bbox.scaffold.xneg + bbox.scaffold.xpos
    ylen = bbox.scaffold.yneg + bbox.scaffold.ypos
    zlen = bbox.scaffold.zneg + bbox.scaffold.zpos
    cube_definition[0] = botleft
    cube_definition[1] = botleft + np.array([xlen, 0, 0])
    cube_definition[2] = botleft + np.array([0, ylen, 0])
    cube_definition[3] = botleft + np.array([0, 0, zlen])

    plot_cube(cube_definition, ax)

    return plt, ax


def plot_cube(cube_definition, ax):
    cube_definition_array = [
        np.array(item)
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

# ---------------- learn_deform ----------------

def plot_loss(loss, expt_idx):
    train_loss = loss[0]
    val_loss = loss[1]
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.ylim([0.0, 0.050])
    plt.xlabel("epochs")
    plt.title("expt #{}".format(expt_idx))
    plt.legend()
    return plt
