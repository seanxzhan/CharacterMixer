import random
import numpy as np
from utils import kit
from anytree import AnyNode
from scipy.spatial.transform import Rotation as R


def make_random_pose_info(num_joints, delta_angle=30):
    dict = {}
    for i in range(num_joints):
        rots = {}
        rots["x_rot"] = 0
        rots["y_rot"] = random.uniform(-delta_angle, delta_angle)
        rots["z_rot"] = random.uniform(-delta_angle, delta_angle)
        dict[i] = rots
    return dict


def make_rest_pose_info(num_nodes):
    dict = {}
    for i in range(num_nodes):
        rots = {}
        rots["x_rot"] = 0
        rots["y_rot"] = 0
        rots["z_rot"] = 0
        dict[i] = rots
    return dict


def get_rot_mats(rot_info):
    num_joints = len(rot_info)
    rot_mats = np.zeros((num_joints, 4, 4), dtype=np.float32)
    for k, v in rot_info.items():
        joint_idx = int(k)
        rot_degrees = [[v["x_rot"], v["y_rot"], v["z_rot"]]]
        r = R.from_euler('xyz', rot_degrees, degrees=True)
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = r.as_matrix()
        rot_mats[joint_idx] = mat
    return rot_mats


def get_one_rot_mat(x, y, z):
    """
    x, y, z: joint angles, euler XYZ
    """
    rot_degrees = [[x, y, z]]
    r = R.from_euler('xyz', rot_degrees, degrees=True)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = r.as_matrix()
    return mat


def pose_bones(node: AnyNode, rot_mats):
    """Poses bones with forward kinematics

    Args:
        node (AnyNode): root node representing bone
        rot_mats (np.ndarray): rotation matrices for each bone
    """
    if node == None:
        return

    bone_idx = int(node.id)
    rot = rot_mats[bone_idx]
    local_coord = node.frame
    if np.array_equal(local_coord[2:, :], np.zeros((3, 3), dtype=np.float32)):
        local_coord[2:, :] = np.eye(3, 3, dtype=np.float32)

    if node.parent is None:
        parent_xform = np.eye(4, dtype=np.float32)
    else:
        parent_xform = node.parent.xform

    xformed_frame = transform_local_coord(local_coord, parent_xform)
    world_to_local = kit.get_one_world_to_local(xformed_frame)
    new_local_coord = rotate_local_coord(
        xformed_frame, world_to_local, rot)
    xform = get_tmat_from_rot(new_local_coord, local_coord)
    node.frame = new_local_coord
    node.xform = xform

    for child in node.children:
        pose_bones(child, rot_mats)


def get_local_coords_from_tree(nodes):
    local_coords = np.zeros((len(nodes), 5, 3), dtype=np.float32)
    for n in nodes:
        local_coords[int(n.id)] = n.frame
    return local_coords


def rotate_local_coord(local_coord, world_to_local, rot):
    new_local_coord = np.zeros((5, 3), dtype=np.float32)
    ht = local_coord[:2]
    ht_in_local = kit.transform_points(ht, world_to_local)
    rot_ht_in_l = kit.transform_points(ht_in_local, rot)
    rot_ht_in_w = kit.transform_points(
        rot_ht_in_l, np.linalg.inv(world_to_local))
    new_local_coord[:2] = rot_ht_in_w

    xyz_vecs = local_coord[2:]
    vecs_in_local = kit.transform_points(xyz_vecs, world_to_local, 'zeros')
    rot_vecs_in_l = kit.transform_points(vecs_in_local, rot, 'zeros')
    rot_vecs_in_w = kit.transform_points(
        rot_vecs_in_l, np.linalg.inv(world_to_local), 'zeros')
    new_local_coord[2:] = rot_vecs_in_w

    return new_local_coord


def get_tmat_from_rot(rotated_local_coord, local_coord):
    new_world_to_local = kit.get_one_world_to_local(rotated_local_coord)
    old_world_to_local = kit.get_one_world_to_local(local_coord)
    old_w_to_new_w = np.linalg.inv(new_world_to_local) @ old_world_to_local
    return old_w_to_new_w


def transform_local_coord(orig_local_coord, xform):
    local_coord = np.zeros((5, 3), dtype=np.float32)
    ht = orig_local_coord[:2]
    new_ht = kit.transform_points(ht, xform)
    local_coord[:2] = new_ht

    vecs = orig_local_coord[2:]
    new_vecs = kit.transform_points(vecs, xform, 'zeros')
    local_coord[2:] = new_vecs

    return local_coord


def pose_character(rest_vertices, skinning_weights,
                   rest_world_to_local, new_world_to_local):
    new_w_to_l = np.rollaxis(new_world_to_local, 0, new_world_to_local.ndim)
    new_l_to_w = np.linalg.inv(new_w_to_l.T).T
    rest_w_to_l = np.rollaxis(rest_world_to_local, 0, rest_world_to_local.ndim)
    old_to_new = np.einsum("ijk, jlk -> ilk", new_l_to_w, rest_w_to_l)
    new_verts = np.zeros_like(rest_vertices)
    for i in range(len(rest_vertices)):
        row_weights = skinning_weights[i]
        nonzero_indices = np.nonzero(row_weights)[0]
        weights = row_weights[nonzero_indices][None, :] # (1, n)
        xform_mats = old_to_new[:, :, nonzero_indices]
        vert = rest_vertices[i]
        vert_hom = np.ones((4, 1), dtype=np.float32)
        vert_hom[:3, 0] = vert
        vert_hom = vert_hom[:, None]
        vert_hom_dup = np.repeat(vert_hom, len(weights), axis=-1)
        xformed_verts = np.einsum("ijk, jlk -> ilk", xform_mats, vert_hom_dup)
        xformed_verts = xformed_verts[:3, 0, :]    # (3, n)
        posed = weights * xformed_verts
        posed = np.sum(posed, axis=-1)
        new_verts[i] = posed
    return new_verts

