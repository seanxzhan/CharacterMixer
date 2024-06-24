import os
import json
import trimesh
import subprocess
import numpy as np
from anytree import AnyNode
from utils import binvox_rw, misc
import scipy.ndimage as ndimage


# <-------------   transformations   ------------->
def voxel_space_to_mesh_space(transform, points_in_voxel_grid):
    dim = transform['dim']
    translate = transform['translate']
    scale = transform['scale']

    points_t = np.transpose(points_in_voxel_grid)
    voxel_grid_x = points_t[0]
    voxel_grid_y = points_t[1]
    voxel_grid_z = points_t[2]

    to_center = transform['to_center']
    voxel_grid_x -= to_center[0]
    voxel_grid_y -= to_center[1]
    voxel_grid_z -= to_center[2]

    original_x = (voxel_grid_x - 0.5) * (scale / dim[0]) + translate[0]
    original_y = (voxel_grid_y - 0.5) * (scale / dim[1]) + translate[1]
    original_z = (voxel_grid_z - 0.5) * (scale / dim[2]) + translate[2]

    original_points = np.stack((original_x, original_y, original_z), axis=0)
    return np.squeeze(np.transpose(original_points))


def mesh_space_to_voxel_space(transform, points_in_mesh_space, centering):
    dim = transform['dim']
    translate = transform['translate']
    scale = transform['scale']
    xs = points_in_mesh_space[:, 0:1]
    ys = points_in_mesh_space[:, 1:2]
    zs = points_in_mesh_space[:, 2:3]

    i = (xs - translate[0]) * (dim[0] / scale) + 0.5
    j = (ys - translate[1]) * (dim[1] / scale) + 0.5
    k = (zs - translate[2]) * (dim[2] / scale) + 0.5

    if centering:
        to_center = transform['to_center']
        i += to_center[0]
        j += to_center[1]
        k += to_center[2]

    new_points = np.concatenate((i, j, k), axis=1)
    return new_points


def make_grid_around_mesh(res, limits=[(-0.65, 0.65), (-0.15, 1.15), (-0.65, 0.65)]):
    """Makes grid points around the mesh.
    These grid points are in the mesh's world space.

    Args:
        res (int): grid resolution
        limits (list, optional): x, y, z limits.
            Defaults to [(-0.5, 0.5), (0.0, 1.0), (-0.5, 0.5)].

    Returns:
        tuple: x, y, z coordinate values
    """
    x_min = limits[0][0]
    x_max = limits[0][1]
    y_min = limits[1][0]
    y_max = limits[1][1]
    z_min = limits[2][0]
    z_max = limits[2][1]
    w = x_max - x_min
    # assert y_max - y_min == w and z_max - z_min == w
    X = np.linspace(x_min, x_max, res+1)
    Y = np.linspace(y_min, y_max, res+1)
    Z = np.linspace(z_min, z_max, res+1)
    xx, yy, zz = np.meshgrid(X, Y, Z)
    xx = xx[:-1, :-1, :-1]
    yy = yy[:-1, :-1, :-1]
    zz = zz[:-1, :-1, :-1]
    unit = w / res
    adj = unit / 2
    xx += adj
    yy += adj
    zz += adj
    # query_points = np.concatenate([xxx, yyy, zzz], axis=-1)
    return xx, yy, zz


def get_unnorm_to_center_t(model, model_unnorm):
    verts = np.argwhere(model > 0).astype(np.float32, copy=False)
    verts_unnorm = np.argwhere(model_unnorm > 0).astype(np.float32, copy=False)

    o_mid_x, o_mid_y, o_mid_z = get_mids(verts)
    mid_x, mid_y, mid_z = get_mids(verts_unnorm)

    return [o_mid_x-mid_x, o_mid_y-mid_y, o_mid_z-mid_z]


def get_mids(verts):
    min_x = np.min(verts[:, 0])
    max_x = np.max(verts[:, 0])
    min_y = np.min(verts[:, 1])
    max_y = np.max(verts[:, 1])
    min_z = np.min(verts[:, 2])
    max_z = np.max(verts[:, 2])
    mid_x = (max_x + min_x) / 2
    mid_y = (max_y + min_y) / 2
    mid_z = (max_z + min_z) / 2

    return mid_x, mid_y, mid_z


def get_transform_from_binvox(path_c, path_nc):
    # gives voxel to mesh transform
    # print("[VOX] reading binvox transform ...")
    with open(path_c, 'r', encoding="cp437") as f:
        # Read first 4 lines of file containing transformation info
        header = [next(f).strip().split() for x in range(4)]

    assert header[1][0] == 'dim'
    assert header[2][0] == 'translate'
    assert header[3][0] == 'scale'
    assert len(header[1]) == 4
    assert len(header[2]) == 4
    assert len(header[3]) == 2

    try:
        dim = [int(header[1][i]) for i in range(1, 4)]
        translate = [float(header[2][i]) for i in range(1, 4)]
        scale = float(header[3][1])
    except ValueError:
        print(
            "Unexpected val type when parsing binvox transformation info")
        exit(-1)

    with open(path_c, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        model = model.data.astype(int)
    with open(path_nc, 'rb') as f:
        model_unnorm = binvox_rw.read_as_3d_array(f)
        model_unnorm = model_unnorm.data.astype(int)
    to_center = get_unnorm_to_center_t(model, model_unnorm)
    transform = {
        'dim': dim,
        'translate': translate,
        'scale': scale,
        'to_center': to_center
    }

    return transform


def transform_points(points, transform, ones_or_zeros='ones'):
    assert ones_or_zeros in ['ones', 'zeros']
    num_points = points.shape[0]
    if ones_or_zeros == 'ones':
        padding = np.ones((num_points, 1), dtype=np.float32)
    else:
        padding = np.zeros((num_points, 1), dtype=np.float32)
    h_points = np.transpose(np.concatenate((points, padding), axis=-1))
    xformed_points = np.transpose(np.matmul(transform, h_points))[:, :3]
    return xformed_points


def transform_one_point(point, transform, ones_or_zeros='ones'):
    """
    in shape: (3, )
    out shape: (3, )
    """
    return transform_points(point[None, :], transform, ones_or_zeros)[0]


def get_one_world_to_local(local_frame):
    """
    in: (5, 3)
    out: (4, 4)
    """
    u = local_frame[0]
    xyz = local_frame[2:5]
    rot = np.zeros((4, 4), dtype=np.float32)
    rot[:3, :3] = xyz
    rot[3, 3] = 1.0
    tran = np.eye(4, dtype=np.float32)
    tran[:3, 3] = -u
    return np.matmul(rot, tran)


def get_world_to_local(local_coords):
    """
    local_coords: n_parts, 5, 3

    out: n_parts, 4, 4

    Computes transformation matrices for global coors to local coors.
    Global frame is centered at the origin
    Input to transformation should be a homogenous column vector
    """
    local_transforms = np.zeros(
        [local_coords.shape[0], 4, 4], dtype=np.float32)
    for i in range(local_coords.shape[0]):
        local_frame = local_coords[i]
        local_transforms[i] = get_one_world_to_local(local_frame)
    local_transforms = local_transforms.astype(np.float32)
    return local_transforms


def get_local_verts(verts, joints_to_faces, world_to_local):
    """
    out: List of num_parts (?, 3) arrays
    """
    all_local_verts = []
    for i in range(len(world_to_local)):
        mesh = trimesh.Trimesh(verts, joints_to_faces[i])
        part_verts = mesh.vertices
        local_verts = transform_points(part_verts, world_to_local[i])
        all_local_verts.append(local_verts)
    return all_local_verts


def gather_heads_tails(node: AnyNode, heads_tails, orig_or_xlated: int):
    if node == None:
        return
    if orig_or_xlated == 0:
        heads_tails[int(node.id)] = node.frame[:2, :]
    else:
        heads_tails[int(node.id)] = node.frame_xlated[:2, :]
    for child in node.children:
        gather_heads_tails(child, heads_tails, orig_or_xlated)
    return heads_tails


def gather_frames(node: AnyNode, frames, orig_or_xlated: int):
    if node == None:
        return
    if orig_or_xlated == 0:
        frames[int(node.id)] = node.frame
    else:
        frames[int(node.id)] = node.frame_xlated
    for child in node.children:
        gather_frames(child, frames, orig_or_xlated)
    return frames


# <-------------  coordinate system  ------------->
def normalize(a):
    """Normalizaes vector a.
    """
    if np.linalg.norm(a) == 0:
        # print("ERROR: DIVIDE BY ZERO")
        # exit(0)
        return 0 * a
    return a / np.linalg.norm(a)


def make_symmetrical(local_coords):
    """Makes local coordinate systems symmetrical over yz plane

    Args:
        local_coords (np.ndarray): num_joints x 5 x 3
    """
    for i in range(local_coords.shape[0]):
        # local_coords: 5x3 matrix, along columns:
        # local_orig, far_joint, x, y, z vecs
        local_orig = local_coords[i][0]
        # if joint lies in positive x axis
        if local_orig[0] > 0:
            # flip y axis positive direction
            local_coords[i][3] = -local_coords[i][3]
    local_coords = local_coords.astype(np.float32)
    return local_coords


def get_heads_tails_pos(path_to_skel, true_joint_ids):
    """
    out: (n_parts, 2, 3)
    """
    # output heads_tails follow the same order as true joint ids from rig.txt
    # length of heads_tails is the same as the number of joints
    f = open(path_to_skel)
    data: dict = json.load(f)
    heads_tails = np.zeros((len(true_joint_ids), 2, 3), dtype=np.float32)
    for i, id in enumerate(true_joint_ids):
        ht = np.array([data[id]["head"], data[id]["tail"]], dtype=np.float32)
        ht = np.reshape(ht, (2, 3))
        heads_tails[i] = ht
    return heads_tails


def get_joints_local_coords(num_joints, model_heads_tails,
                            ref_axis=normalize([1, 0.01, 0.01]),
                            # ref_axis=normalize([0.01, 0.01, 1]),
                            symmetrical=False,
                            check_dot=False):
    return get_joints_local_coords_blender(num_joints, model_heads_tails)
    """
    model_heads_tails: (n_parts, 2, 3)

    out: (n_parts, 5, 3)

    check_dot=True for 1322, 1323
    check_dot=True and -y axis as ref axis for 3427, 3430
    """
    local_coords = np.zeros((num_joints, 5, 3))

    for i in range(num_joints):
        local_orig = model_heads_tails[i][0]
        tail = model_heads_tails[i][1]
        x_axis_vec = normalize(tail - local_orig)
        if check_dot:
            if np.dot(x_axis_vec, np.array([0, 1, 0])) > 0:
                y_axis_vec = normalize(np.cross(x_axis_vec, ref_axis))
            else:
                y_axis_vec = normalize(np.cross(x_axis_vec, -ref_axis))
                # y_axis_vec = normalize(np.cross(x_axis_vec, -normalize([0.01, 1.0, 0.01])))
        else:
            y_axis_vec = normalize(np.cross(x_axis_vec, ref_axis))
        z_axis_vec = normalize(np.cross(x_axis_vec, y_axis_vec))
        loc = np.array([local_orig, tail, y_axis_vec, x_axis_vec, z_axis_vec],
                       dtype=np.float32)
        local_coords[i] = loc

    return make_symmetrical(local_coords) if symmetrical else local_coords


def get_joints_local_coords_blender(num_joints, model_heads_tails):
    """
    model_heads_tails: (n_parts, 2, 3)

    out: (n_parts, 5, 3)

    check_dot=True for 1322, 1323
    check_dot=True and -y axis as ref axis for 3427, 3430
    """
    local_coords = np.zeros((num_joints, 5, 3), dtype=np.float32)

    SAFE_THRESHOLD = 6.1e-3
    CRITICAL_THRESHOLD = 2.5e-4
    THRESHOLD_SQUARED = CRITICAL_THRESHOLD * CRITICAL_THRESHOLD

    for i in range(num_joints):
        local_orig = model_heads_tails[i][0]
        tail = model_heads_tails[i][1]
        bone_vec = normalize(tail - local_orig)

        x = bone_vec[0]
        y = bone_vec[1]
        z = bone_vec[2]

        theta = 1 + bone_vec[1]
        theta_alt = x * x + z * z
        
        M = np.zeros((3, 3), dtype=np.float32)

        if theta > SAFE_THRESHOLD or theta_alt > THRESHOLD_SQUARED:
            M[0][1] = -x
            M[1][0] = x
            M[1][1] = y
            M[1][2] = z
            M[2][1] = -z

            if theta <= SAFE_THRESHOLD:
                theta = 0.5 * theta_alt + 0.125 * theta_alt * theta_alt
            
            M[0][0] = 1 - x * x / theta
            M[2][2] = 1 - z * z / theta
            M[0][2] = - x * z / theta
            M[2][0] = - x * z / theta
        else:
            M[0][0] = -1
            M[1][1] = -1
            M[2][2] = 1

        local_coords[i][0] = local_orig
        local_coords[i][1] = tail
        local_coords[i][2:] = M

    return local_coords


# <-------------    sdf     ------------->
def setup_vox(obj_dir):
    subprocess.call(
        ["cp data_prep/binvox "+obj_dir+"/binvox"], shell=True)
    subprocess.call(
        ["cp data_prep/voxelize.sh "+obj_dir+"/voxelize.sh"], shell=True)


def teardown_vox(obj_dir):
    subprocess.call(["rm {}/voxelize.sh".format(obj_dir)], shell=True)
    subprocess.call(["rm {}/binvox".format(obj_dir)], shell=True)
    subprocess.call(["killall Xvfb"], shell=True)


def voxelize_obj(obj_dir, obj_filename, res,
                 out_vox_c_path, out_vox_nc_path,
                 bbox="", min_max=["", "", "", "", "", ""]):
    assert os.path.exists(os.path.join(obj_dir, 'binvox'))
    assert os.path.exists(os.path.join(obj_dir, 'voxelize.sh'))
    files = misc.sorted_alphanumeric(os.listdir(obj_dir))
    files.remove('voxelize.sh')
    files.remove('binvox')
    obj_id = os.path.splitext(obj_filename)[0]
    devnull = open(os.devnull, 'w')
    if bbox == "":
        call_string_c = "cd {} && bash voxelize.sh {} {} c".format(
            obj_dir, obj_filename, res)
    else:
        call_string_c =\
            "cd {} && bash voxelize.sh {} {} c {} {} {} {} {} {} {}".format(
                obj_dir, obj_filename, res, bbox,
                min_max[0], min_max[1], min_max[2],
                min_max[3], min_max[4], min_max[5])
    mv_c = "mv {}/{}.binvox {}".format(
        obj_dir, obj_id, out_vox_c_path)
    subprocess.call([call_string_c], shell=True,
        stdout=devnull, stderr=devnull
    )
    subprocess.call([mv_c], shell=True)
    if bbox == "":
        call_string_nc = "cd {} && bash voxelize.sh {} {} nc".format(
            obj_dir, obj_filename, res)
    else:
        call_string_nc =\
            "cd {} && bash voxelize.sh {} {} nc {} {} {} {} {} {} {}".format(
                obj_dir, obj_filename, res, bbox,
                min_max[0], min_max[1], min_max[2],
                min_max[3], min_max[4], min_max[5])
    mv_nc = "mv {}/{}.binvox {}".format(
        obj_dir, obj_id, out_vox_nc_path)
    subprocess.call([call_string_nc], shell=True,
        stdout=devnull, stderr=devnull
    )
    subprocess.call([mv_nc], shell=True)


def bin2sdf(vox_path='', input=None):
    """Converts voxels into SDF grid field.
       Negative for inside, positive for outside. Range: (-1, 1)

    Args:
        input (np.ndarray): voxels of shape (dim, dim, dim)

    Returns:
        np.ndarray: SDF representation of shape (dim, dim, dim)
    """
    assert (vox_path != '') != (input is not None),\
        "must supply either [vox_path] or [input]"

    if vox_path != '':
        with open(vox_path, 'rb') as f:
            voxel_model_dense = binvox_rw.read_as_3d_array(f)
            input = voxel_model_dense.data.astype(int)

    fill_map = np.zeros(input.shape, dtype=np.bool)
    output = np.zeros(input.shape, dtype=np.float16)
    # fill inside
    changing_map = input.copy()
    sdf_in = -1
    while np.sum(fill_map) != np.sum(input):
        changing_map_new = ndimage.binary_erosion(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        output[np.where(changing_map_new!=changing_map)] = sdf_in
        changing_map = changing_map_new.copy()
        sdf_in -= 1
    # fill outside.
    # No need to fill all of them, since during training,
    # outside part will be masked.
    changing_map = input.copy()
    sdf_out = 1
    while np.sum(fill_map) != np.size(input):
        changing_map_new = ndimage.binary_dilation(changing_map)
        fill_map[np.where(changing_map_new!=changing_map)] = True
        output[np.where(changing_map_new!=changing_map)] = sdf_out
        changing_map = changing_map_new.copy()
        sdf_out += 1
    # Normalization
    output[np.where(output < 0)] /= (-sdf_in-1)
    output[np.where(output > 0)] /= (sdf_out-1)

    output = output.astype(np.float32)
    return output


def lerp(x, y, t):
    return x + (y - x) * t

# <-------------    alignment     ------------->
def align_voxels(voxels, translation):
    new_voxels = np.zeros_like(voxels)
    dim = voxels.shape[0]
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if i + translation[0] < 0 or i + translation[0] >= dim or\
                    j + translation[1] < 0 or j + translation[1] >= dim or\
                        k + translation[2] < 0 or k + translation[2] >= dim:
                            continue
                new_voxels[
                    i + translation[0],
                    j + translation[1],
                    k + translation[2]] = voxels[i, j, k]
    return new_voxels
