import os
import sys
import re
import pymeshlab
from multiprocessing import Queue, Process
from anytree import AnyNode
import queue
from itertools import groupby, chain
import numpy as np
import trimesh
import pymeshfix
from tqdm import tqdm
from trimesh.exchange.export import export_mesh
from utils import kit, misc, visualize


def make_v_f_dicts(path_to_obj):
    key_func = lambda x: x[:2]
    with open(path_to_obj, 'r') as file:
        lines = file.readlines()
    # group by the first two characters
    keys = [key for key, _ in groupby(lines, key=key_func)]
    grouped = [list(group) for _, group in groupby(lines, key=key_func)]
    v_indices = [i for i, x in enumerate(keys) if x == "v "]
    vn_indices = [i for i, x in enumerate(keys) if x == "vn"]
    f_indices = [i for i, x in enumerate(keys) if x == "f "]
    vertices = [grouped[i] for i in v_indices]
    vertices = list(chain(*vertices))
    normals = [grouped[i] for i in vn_indices]
    normals = list(chain(*normals))
    faces = [grouped[i] for i in f_indices]
    faces = list(chain(*faces))

    make_float = lambda ele: float(ele)
    get_split_values2 = lambda str: re.split(r'\s', str[2:][:-1])
    get_split_values3 = lambda str: re.split(r'\s', str[3:][:-1])
    # get vertex indices from face triangle triplets
    get_vert_indices = lambda str: re.split(r'//|\s',str[2:][:-1])

    vertices_lst = []
    for value in vertices:
        vertices_lst.append(list(map(make_float, get_split_values2(value))))
    
    vertex_normals_lst = []
    for value in normals:
        vertex_normals_lst.append(
            list(map(make_float, get_split_values3(value))))

    faces_lst = []
    for value in faces:
        faces_lst.append(
            list(map(lambda ele: int(ele) - 1,
                     get_vert_indices(value)[::2])))

    # face indices mapped to vertex and vertex normals indices
    faces_to_vertices = {}
    for count, value in enumerate(faces):
        faces_to_vertices[count] =\
            list(map(lambda ele: int(ele) - 1,
                     get_vert_indices(value)[::2]))

    # vertex indices mapped to face indices
    vertices_to_faces = {}
    for i in range(len(vertices)):
        list_of_keys = [k for k, v in faces_to_vertices.items() if i in v]
        vertices_to_faces[i] = list_of_keys

    return vertices_lst, vertex_normals_lst, faces_lst, vertices_to_faces


def make_b_v_f_dicts(path_to_rig_info):
    # for each joint, get all vertices with skinning weight over alpha
    with open(path_to_rig_info, 'r') as file:
        lines = file.readlines()

    key_func = lambda x: x.split(" ")[0]
    keys = [key for key, _ in groupby(lines, key=key_func)]
    grouped = [list(group) for _, group in groupby(lines, key=key_func)]

    joints_index = keys.index('joints')
    skin_index = keys.index('skin')
    hier_index = keys.index('hier')
    joints = grouped[joints_index]
    skin = grouped[skin_index]
    hier = grouped[hier_index]

    joints_idx_to_id = {}
    joints_pos = []
    joint_ids = []
    get_joint_id = lambda str: re.split(r'\s|\n', str)[1]
    get_joint_pos = lambda str: list(map(lambda x: float(x),
                                         re.split(r'\s|\n', str)[2:5]))
    for count, value in enumerate(joints):
        joint_id = get_joint_id(value)
        joint_ids.append(joint_id)
        joints_idx_to_id[count] = joint_id
        joints_pos.append(get_joint_pos(value))

    joints_id_to_idx = {}
    for i in range(len(joints)):
        joints_id_to_idx[joints_idx_to_id[i]] = i

    bones = {}
    bones_pos = []
    for count, value in enumerate(hier):
        joint_1 = re.split(r'\s|\n', value)[1]
        joint_2 = re.split(r'\s|\n', value)[2]
        idx_1 = joints_id_to_idx[joint_1]
        idx_2 = joints_id_to_idx[joint_2]
        bones[count] = [idx_1, idx_2]
        bones_pos.append([joints_pos[idx_1], joints_pos[idx_2]])

    vertex_to_joint = {}
    vertex_to_joint_idx = []
    skinning_weights = np.zeros((len(skin), len(joints_pos)), np.float32)
    for i in range(len(skin)):
        tokens = list(filter(lambda x: x != '', re.split(r'\s|\n', skin[i])))
        it = iter(tokens)

        # NOTE: this block of code is not being used
        # vertex_to_joint[int(tokens[1])] is joint_id with highest weight
        ids_and_weights = list(zip(it, it))[1:]
        highest_w = -1.0
        ids_w_highest_w = []
        for tup in ids_and_weights:
            if float(tup[1]) >= highest_w:
                highest_w = float(tup[1])
                ids_w_highest_w.append(tup[0])
        vertex_to_joint[int(tokens[1])] = ids_w_highest_w

        highest_w = -1.0
        idx_of_highest_w = -1
        for tup in ids_and_weights:
            # skinning_weights[i] = 
            joint_idx = joints_id_to_idx[tup[0]]
            skinning_weights[i, joint_idx] = float(tup[1])
            if float(tup[1]) > highest_w:
                highest_w = float(tup[1])
                idx_of_highest_w = joints_id_to_idx[tup[0]]
            if float(tup[1]) == highest_w:
                # flip = random.randint(0, 1)
                flip = 0
                if flip == 0:
                    highest_w = float(tup[1])
                    idx_of_highest_w = joints_id_to_idx[tup[0]]
        vertex_to_joint_idx.append(idx_of_highest_w)

    joint_to_vertex = {}
    for i in range(len(joints)):
        joint_id = joints_idx_to_id[i]
        list_of_keys = [k for k, v in vertex_to_joint.items() if joint_id in v]
        joint_to_vertex[i] = list_of_keys

    # get bones to vertices
    bones_to_vertices = {}
    for i in range(len(hier)):
        joints_indices = bones[i]
        vert_list = map(lambda x: joint_to_vertex[x], joints_indices)
        bones_to_vertices[i] = list(chain(*vert_list))

    # TODO: all this stuff should be saved somewhere first
    return bones_to_vertices, vertex_to_joint_idx, len(joints),\
        joint_ids, joints_pos, bones_pos, skinning_weights


def get_skel_tree(num_joints, path_to_rig, local_coords=None):
    """Given model index, return root encoding skeleton tree.

    Args:
        model_index (int): model index

    Returns:
        anytree.AnyNode: root of a skeleton's tree representation
    """
    # node ids will be joints' indices
    # we can do this because all data is sequentially ordered
    joint_ids = []
    root_joint_id = ''
    joint_nodes = []
    joints_to_bones = {}
    bones_to_joints = {}
    
    # edges are in order of bone indices
    edges = []

    for i in range(num_joints):
        joints_to_bones[i] = []

    # assuming all joints are used in bones
    fp = path_to_rig
    bone_idx = 0
    with open(fp, 'r') as f:

        while True:
            line = f.readline()
            if not line:
                break

            tokens = line.split()
            if tokens[0] == 'joints':
                joint_ids.append(tokens[1])

            if tokens[0] == 'root':
                root_joint_id = tokens[1]
                # AnyNode's ids are joint indices
                # note by the time we read root, we would have already
                # read all the joints
                for i in range(len(joint_ids)):
                    if local_coords is None:
                        joint_nodes.append(AnyNode(id=str(i)))
                    else:
                        joint_nodes.append(
                            AnyNode(
                                id=str(i),
                                frame=local_coords[i],
                                xform=np.eye(4, dtype=np.float32)))

            if tokens[0] == 'hier':
                id_src_idx = joint_ids.index(tokens[1])
                id_dest_idx = joint_ids.index(tokens[2])

                bones_to_joints[bone_idx] = [id_src_idx, id_dest_idx]

                src_list = joints_to_bones[id_src_idx]
                src_list.append(bone_idx)
                joints_to_bones[id_src_idx] = src_list

                dest_list = joints_to_bones[id_dest_idx]
                dest_list.append(bone_idx)
                joints_to_bones[id_dest_idx] = dest_list

                joint_nodes[id_dest_idx].parent = joint_nodes[id_src_idx]
                edges.append((id_src_idx, id_dest_idx))

                bone_idx += 1

    root_joint_idx = joint_ids.index(root_joint_id)
    return root_joint_idx, joint_nodes, edges, joints_to_bones, bones_to_joints


def get_joint_faces(j_to_v: dict, v_to_f: dict, faces: list):
    all_joint_faces = []

    for _, vert_indices in j_to_v.items():
        get_face_idx = lambda v_i: v_to_f[v_i]
        face_indices = list(map(get_face_idx, vert_indices))
        face_indices = list(np.unique(list(chain(*face_indices))))

        get_faces = lambda f_i: faces[f_i]
        joint_faces = list(map(get_faces, face_indices))
        joint_faces = np.asarray(joint_faces, dtype=np.int64)
        all_joint_faces.append(joint_faces)
    return all_joint_faces


def save_seg_patches(path_to_obj, path_to_rig_info,
                     img_out_path='', bones=None,
                     highlight_part_idx=-1, rot=45):
    vertices, _, faces, v_to_f = make_v_f_dicts(path_to_obj)
    _, verts_to_joints, num_joints, _, _, _, _ = make_b_v_f_dicts(
        path_to_rig_info)
    vertices = np.array(vertices)

    joints_to_verts = {}
    for j in range(num_joints):
        joints_to_verts[j] = []

    for j in range(len(verts_to_joints)):
        joints_to_verts[verts_to_joints[j]].append(j)

    joints_to_faces = get_joint_faces(joints_to_verts, v_to_f, faces)

    print("[SEGMENT] Saving segmented patches visualization")
    plt, _ = visualize.show_verts_to_faces_and_bones(
        vertices, joints_to_faces, num_joints, bones,
        highlight_part_idx=highlight_part_idx, rot=rot)
    misc.save_fig(plt, '', img_out_path, rotate=True)
    plt.close()


def get_model_parts(path_to_obj, path_to_rig_info, path_to_skel,
                    parts_obj_out_dir='',
                    out_vox_c_dir='',
                    out_vox_nc_dir='',
                    res=-1,
                    img_out_dir='',
                    just_rig_info=False,
                    vis_part_voxels=False):
    print("[SEGMENT] Getting model parts") if not just_rig_info else None
    vertices, _, faces, v_to_f = make_v_f_dicts(path_to_obj)
    _, verts_to_joints, num_joints, joint_ids, joints_pos, bones_pos, skin_w =\
        make_b_v_f_dicts(path_to_rig_info)
    joints_pos = np.array(joints_pos, dtype=np.float32)
    bones_pos = np.array(bones_pos, dtype=np.float32)
    heads_tails = kit.get_heads_tails_pos(path_to_skel, joint_ids)

    local_coords = kit.get_joints_local_coords(num_joints, heads_tails)
    world_to_local = kit.get_world_to_local(local_coords)

    # these vertices are in original mesh space
    vertices = np.asarray(vertices)

    joints_to_verts = {}
    for j in range(num_joints):
        joints_to_verts[j] = []

    for j in range(len(verts_to_joints)):
        joints_to_verts[verts_to_joints[j]].append(j)

    joints_to_faces = get_joint_faces(joints_to_verts, v_to_f, faces)

    if just_rig_info:
        package = {}
        package["joints_pos"] = joints_pos
        package["bones_pos"] = bones_pos
        package["heads_tails"] = heads_tails
        package["local_coords"] = local_coords
        package["world_to_local"] = world_to_local
        package["vertices"] = vertices
        package["part_idx"] = verts_to_joints
        package["faces"] = faces
        package["joints_to_verts"] = joints_to_verts
        package["joints_to_faces"] = joints_to_faces
        package["skinning_weights"] = skin_w
        return package

    num_joints = len(joints_to_faces)

    num_workers = 10
    list_of_lists = misc.chunks(list(range(num_joints)), num_workers)
    
    q = Queue()
    workers = [
        Process(target=voxelize_one_part,
                args=(q, lst, res, world_to_local, vertices, joints_to_faces,
                      parts_obj_out_dir, img_out_dir,
                      out_vox_c_dir, out_vox_nc_dir, vis_part_voxels))
        for lst in list_of_lists]

    kit.setup_vox(parts_obj_out_dir)

    for p in workers:
        p.start()
    pbar = tqdm(total=num_joints)
    while True:
        flag = True
        try:
            _ = q.get(True, 1.0)
        except queue.Empty:
            flag = False
        if flag:
            pbar.update(1)
        all_exited = True
        for p in workers:
            if p.exitcode is None:
                all_exited = False
                break
        if all_exited and q.empty():
            break
    pbar.close()
    for p in workers:
        p.join()
    
    kit.teardown_vox(parts_obj_out_dir)

    return joints_pos, bones_pos, heads_tails, local_coords, world_to_local


def multi_thread_vox_jumper(num_joints, res, world_to_local, vertices,
                            joints_to_faces, parts_obj_out_dir, img_out_dir,
                            out_vox_c_dir, out_vox_nc_dir, n_workers=10):
    num_workers = n_workers
    list_of_lists = misc.chunks(list(range(num_joints)), num_workers)

    q = Queue()
    workers = [
        Process(target=voxelize_one_part,
                args=(q, lst, res, world_to_local, vertices, joints_to_faces,
                      parts_obj_out_dir, img_out_dir,
                      out_vox_c_dir, out_vox_nc_dir)) for lst in list_of_lists]

    kit.setup_vox(parts_obj_out_dir)

    for p in workers:
        p.start()
    pbar = tqdm(total=num_joints)
    while True:
        flag = True
        try:
            _ = q.get(True, 1.0)
        except queue.Empty:
            flag = False
        if flag:
            pbar.update(1)
        all_exited = True
        for p in workers:
            if p.exitcode is None:
                all_exited = False
                break
        if all_exited and q.empty():
            break
    pbar.close()
    for p in workers:
        p.join()
    
    kit.teardown_vox(parts_obj_out_dir)


def voxelize_one_part(q, joint_indices, res, world_to_local,
                      vertices, joints_to_faces,
                      parts_obj_out_dir, img_out_dir,
                      out_vox_c_dir, out_vox_nc_dir,
                      vis_part_voxels=False):
    """Brings part meshes into local space then voxlizes them
    """
    for i in joint_indices:
        # print(c, '/', len(joint_indices))
        # turn off verbose output
        sys.stdout = open(os.devnull, 'w')
        
        mesh = trimesh.Trimesh(vertices, joints_to_faces[i])
        part_verts = mesh.vertices
        part_faces = mesh.faces

        local_vert = kit.transform_points(part_verts, world_to_local[i])
        local_mesh = trimesh.Trimesh(local_vert, part_faces)

        fix_mesh = 2

        if fix_mesh == 1:
            tin = pymeshfix.PyTMesh()
            tin.load_array(local_mesh.vertices, local_mesh.faces)
            tin.join_closest_components()
            tin.fill_small_boundaries()
            vclean, fclean = tin.return_arrays()
            fixed_mesh = trimesh.Trimesh(vclean, fclean)
            fixed_mesh.fix_normals()
        else:
            fixed_mesh = local_mesh

        mesh_filename = str(i) + '.obj'
        mesh_out_path = os.path.join(parts_obj_out_dir, mesh_filename)
        export_mesh(fixed_mesh, mesh_out_path, file_type='obj')

        if fix_mesh == 2:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(mesh_out_path)
            ms.meshing_repair_non_manifold_edges()
            ms.meshing_close_holes()
            ms.save_current_mesh(mesh_out_path)

        out_vox_c_path = os.path.join(out_vox_c_dir, str(i) + '.binvox')
        out_vox_nc_path = os.path.join(out_vox_nc_dir, str(i) + '.binvox')
        kit.voxelize_obj(
            parts_obj_out_dir, mesh_filename, res,
            out_vox_c_path, out_vox_nc_path)

        if vis_part_voxels:
            part_img_out_path = os.path.join(img_out_dir, str(i) + '.png')
            voxels = misc.load_voxels(out_vox_c_path)
            plt = visualize.vis_voxels(voxels, a=0.8)
            misc.save_fig(plt, '', part_img_out_path, rotate=False)

        sys.stdout = sys.__stdout__
        q.put(i)


def voxelize_part(id, mesh, vox_c_dir, vox_nc_dir, obj_dir, res):
    fix_mesh = 2
    if fix_mesh == 1:
        tin = pymeshfix.PyTMesh()
        tin.load_array(mesh.vertices, mesh.faces)
        tin.join_closest_components()
        tin.fill_small_boundaries()
        vclean, fclean = tin.return_arrays()
        fixed_mesh = trimesh.Trimesh(vclean, fclean)
        fixed_mesh.fix_normals()
    else:
        fixed_mesh = mesh

    mesh_filename = str(id) + '.obj'
    mesh_out_path = os.path.join(obj_dir, mesh_filename)
    export_mesh(fixed_mesh, mesh_out_path, file_type='obj')

    if fix_mesh == 2:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_out_path)
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_close_holes()
        ms.save_current_mesh(mesh_out_path)

    out_vox_c_path = os.path.join(vox_c_dir, str(id) + '.binvox')
    out_vox_nc_path = os.path.join(vox_nc_dir, str(id) + '.binvox')
    kit.setup_vox(obj_dir)
    kit.voxelize_obj(
        obj_dir, mesh_filename, res,
        out_vox_c_path, out_vox_nc_path)
    kit.teardown_vox(obj_dir)
