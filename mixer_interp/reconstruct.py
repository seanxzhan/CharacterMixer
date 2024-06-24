import mcubes
import trimesh
import numpy as np
from tqdm import tqdm
from utils import kit, bbox
from typing import List
from anytree import AnyNode
from scipy.interpolate import interpn
from utils.bbox import Bbox


def reconstruct_whole(mesh_out_path, sdf,
                      transform=True, transform_info={},
                      negate=True, level_set=0, show_msg=False):
    # marching_cubes thinks positive means inside, negative means outside
    if show_msg:
        print("[RECONSTRUCT] Reconstructing from the entire model's sdf grid")
    sdf_grid = -sdf if negate else sdf
    vertices, triangles = mcubes.marching_cubes(sdf_grid, level_set)
    if transform:
        assert transform_info != {}
        vertices = kit.voxel_space_to_mesh_space(transform_info, vertices)
    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fix_normals(mesh)
    trimesh.exchange.export.export_mesh(mesh, mesh_out_path, file_type='obj')
    return mesh


def trilinear_interp(points_in_grid, part_res,
                     part_points, sdf_field):
    """

    returns: (part_res ** 3, )
    """
    n_points = len(points_in_grid)
    val = np.zeros((n_points,), dtype=np.float32)
    l1 = np.argwhere(np.any(points_in_grid < 0.5, axis=1))
    l2 = np.argwhere(np.any(points_in_grid > part_res - 0.5, axis=1))
    out_of_bound_idx = np.union1d(l1, l2)
    val[out_of_bound_idx] = 1.0
    in_bound_idx = list(
        set(range(n_points)) - set(out_of_bound_idx.tolist()))
    if len(in_bound_idx) != 0:
        val[in_bound_idx] = interpn(
            part_points, sdf_field,
            points_in_grid[in_bound_idx])
    return val


def reconstruct_from_parts(mesh_out_path, all_sdfs,
                           all_binvox_xforms, world_to_local,
                           dim=128, negate=True, level_set=0):
    print("[RECONSTRUCT] Reconstructing from parts")
    num_parts = len(all_sdfs)
    assert len(all_binvox_xforms) == num_parts
    assert len(world_to_local) == num_parts

    part_res = all_sdfs[0].shape[0]
    x = np.linspace(0, part_res - 1, part_res) + 0.5
    y = np.linspace(0, part_res - 1, part_res) + 0.5
    z = np.linspace(0, part_res - 1, part_res) + 0.5
    part_points = (x, y, z)

    xx, yy, zz = kit.make_grid_around_mesh(dim)
    xx = np.reshape(xx, (dim**3, 1))
    yy = np.reshape(yy, (dim**3, 1))
    zz = np.reshape(zz, (dim**3, 1))
    points_in_world = np.concatenate([xx, yy, zz], axis=-1)

    all_vals = []
    pbar = tqdm(total=num_parts)
    for part_idx in range(num_parts):
        points_in_local_space = kit.transform_points(
            points_in_world, world_to_local[part_idx])
        points_in_vox_space = kit.mesh_space_to_voxel_space(
            all_binvox_xforms[part_idx], points_in_local_space, True)
        vals = trilinear_interp(
            points_in_vox_space, part_res, part_points, all_sdfs[part_idx])
        all_vals.append(vals[:, None])
        pbar.update(1)
    pbar.close()

    whole_sdf_grid = np.concatenate(all_vals, axis=-1)
    whole_sdf_grid = np.min(whole_sdf_grid, axis=-1)
    whole_sdf_grid = np.reshape(whole_sdf_grid, (dim, dim, dim))

    whole_sdf_grid = np.swapaxes(whole_sdf_grid, 0, 1)
    whole_sdf_grid = -whole_sdf_grid if negate else whole_sdf_grid
    vertices, triangles = mcubes.marching_cubes(whole_sdf_grid, level_set)
    # orig mesh, x: [-0.5, 0.5], y: [0, 1], z: [-0.5, 0.5]
    vertices[:, 0] = vertices[:, 0] / (dim - 1) - 0.5
    vertices[:, 1] = vertices[:, 1] / (dim - 1)
    vertices[:, 2] = vertices[:, 2] / (dim - 1) - 0.5
    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fix_normals(mesh)
    trimesh.exchange.export.export_mesh(mesh, mesh_out_path, file_type='obj')
    print("[RECONSTRUCT] Done.")
    return mesh


def reconstruct_from_parts_interp_tree(t, part_res,
                                       aug_skel_nodes: List[AnyNode],
                                       src_nodes: List[AnyNode],
                                       dest_nodes: List[AnyNode], 
                                       src_binvox_xforms, dest_binvox_xforms,
                                       src_all_sdfs, dest_all_sdfs,
                                       posed_aug_frames=None,
                                       posed_src_frames=None,
                                       posed_dest_frames=None,
                                       dim=128, negate=True, level_set=0):
    print("[RECONSTRUCT] Reconstructing from parts")
    x = np.linspace(0, part_res - 1, part_res) + 0.5
    y = np.linspace(0, part_res - 1, part_res) + 0.5
    z = np.linspace(0, part_res - 1, part_res) + 0.5
    part_points = (x, y, z)

    xx, yy, zz = kit.make_grid_around_mesh(dim)
    xx = np.reshape(xx, (dim**3, 1))
    yy = np.reshape(yy, (dim**3, 1))
    zz = np.reshape(zz, (dim**3, 1))
    points_in_world = np.concatenate([xx, yy, zz], axis=-1)

    n_parts = len(aug_skel_nodes)
    all_vals = []
    # all_interp_ht, all_interp_frames, all_interp_w_to_l should be POSED
    all_interp_ht = np.zeros((n_parts, 2, 3), dtype=np.float32)
    all_interp_frames = np.zeros((n_parts, 5, 3), dtype=np.float32)
    all_interp_w_to_l = np.zeros((n_parts, 4, 4), dtype=np.float32)
    pbar = tqdm(total=n_parts, leave=True)

    for ai, node in enumerate(aug_skel_nodes):
        if posed_aug_frames is None:
            # rest pose
            interp_virt_ht = kit.lerp(
                node.dest_virt_ht, node.src_virt_ht, t)
        else:
            # posed
            interp_virt_ht = posed_aug_frames[ai, :2]
        if posed_aug_frames is None:
            # rest pose
            interp_local_frame = kit.get_joints_local_coords(
                1, interp_virt_ht[None, :])[0]
        else:
            # posed
            interp_local_frame = posed_aug_frames[ai]
        interp_w_to_l = kit.get_one_world_to_local(interp_local_frame)

        all_interp_ht[ai] = interp_virt_ht
        all_interp_frames[ai] = interp_local_frame
        all_interp_w_to_l[ai] = interp_w_to_l

        if t == 0:
            if node.dest_bbox is None:
                pbar.update(1)
                continue
            else:
                if posed_dest_frames is None:
                    local_frame = dest_nodes[node.dest_subdiv_index].frame
                else:
                    local_frame = posed_dest_frames[node.dest_subdiv_index]
                w_to_l = kit.get_one_world_to_local(local_frame)
                points_in_dest_orig_local = kit.transform_points(
                    points_in_world, w_to_l)
                points_in_dest_orig_binvox = kit.mesh_space_to_voxel_space(
                    dest_binvox_xforms[node.dest_subdiv_index], 
                    points_in_dest_orig_local,
                    True)
                val_dest = trilinear_interp(
                    points_in_dest_orig_binvox, part_res, part_points,
                    dest_all_sdfs[node.dest_subdiv_index])
                all_vals.append(val_dest[:, None])

                pbar.update(1)
                continue

        if t == 1:
            if node.src_bbox is None:
                pbar.update(1)
                continue
            else:
                if posed_src_frames is None:
                    local_frame = src_nodes[node.src_subdiv_index].frame
                else:
                    local_frame = posed_src_frames[node.src_subdiv_index]
                w_to_l = kit.get_one_world_to_local(local_frame)
                points_in_src_orig_local = kit.transform_points(
                    points_in_world, w_to_l)
                points_in_src_orig_binvox = kit.mesh_space_to_voxel_space(
                    src_binvox_xforms[node.src_subdiv_index], 
                    points_in_src_orig_local,
                    True)
                val_src = trilinear_interp(
                    points_in_src_orig_binvox, part_res, part_points,
                    src_all_sdfs[node.src_subdiv_index])
                all_vals.append(val_src[:, None])

                pbar.update(1)
                continue

        if node.dest_bbox is None:
            # interpolate a zero bbox
            zero_bbox = bbox.get_zero_bbox()
            interp_bbox = bbox.interp_bbox(
                zero_bbox, node.src_bbox, t,
                interp_virt_ht[0], interp_w_to_l)
            mapping_interp_to_src = bbox.get_bbox_mapping(
                node.src_bbox, interp_bbox)

            points_in_interp_local = kit.transform_points(
                points_in_world, interp_w_to_l)
            points_in_interp_bbox = kit.transform_points(
                points_in_interp_local, interp_bbox.local_to_bbox)
            
            points_in_src_bbox = mapping_interp_to_src.send_all(
                points_in_interp_bbox)
            points_in_src_orig_bbox = kit.transform_points(
                points_in_src_bbox, node.src_subdiv_to_orig_xform)
            src_orig_node = src_nodes[node.src_subdiv_index]
            points_in_src_orig_local = kit.transform_points(
                points_in_src_orig_bbox,
                np.linalg.inv(src_orig_node.bbox.local_to_bbox))
            points_in_src_orig_binvox = kit.mesh_space_to_voxel_space(
                src_binvox_xforms[node.src_subdiv_index], 
                points_in_src_orig_local,
                True)

            val_src = trilinear_interp(
                points_in_src_orig_binvox, part_res, part_points,
                src_all_sdfs[node.src_subdiv_index])
            all_vals.append(val_src[:, None])

            pbar.update(1)
            continue
        
        if node.src_bbox is None:
            # interpolate a zero bbox
            zero_bbox = bbox.get_zero_bbox()
            interp_bbox = bbox.interp_bbox(
                node.dest_bbox, zero_bbox, t,
                interp_virt_ht[0], interp_w_to_l)
            mapping_interp_to_dest = bbox.get_bbox_mapping(
                node.dest_bbox, interp_bbox)

            points_in_interp_local = kit.transform_points(
                points_in_world, interp_w_to_l)
            points_in_interp_bbox = kit.transform_points(
                points_in_interp_local, interp_bbox.local_to_bbox)
            
            points_in_dest_bbox = mapping_interp_to_dest.send_all(
                points_in_interp_bbox)
            points_in_dest_orig_bbox = kit.transform_points(
                points_in_dest_bbox, node.dest_subdiv_to_orig_xform)
            dest_orig_node = dest_nodes[node.dest_subdiv_index]
            points_in_dest_orig_local = kit.transform_points(
                points_in_dest_orig_bbox,
                np.linalg.inv(dest_orig_node.bbox.local_to_bbox))
            points_in_dest_orig_binvox = kit.mesh_space_to_voxel_space(
                dest_binvox_xforms[node.dest_subdiv_index], 
                points_in_dest_orig_local,
                True)

            val_dest = trilinear_interp(
                points_in_dest_orig_binvox, part_res, part_points,
                dest_all_sdfs[node.dest_subdiv_index])
            all_vals.append(val_dest[:, None])

            pbar.update(1)
            continue

        interp_bbox: Bbox = bbox.interp_bbox(
            node.dest_bbox, node.src_bbox, t,
            interp_virt_ht[0], interp_w_to_l)
        src_bbox: Bbox = node.src_bbox
        dest_bbox: Bbox = node.dest_bbox
        mapping_interp_to_dest = bbox.get_bbox_mapping(
            node.dest_bbox, interp_bbox)
        mapping_interp_to_src = bbox.get_bbox_mapping(
            node.src_bbox, interp_bbox)

        points_in_interp_local = kit.transform_points(
            points_in_world, interp_w_to_l)
        points_in_interp_bbox = kit.transform_points(
            points_in_interp_local, interp_bbox.local_to_bbox)

        points_in_dest_bbox = mapping_interp_to_dest.send_all(
            points_in_interp_bbox)
        points_in_dest_orig_bbox = kit.transform_points(
            points_in_dest_bbox, node.dest_subdiv_to_orig_xform)
        dest_orig_node = dest_nodes[node.dest_subdiv_index]
        points_in_dest_orig_local = kit.transform_points(
            points_in_dest_orig_bbox,
            np.linalg.inv(dest_orig_node.bbox.local_to_bbox))
        points_in_dest_orig_binvox = kit.mesh_space_to_voxel_space(
            dest_binvox_xforms[node.dest_subdiv_index], 
            points_in_dest_orig_local,
            True)

        points_in_src_bbox = mapping_interp_to_src.send_all(
            points_in_interp_bbox)
        points_in_src_orig_bbox = kit.transform_points(
            points_in_src_bbox, node.src_subdiv_to_orig_xform)
        src_orig_node = src_nodes[node.src_subdiv_index]
        points_in_src_orig_local = kit.transform_points(
            points_in_src_orig_bbox,
            np.linalg.inv(src_orig_node.bbox.local_to_bbox))
        points_in_src_orig_binvox = kit.mesh_space_to_voxel_space(
            src_binvox_xforms[node.src_subdiv_index], 
            points_in_src_orig_local,
            True)

        val_dest = trilinear_interp(
            points_in_dest_orig_binvox, part_res, part_points,
            dest_all_sdfs[node.dest_subdiv_index])
        val_src = trilinear_interp(
            points_in_src_orig_binvox, part_res, part_points,
            src_all_sdfs[node.src_subdiv_index])

        interp_sdf = kit.lerp(val_dest, val_src, t)
        all_vals.append(interp_sdf[:, None])

        pbar.update(1)
    pbar.close()

    if len(all_vals) == 0:
        all_vals = [np.zeros((dim**3, 1))]
    whole_sdf_grid = np.concatenate(all_vals, axis=-1)
    whole_sdf_grid = np.min(whole_sdf_grid, axis=-1)
    whole_sdf_grid = np.reshape(whole_sdf_grid, (dim, dim, dim))

    whole_sdf_grid = np.swapaxes(whole_sdf_grid, 0, 1)
    whole_sdf_grid = -whole_sdf_grid if negate else whole_sdf_grid
    vertices, triangles = mcubes.marching_cubes(whole_sdf_grid, level_set)
    vertices[:, 0] = vertices[:, 0] / (dim - 1) - 0.5
    vertices[:, 1] = vertices[:, 1] / (dim - 1)
    vertices[:, 2] = vertices[:, 2] / (dim - 1) - 0.5
    mesh = trimesh.Trimesh(vertices, triangles)
    trimesh.repair.fix_normals(mesh)

    return mesh, all_interp_ht, all_interp_frames, all_interp_w_to_l

