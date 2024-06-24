import os
import gc
import queue
import trimesh
import numpy as np
from tqdm import tqdm
from data_prep import segment
from utils import misc, kit, visualize
from mixer_interp import Info, reconstruct
from pose_char import pose_utils
from multiprocessing import Queue, Process


def get_gt(src_info: Info, dest_info: Info,
           src_rot_mats, dest_rot_mats,
           pose_dir, res=128, part_res=128, do_pose=True,
           posed_src_local_coords=None,
           posed_dest_local_coords=None,
           just_segment=False, n_workers=-1, adj='',
           cg=[], colors=[]):

    gt_parts_dir = os.path.join(pose_dir, f'gt_parts{adj}')
    misc.check_dir(gt_parts_dir)
    vox_dir = os.path.join(gt_parts_dir, 'vox')
    misc.check_dir(vox_dir)
    vox_c_dir = os.path.join(vox_dir, '{}_vox_c'.format(res))
    misc.check_dir(vox_c_dir)
    vox_nc_dir = os.path.join(vox_dir, '{}_vox_nc'.format(res))
    misc.check_dir(vox_nc_dir)
    obj_dir = os.path.join(gt_parts_dir, 'obj')
    misc.check_dir(obj_dir)

    print("getting gt mesh and img of src model: {}".format(src_info.model_id))
    src_parts_sdf_path, src_vox_c_dir, src_new_verts, src_package = pose_sdf_one_model(
        pose_dir, src_info, src_rot_mats, res, part_res,
        vox_c_dir, vox_nc_dir, obj_dir, do_pose, posed_src_local_coords,
        just_segment, n_workers=n_workers, adj=adj,
        cg=cg, colors=colors, src_or_dest='src')
    print("getting gt mesh and img of dest model: {}".format(dest_info.model_id))
    dest_parts_sdf_path, dest_vox_c_dir, dest_new_verts, dest_pacakge = pose_sdf_one_model(
        pose_dir, dest_info, dest_rot_mats, res, part_res,
        vox_c_dir, vox_nc_dir, obj_dir, do_pose, posed_dest_local_coords,
        just_segment, n_workers=n_workers, adj=adj,
        cg=cg, colors=colors, src_or_dest='dest')
    
    return src_parts_sdf_path, src_vox_c_dir, src_new_verts, src_package, \
        dest_parts_sdf_path, dest_vox_c_dir, dest_new_verts, dest_pacakge
    

def pose_sdf_one_model(pose_dir, model_info: Info, model_rot_mats,
                       res, part_res,
                       vox_c_dir, vox_nc_dir, obj_dir, do_pose=True,
                       posed_local_coords=None, just_segment=False,
                       global_recon=False, n_workers=-1, adj='',
                       cg=[], colors=[], src_or_dest=''):
    model_id = model_info.model_id
    model_vox_c_dir = os.path.join(vox_c_dir, '{}_posed'.format(model_id))
    misc.check_dir(model_vox_c_dir)
    model_vox_nc_dir = os.path.join(vox_nc_dir, '{}_posed'.format(model_id))
    misc.check_dir(model_vox_nc_dir)
    model_obj_dir = os.path.join(obj_dir, '{}_posed'.format(model_id))
    misc.check_dir(model_obj_dir)

    out_sdf_path = os.path.join(pose_dir, f'gt_parts{adj}', '{}.npz'.format(model_id))

    if not do_pose:
        return out_sdf_path, model_vox_c_dir, None, None

    model_package = segment.get_model_parts(model_info.path_to_obj,
                                            model_info.path_to_rig_info,
                                            model_info.path_to_skel,
                                            just_rig_info=True)
    model_local_coords = model_package["local_coords"]
    model_world_to_local = model_package["world_to_local"]
    model_vertices = model_package["vertices"]
    model_faces = model_package["faces"]
    model_skinning_weights = model_package["skinning_weights"]
    model_joints_to_faces = model_package["joints_to_faces"]

    if posed_local_coords is None:
        model_root_idx, model_nodes, _, _, _ = segment.get_skel_tree(
            len(model_local_coords),
            model_info.path_to_rig_info, local_coords=model_local_coords)
        pose_utils.pose_bones(model_nodes[model_root_idx], model_rot_mats)
        model_new_local_coords = pose_utils.get_local_coords_from_tree(
            model_nodes)
    else:
        model_new_local_coords = posed_local_coords
    model_new_w_to_l = kit.get_world_to_local(model_new_local_coords)

    model_new_verts = pose_utils.pose_character(
        model_vertices, model_skinning_weights,
        model_world_to_local, model_new_w_to_l)
    
    if not just_segment:
        # make a new mesh model
        new_mesh = trimesh.Trimesh(model_new_verts, model_faces, process=False)
        mesh_out_path = os.path.join(pose_dir, 'gt_{}.obj'.format(model_id))
        img_out_path = os.path.join(pose_dir, 'gt_{}.png'.format(model_id))
        trimesh.exchange.export.export_mesh(
            new_mesh, mesh_out_path, file_type='obj')
        visualize.save_mesh_vis(new_mesh, img_out_path)

        # this is early stopping to get posed patches
        model_joints_to_verts = model_package["joints_to_verts"] 
        model_joints_to_faces = model_package["joints_to_faces"] 
        patches_out_path = os.path.join(pose_dir, f'gt_{model_id}_patches.png')
        visualize.save_corresponding_patches_trimesh(
            model_new_verts, model_faces,
            model_joints_to_verts, model_joints_to_faces,
            src_or_dest, cg, colors, patches_out_path)
        # uncomment this to skip global recon
        # return None, None, model_new_verts, model_package

    gt_recon_dir = os.path.join(pose_dir, 'gt_recon')
    misc.check_dir(gt_recon_dir)

    if global_recon:
        gt_whole_dir = os.path.join(pose_dir, 'gt_whole')
        misc.check_dir(gt_whole_dir)
        model_whole_dir = os.path.join(gt_whole_dir, str(model_id))
        misc.check_dir(model_whole_dir)
        out_vox_c_path = os.path.join(model_whole_dir, 'vox_c.binvox')
        out_vox_nc_path = os.path.join(model_whole_dir, 'vox_nc.binvox')

        # global recon
        print("voxelizing mesh...")
        if not (os.path.exists(out_vox_c_path) and
                os.path.exists(out_vox_nc_path)):
            kit.setup_vox(pose_dir)
            kit.voxelize_obj(
                pose_dir, 'gt_{}.obj'.format(model_id), res,
                out_vox_c_path, out_vox_nc_path)
            kit.teardown_vox(pose_dir)
        else:
            print("skipped")

        transform = kit.get_transform_from_binvox(out_vox_c_path, out_vox_nc_path)
        sdf_grid = kit.bin2sdf(out_vox_c_path)

        print("reconstructing gt from whole...")
        mesh_out_path = os.path.join(
            gt_recon_dir, 'gt_{}_recon_whole_{}.obj'.format(model_id, res))
        img_out_path = os.path.join(
            gt_recon_dir, 'gt_{}_recon_whole_{}.png'.format(model_id, res))
        if not os.path.exists(mesh_out_path):
            recon_mesh = reconstruct.reconstruct_whole(
                mesh_out_path, sdf_grid,
                transform=True, transform_info=transform,
                negate=True, level_set=0, show_msg=True)
            visualize.save_mesh_vis(recon_mesh, img_out_path)
        else:
            print("skipped, already reconstructed from whole")

    print("segmenting model gt part voxels...")
    # actually voxelize parts of the posed character
    if not os.path.exists(os.path.join(
            model_vox_nc_dir,
            '{}.binvox'.format(len(model_new_local_coords) - 1))):
    # if True:
        segment.multi_thread_vox_jumper(
            len(model_new_local_coords), part_res, model_new_w_to_l, model_new_verts,
            model_joints_to_faces, model_obj_dir, 'tmp',
            model_vox_c_dir, model_vox_nc_dir, n_workers=1)
    else:
        print("skipped, segmentation data already exists")

    print("processing gt voxelized parts...")
    # out_sdf_path = os.path.join(pose_dir, 'gt_parts', '{}.npz'.format(model_id))
    if not os.path.exists(out_sdf_path):
    # if True:
        fns = misc.sorted_alphanumeric(os.listdir(model_vox_c_dir))
        all_gt_sdfs = [[]] * len(fns)
        all_gt_binvox_xforms = [[]] * len(fns)

        num_workers = 6
        list_of_lists = misc.chunks(list(range(len(fns))), num_workers)

        q = Queue()
        workers = [
            Process(target=process_voxelized_parts,
                    args=(q, lst, fns, model_vox_c_dir, model_vox_nc_dir))
            for lst in list_of_lists
        ]

        for p in workers:
            p.start()
        pbar = tqdm(total=len(fns))
        while True:
            flag = True
            try:
                idx, binvox_xform, sdf = q.get(True, 1.0)
            except queue.Empty:
                flag = False
            if flag:
                all_gt_sdfs[idx] = sdf
                all_gt_binvox_xforms[idx] = binvox_xform
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

        np.savez_compressed(out_sdf_path,
                            part_sdf_grids=all_gt_sdfs,
                            binvox_xforms=all_gt_binvox_xforms)
    else:
        gt_data = np.load(out_sdf_path, allow_pickle=True)
        all_gt_sdfs = gt_data["part_sdf_grids"]
        all_gt_binvox_xforms = gt_data["binvox_xforms"]
        print("skipped, voxelized parts' sdf already exists")

    print("reconstructing gt from parts...")
    mesh_out_path = os.path.join(
        gt_recon_dir, 'gt_{}_recon_parts_{}.obj'.format(model_id, res))
    img_out_path = os.path.join(
        gt_recon_dir, 'gt_{}_recon_parts_{}.png'.format(model_id, res))
    # if not os.path.exists(mesh_out_path):
    if True and not just_segment:
        mesh = reconstruct.reconstruct_from_parts(
            mesh_out_path,
            all_gt_sdfs, all_gt_binvox_xforms, model_new_w_to_l, dim=res)
        visualize.save_mesh_vis(mesh, img_out_path)
    else:
        print("skipped")

    del all_gt_sdfs
    gc.collect()
    return out_sdf_path, model_vox_c_dir, model_new_verts, model_package


def process_voxelized_parts(q: Queue, indices, fns,
                            model_vox_c_dir, model_vox_nc_dir):
    for idx in indices:
        fn = fns[idx]
        part_c_path = os.path.join(model_vox_c_dir, fn)
        part_nc_path = os.path.join(model_vox_nc_dir, fn)
        transform = kit.get_transform_from_binvox(part_c_path, part_nc_path)
        part_voxels = misc.load_voxels(part_c_path)
        sdf = kit.bin2sdf(input=part_voxels)
        q.put([idx, transform, sdf])
