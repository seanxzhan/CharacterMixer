import os
import time
import trimesh
import numpy as np
import argparse
from pose_char import pose_utils, drive_src_rigs
from utils import misc, visualize
from mixer_interp import interp_skel, InterpInfo, gt, reconstruct
from produce_results import make_t_steps


def interpolate_posed(interp_info: InterpInfo,
                      pose_dir, interp_step, t,
                      posed_aug_frames=None,
                      posed_src_frames=None,
                      posed_dest_frames=None,
                      dim=128, part_res=128,
                      expt_dir=None, adj=''):
    if expt_dir is None:
        expt_dir = os.path.join(pose_dir, 'step_{}'.format(interp_step))

    # we are reconstructing with ground truth sdfs from voxels
    # NOTE: overwriting sdf grids and binvox xforms here
    src_out_sdf_path = os.path.join(
        pose_dir, f'gt_parts{adj}', '{}.npz'.format(interp_info.src_id))
    src_sdf_info = np.load(src_out_sdf_path, allow_pickle=True)
    dest_out_sdf_path = os.path.join(
        pose_dir, f'gt_parts{adj}', '{}.npz'.format(interp_info.dest_id))
    dest_sdf_info = np.load(dest_out_sdf_path, allow_pickle=True)
    src_binvox_xforms = src_sdf_info["binvox_xforms"]
    dest_binvox_xforms = dest_sdf_info["binvox_xforms"]
    src_sdf_grids = src_sdf_info["part_sdf_grids"]
    dest_sdf_grids = dest_sdf_info["part_sdf_grids"]

    # NOTE: this is only for 1 to 1 correspondence
    mesh, all_interp_ht, all_interp_frames, _ =\
        reconstruct.reconstruct_from_parts_interp_tree(
            t, part_res, interp_info.aug_nodes,
            interp_info.src_nodes, interp_info.dest_nodes,
            src_binvox_xforms, dest_binvox_xforms,
            src_sdf_grids, dest_sdf_grids,
            posed_aug_frames=posed_aug_frames,
            posed_src_frames=posed_src_frames,
            posed_dest_frames=posed_dest_frames,
            dim=dim)

    mesh_out_path = os.path.join(expt_dir, 'posed_gt.obj')
    img_out_path = os.path.join(expt_dir, 'posed_gt.png')

    trimesh.exchange.export.export_mesh(mesh, mesh_out_path, file_type='obj')

    # NOTE: comment if timing
    visualize.save_mesh_vis(mesh, img_out_path)

    armature_path = os.path.join(expt_dir, 'aug_skel.npz')

    anodes = interp_info.aug_nodes
    hierarchy = []
    for n in anodes[1:]:
        hierarchy.append((int(n.parent.id), int(n.id)))

    np.savez_compressed(armature_path,
                        hierarchy=hierarchy,
                        all_ht=all_interp_ht,
                        all_frames=all_interp_frames)
    
    aug_nodes_path = os.path.join(expt_dir, 'aug_nodes.npz')
    np.savez_compressed(aug_nodes_path,
                        aug_nodes=interp_info.aug_nodes,
                        all_ht=all_interp_ht,
                        all_frames=all_interp_frames)

    ht_path = os.path.join(expt_dir, 'ht.png')
    plt, _ = visualize.show_corresponding_bones(
        all_interp_ht, interp_info.aug_groups, 'src',
        custom_colors=interp_info.colors, rot=45, text=False)
    misc.save_fig(plt, '', ht_path)
    plt.close()

    ht_frames_path = os.path.join(expt_dir, 'ht_frames.png')
    plt = visualize.show_bones_and_local_coors(
        all_interp_ht, all_interp_frames)
    misc.save_fig(
        plt, '', ht_frames_path)
    plt.close()

    return img_out_path, ht_path, ht_frames_path


def pose_interp(interp_info: InterpInfo, pair_interp_dir, num_interp, res,
                aug_joint_angles, interp_step, t,
                seq_id, seq_frame, num_frames):
    aug_rot_mats = pose_utils.get_rot_mats(aug_joint_angles)
    
    seq_dir = os.path.join(
        pair_interp_dir, 'seq_id_{}_{}'.format(seq_id, res))
    misc.check_dir(seq_dir)
    pose_dir = os.path.join(seq_dir, f'posed_id_{seq_frame}_{res}')
    misc.check_dir(pose_dir)

    expt_dir = os.path.join(
        pose_dir,
        f'step_{interp_step}_{num_interp}_{seq_frame}_{num_frames}')
    misc.check_dir(expt_dir)

    posed_aug_frames, src_rot_mats, posed_src_frames,\
        dest_rot_mats, posed_dest_frames =\
            drive_src_rigs.pose_src_dest_from_aug(
                t, interp_info, aug_rot_mats, aug_joint_angles, expt_dir)

    gt_start_time = time.time()

    # voxelize posed meshes
    # NOTE: this function just segments the parts
    print("---------------- segmenting posed gt parts ----------------")
    gt.get_gt(interp_info.src_info, interp_info.dest_info,
              src_rot_mats, dest_rot_mats, pose_dir,
              res=res, part_res=res, do_pose=True,
              posed_src_local_coords=posed_src_frames,
             posed_dest_local_coords=posed_dest_frames,
             just_segment=True, n_workers=1)
    print("---------------- done segmenting posed gt parts ----------------")

    # reconstruct from voxelized parts of posed meshes
    print("---------------- reconstructing vox posed ----------------")
    interpolate_posed(
       interp_info, 
       pose_dir, interp_step, t,
       posed_aug_frames=posed_aug_frames,
       posed_src_frames=posed_src_frames,
       posed_dest_frames=posed_dest_frames,
       dim=res, part_res=res,
       expt_dir=expt_dir)
    print("---------------- done reconstructing vox posed ----------------")
    
    gt_end_time = time.time()
    gt_time = gt_end_time - gt_start_time
    print("gt_time: ", gt_time)
    gt_time_path = os.path.join(expt_dir, 'time_gt.txt')
    with open(gt_time_path, 'w') as f:
       f.write(str(gt_time))

    print("---------------- reconstructing gt posed ----------------")
    gt.get_gt(interp_info.src_info, interp_info.dest_info,
               src_rot_mats, dest_rot_mats, pose_dir,
               res=res, part_res=res, do_pose=True,
               posed_src_local_coords=posed_src_frames,
               posed_dest_local_coords=posed_dest_frames,
               just_segment=False, n_workers=1,
               cg=interp_info.corres_groups, colors=interp_info.colors)
    print("---------------- done reconstructing gt posed ----------------")


if __name__ == "__main__":
    out_dir = os.path.join('results', 'interp_sdf')
    misc.check_dir(out_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_id', type=int)
    parser.add_argument('--dest_id', type=int)
    parser.add_argument('--res', type=int)
    parser.add_argument('--seq_id', type=int)
    parser.add_argument('--num_interp', type=int)
    parser.add_argument('--frame_start', type=int)
    parser.add_argument('--frame_end', type=int)
    parser.add_argument('--interp_step', type=int)
    parser.add_argument('--fix_t', type=int)
    args = parser.parse_args()

    src_id = args.src_id
    dest_id = args.dest_id
    res = args.res
    num_interp = args.num_interp
    delta_t = 1 / num_interp
    interp_step = args.interp_step
    t = interp_step * delta_t
    # if fix_t is false, num_interp, interp_step and t will be overwritten
    fix_t = bool(args.fix_t)
    seq_id = args.seq_id

    pair_interp_dir = os.path.join(out_dir, '{}_{}'.format(src_id, dest_id))
    seq_dir = os.path.join(pair_interp_dir, 'seq_id_{}_{}'.format(seq_id, res))
    
    lst_of_joint_angles = []

    if seq_id != 0:
        # an actual animated sequence
        rots_fp = os.path.join('anims', f'{src_id}_{dest_id}.npz')
        all_mats = np.load(rots_fp, allow_pickle=True)["all_mats"]
        for json_mat in all_mats:
            ja = {}
            for k, v in json_mat.items():
                ja[int(k)] = v
            lst_of_joint_angles.append(ja)
    else:
        # rest pose
        interp_info: InterpInfo = interp_skel.export_aug_skel(
            src_id, dest_id, pair_interp_dir, num_interp, vis=True)
        num_aug_bones = len(interp_info.aug_nodes)
        num_frames = num_interp + 1
        lst_of_joint_angles = [
            pose_utils.make_rest_pose_info(num_aug_bones)] * num_frames

    lst_of_t = []
    if fix_t:
        lst_of_t = [t] * len(lst_of_joint_angles)
        lst_of_interp_steps = [interp_step] * len(lst_of_joint_angles)
        if seq_id == 3:
            _, _, num_interp = make_t_steps.make_inf_loop(len(lst_of_joint_angles))
    else:
        if seq_id == 0:
            lst_of_t, lst_of_interp_steps, num_interp =\
                make_t_steps.make_t_pose(len(lst_of_joint_angles))
        else:
            if (src_id, dest_id) in [(432, 1299), (1379, 14035)]:
                lst_of_joint_angles *= 2
                lst_of_t, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_inf_loop(len(lst_of_joint_angles))

            if (src_id, dest_id) == (2097, 2091):
                lst_of_t, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_2097_2091(len(lst_of_joint_angles))

            if (src_id, dest_id) == (4010, 1919):
                lst_of_t, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_4010_1919(len(lst_of_joint_angles))
                
            if (src_id, dest_id) == (12852, 12901):
                lst_of_t, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_12852_12901(len(lst_of_joint_angles))
                
            if (src_id, dest_id) == (16880, 16827):
                lst_of_t, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_16880_16827(len(lst_of_joint_angles))

    if seq_id != 0:
        interp_info: InterpInfo = interp_skel.export_aug_skel(
            src_id, dest_id, pair_interp_dir, num_interp, vis=True)

    n_frames = len(lst_of_joint_angles)
    for i in range(n_frames):
        print(f'=============== Processing frame {i+1}/{n_frames} ===============')
        this_interp_step = lst_of_interp_steps[i]
        this_frame_t = lst_of_t[i]

        if seq_id == 0:
            seq_frame = 0
            num_frames = 0
        else:
            seq_frame = i + 1
            num_frames = n_frames

        pose_interp(interp_info, pair_interp_dir, num_interp, res,
                    lst_of_joint_angles[i], this_interp_step, this_frame_t,
                    seq_id, seq_frame, num_frames)
