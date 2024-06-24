import os
import copy
import numpy as np
from typing import List
from anytree import AnyNode
from pose_char import pose_utils
from mixer_interp import InterpInfo
from utils import misc, kit, visualize


def pose_src_dest_from_aug(t, interp_info: InterpInfo,
                           aug_rot_mats, aug_joint_angles, expt_dir):
    """
    out: src_rot_mats, dest_rot_mats
    """
    comp_nodes = interp_info.comp_nodes
    posed_aug_nodes = copy.deepcopy(interp_info.aug_nodes)
    src_rot_mats_dict = {}
    dest_rot_mats_dict = {}
    # first, pose the augmented skeleton from aug_rot_mats
    # posed skeleton has updated frames (local coords)
    write_frame_and_xform_to_aug_nodes(t, posed_aug_nodes, interp_info)
    pose_utils.pose_bones(posed_aug_nodes[0], aug_rot_mats)

    posed_aug_frames = kit.gather_frames(
        posed_aug_nodes[0],
        np.zeros((len(posed_aug_nodes), 5, 3), dtype=np.float32), 0)

    skel_rot = 45

    plt, _ = visualize.show_corresponding_bones(
        posed_aug_frames[:, :2], interp_info.aug_groups, 'src',
        custom_colors=interp_info.colors, rot=skel_rot
        # , text=True
        )
    misc.save_fig(plt, '', os.path.join(expt_dir, 'posed_aug_ht.png')
                #   , transparent=False
                  )
    plt.close()

    plt = visualize.show_bones_and_local_coors(
        posed_aug_frames[:, :2], posed_aug_frames)
    misc.save_fig(
        plt, '', os.path.join(expt_dir, 'posed_aug_ht_frames.png'))
    plt.close()

    for cn in comp_nodes:
        src_ids = cn.src_ids
        dest_ids = cn.dest_ids
        aug_ids = cn.aug_ids
        if src_ids == -1:
            assert len(aug_ids) == len(dest_ids)
            for i in range(len(aug_ids)):
                dest_rot_mats_dict[dest_ids[i]] = aug_rot_mats[aug_ids[i]]
            continue
        if dest_ids == -1:
            assert len(aug_ids) == len(src_ids)
            for i in range(len(aug_ids)):
                src_rot_mats_dict[src_ids[i]] = aug_rot_mats[aug_ids[i]]
            continue
        if len(src_ids) == 1 and len(dest_ids) == 1:
            src_rot_mats_dict[src_ids[0]] = aug_rot_mats[aug_ids[0]]
            dest_rot_mats_dict[dest_ids[0]] = aug_rot_mats[aug_ids[0]]
            continue
        if len(src_ids) == 1 and len(dest_ids) > 1:
            x_accum = 0; y_accum = 0; z_accum = 0
            assert len(aug_ids) == len(dest_ids)    # len aug nodes = max dof
            for i in range(len(aug_ids)):
                dest_rot_mats_dict[dest_ids[i]] = aug_rot_mats[aug_ids[i]]
                x_accum += aug_joint_angles[aug_ids[i]]["x_rot"]
                y_accum += aug_joint_angles[aug_ids[i]]["y_rot"]
                z_accum += aug_joint_angles[aug_ids[i]]["z_rot"]
            x_avg = x_accum / len(aug_ids)
            y_avg = y_accum / len(aug_ids)
            z_avg = z_accum / len(aug_ids)
            src_rot_mats_dict[src_ids[0]] = pose_utils.get_one_rot_mat(
                x_avg, y_avg, z_avg)
            continue
        if len(src_ids) > 1 and len(dest_ids) == 1:
            x_accum = 0; y_accum = 0; z_accum = 0
            assert len(aug_ids) == len(src_ids)    # len aug nodes = max dof
            for i in range(len(aug_ids)):
                src_rot_mats_dict[src_ids[i]] = aug_rot_mats[aug_ids[i]]
                x_accum += aug_joint_angles[aug_ids[i]]["x_rot"]
                y_accum += aug_joint_angles[aug_ids[i]]["y_rot"]
                z_accum += aug_joint_angles[aug_ids[i]]["z_rot"]
            x_avg = x_accum / len(aug_ids)
            y_avg = y_accum / len(aug_ids)
            z_avg = z_accum / len(aug_ids)
            dest_rot_mats_dict[dest_ids[0]] = pose_utils.get_one_rot_mat(
                x_avg, y_avg, z_avg)
            continue
    
    src_rot_mats = []
    dest_rot_mats = []
    for i in range(len(interp_info.src_nodes)):
        src_rot_mats.append(src_rot_mats_dict[i])
    for i in range(len(interp_info.dest_nodes)):
        dest_rot_mats.append(dest_rot_mats_dict[i])

    posed_src_nodes = copy.deepcopy(interp_info.src_nodes)
    posed_dest_nodes = copy.deepcopy(interp_info.dest_nodes)
    pose_utils.pose_bones(posed_src_nodes[0], src_rot_mats_dict)
    pose_utils.pose_bones(posed_dest_nodes[0], dest_rot_mats_dict)
    posed_src_frames = kit.gather_frames(
        posed_src_nodes[0],
        np.zeros((len(posed_src_nodes), 5, 3), dtype=np.float32), 0)
    posed_dest_frames = kit.gather_frames(
        posed_dest_nodes[0],
        np.zeros((len(posed_dest_nodes), 5, 3), dtype=np.float32), 0)

    plt, _ = visualize.show_corresponding_bones(
        posed_src_frames[:, :2], interp_info.corres_groups, 'src',
        custom_colors=interp_info.colors, rot=skel_rot
        # , text=True
        )
    misc.save_fig(plt, '', os.path.join(expt_dir, 'posed_src_ht.png')
                #   , transparent=False
                  )
    plt.close()

    plt = visualize.show_bones_and_local_coors(
        posed_src_frames[:, :2], posed_src_frames)
    misc.save_fig(
        plt, '', os.path.join(expt_dir, 'posed_src_ht_frames.png'))
    plt.close()

    plt, _ = visualize.show_corresponding_bones(
        posed_dest_frames[:, :2], interp_info.corres_groups, 'dest',
        custom_colors=interp_info.colors, rot=skel_rot
        # , text=True
        )
    misc.save_fig(plt, '', os.path.join(expt_dir, 'posed_dest_ht.png')
                #   , transparent=False
                  )
    plt.close()

    plt = visualize.show_bones_and_local_coors(
        posed_dest_frames[:, :2], posed_dest_frames)
    misc.save_fig(
        plt, '', os.path.join(expt_dir, 'posed_dest_ht_frames.png'))
    plt.close()

    return posed_aug_frames,\
        src_rot_mats, posed_src_frames,\
        dest_rot_mats, posed_dest_frames


def write_frame_and_xform_to_aug_nodes(t, nodes: List[AnyNode],
                                       interp_info: InterpInfo):
    """
    in: interp time, augmented nodes
    """
    for n in nodes:
        new_virt_ht = kit.lerp(n.dest_virt_ht, n.src_virt_ht, t)
        n.frame = kit.get_joints_local_coords(
            1, new_virt_ht[None, :])[0]

        if np.array_equal(n.frame[3], np.zeros((3,), dtype=np.float32)):
            n.frame[3][0] = 0
            n.frame[3][1] = 1
            n.frame[3][2] = 0
        n.xform = np.eye(4, dtype=np.float32)
