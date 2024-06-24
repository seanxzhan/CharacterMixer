import os
import bpy
import mathutils
import numpy as np
from math import degrees

# This file is intended to be used by bpy 2.93

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def output_rot_mats(fp):
    bpy.ops.wm.open_mainfile(filepath=fp)

    arm_data = bpy.data.objects['MyArmature']
    arm = arm_data.pose
    bpy.ops.object.mode_set(mode='POSE')

    start = bpy.context.scene.frame_start
    end = bpy.context.scene.frame_end + 1

    all_arm_mats = []
    for frame in range(start, end):
        bpy.context.scene.frame_set(frame)
        arm_mats = {}
        for i in range(len(arm.bones)):
            bone = arm.bones[f'bone_{i}']
            rot = bone.rotation_quaternion
            rot = mathutils.Quaternion(rot).to_euler('XYZ')
            dict = {}
            dict["x_rot"] = degrees(rot.x)
            dict["y_rot"] = degrees(rot.y)
            dict["z_rot"] = degrees(rot.z)
            arm_mats[str(i)] = dict

        all_arm_mats.append(arm_mats)
    
    return all_arm_mats


if __name__ == "__main__":
    src_id = 432
    dest_id = 1299
    res = 128

    seq_id = 1

    out_dir = 'absolute_path_to/CharacterMixer/results/interp_sdf' 
    pair_interp_dir = os.path.join(out_dir, '{}_{}'.format(src_id, dest_id))
    seq_dir = os.path.join(
        pair_interp_dir, 'seq_id_{}_{}'.format(seq_id, res))
    check_dir(seq_dir)

    fbx_fp = os.path.join(seq_dir, f'{src_id}_{dest_id}_{seq_id}.blend')
    all_arm_rots = output_rot_mats(fbx_fp)

    rots_fp = os.path.join(seq_dir, f'{src_id}_{dest_id}_{seq_id}.npz')
    np.savez_compressed(rots_fp, all_mats=all_arm_rots)
