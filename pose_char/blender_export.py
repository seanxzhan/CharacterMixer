import os
import sys
import math
import mathutils
import bpy
import numpy as np


# this file is written in bpy 2.79, intended to be used by bpy 2.79


def export_blend(hierarchy, 
                 all_ht: np.ndarray, all_coord_frames: np.ndarray,
                 fp):
    arm_data = bpy.data.armatures.new('Armature')
    arm_obj = bpy.data.objects.new('MyArmature', arm_data)
    bpy.context.scene.objects.active = arm_obj

    scene = bpy.context.scene
    scene.objects.link(arm_obj)
    scene.objects.active = arm_obj

    bpy.ops.object.mode_set(mode='EDIT')

    num_bones = len(all_ht)
    for i in range(num_bones):
        bone = arm_data.edit_bones.new(f'bone_{i}')
        bone.head = all_ht[i][0]
        bone.tail = all_ht[i][1]
        
        mat3 = all_coord_frames[i][2:]
        rot = np.eye(4)
        rot[:3, :3] = mat3
        mat = mathutils.Matrix(rot).transposed()
        bone.matrix = mat
        bone.translate(mathutils.Vector(all_ht[i][0]))
        rot_matrix = mathutils.Matrix.Rotation(90.0 * 3.14159 / 180.0, 4, 'X')
        bone.matrix = rot_matrix * bone.matrix

    for hier in hierarchy:
        arm_data.edit_bones[hier[1]].parent = arm_data.edit_bones[hier[0]]
        pt = arm_data.edit_bones[hier[1]].parent.tail
        h = arm_data.edit_bones[hier[1]].head
        if pt - h <= mathutils.Vector((0.001, 0.001, 0.001)):
            arm_data.edit_bones[hier[1]].use_connect = True

    bpy.ops.wm.save_as_mainfile(filepath=fp)
 

if __name__ == "__main__":
    src_id = 432
    dest_id = 1299
    step = 5
    num_frames = 10
    # use low res to inspect interpolation
    res = 128

    interp_sdf_dir = 'absolute_path_to/CharacterMixer/results/interp_sdf' 
    aug_skel_path = interp_sdf_dir+f'/{src_id}_{dest_id}/seq_id_0_{res}/posed_id_0_{res}/step_{step}_{num_frames}_0_0/aug_skel.npz'
    out_fp = interp_sdf_dir+\
        f'/{src_id}_{dest_id}/seq_id_0_{res}/posed_id_0_{res}/step_{step}_{num_frames}_0_0/'+\
        f'aug_skel_{src_id}_{dest_id}_step_{step}.blend'

    aug_skel_data = np.load(aug_skel_path, allow_pickle=True)
    hierarchy = aug_skel_data['hierarchy']
    all_ht = aug_skel_data['all_ht']
    all_frames = aug_skel_data['all_frames']

    export_blend(hierarchy, all_ht, all_frames, out_fp)
