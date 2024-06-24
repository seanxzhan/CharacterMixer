import bpy
import os
import json


def print_obj_names(lst):
    for obj in lst:
        if obj is None:
            print('None')
        else:
            print(obj.name)


def load(path):
    bpy.ops.import_scene.fbx(
        filepath=path,
        force_connect_children=True,
        automatic_bone_orientation=True)
    deselect_all()


def deselect_all():
    for obj in bpy.context.selected_objects:
        # obj.select_set(False)
        obj.select = False


def existsP(name):
    names = []
    for obj in bpy.data.objects:
        names.append(obj.name)

    return name in names


def clear():
    for obj in bpy.context.scene.objects:
        # obj.select_set(True)
        obj.select = True
    bpy.ops.object.delete()


def get_all_bone_names():
    armature = bpy.context.object.data
    bpy.ops.object.mode_set(mode='EDIT')

    all_bone_names = []
    for b in armature.edit_bones:
        all_bone_names.append(b.name)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    return all_bone_names


def get_full_armature(all_b_names, scale_info=None):
    bpy.ops.object.mode_set(mode='POSE')
    armature = bpy.context.object.pose

    armature_info = {}
    for b_name in all_b_names:
        bone = armature.bones[b_name]
        bh = [bone.head[0], bone.head[1], bone.head[2]]
        bt = [bone.tail[0], bone.tail[1], bone.tail[2]]
        
        bone_info = {}
        if scale_info != None:
            bone_info["scale"] = scale_info[b_name]
        bone_info["head"] = bh
        bone_info["tail"] = bt
        
        armature_info[b_name] = bone_info

    bpy.ops.object.mode_set(mode='OBJECT')
    
    return armature_info


def get_root_name():
    bpy.ops.object.mode_set(mode='POSE')
    armature = bpy.context.object.pose
    
    root_joint = armature.bones[0]
    while True:
        parent = root_joint.parent
        if parent == None:
            break
        root_joint = parent
        
    bpy.ops.object.mode_set(mode='OBJECT')
    return root_joint.name


def get_joints_dict(root_name):
    bpy.ops.object.mode_set(mode='POSE')
    armature = bpy.context.object.pose
    
    joint_info = {}
    this_level = [root_name]
    
    while this_level:
        next_level = []
        for b_name in this_level:
            curr_bone = armature.bones[b_name]
            pos = curr_bone.head
            joint_info[b_name] = {}
            joint_info[b_name]['pos'] = pos
            
            bone_chs = curr_bone.children
            ch_names = list(map(lambda x: x.name, bone_chs))
            joint_info[b_name]['ch'] = ch_names
            next_level += ch_names
            
            bone_pa = curr_bone.parent
            joint_info[b_name]['pa'] = bone_pa.name\
                if bone_pa != None else 'None'
        this_level = next_level
    return joint_info


def record_info(root_name, joint_dict, file):
    # write bone head positions
    for k, v in joint_dict.items():
        file.write('joints {0} {1:.8f} {2:.8f} {3:.8f} \n'.format(
            k, v['pos'][0], v['pos'][1], v['pos'][2]))
    
    # write root name
    file.write('root {} \n'.format(root_name))
    
    # getting skinning weights using vertex groups
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = list(filter(lambda x: x.type == 'MESH', bpy.data.objects))[0]
    armature = list(
        filter(lambda x: x.type == 'ARMATURE', bpy.data.objects))[0]
    obj_verts = obj.data.vertices
    obj_group_names = [g.name for g in obj.vertex_groups]
    vert_dict = {}
    for i in range(len(obj_verts)):
        vert_dict[i] = []
        
    for bone in armature.pose.bones:
        if bone.name not in obj_group_names:
            continue
        gidx = obj.vertex_groups[bone.name].index
        bone_verts = [
            v for v in obj_verts if gidx in [g.group for g in v.groups]]
        for v in bone_verts:
            for g in v.groups:
                if g.group == gidx:
                    w = g.weight
                    vert_dict[v.index].append((bone.name, w))
    
    # write skinning weights
    for vert_idx, lst_of_w in vert_dict.items():
        line = 'skin {} '.format(vert_idx)
        for tup in lst_of_w:
            tup_info = '{0} {1:.4f} '.format(tup[0], tup[1])
            line += tup_info
        line += '\n'
        file.write(line)
    
    # write bone hierarchy
    for k, v in joint_dict.items():
        if v['pa'] != 'None':
            file.write('hier {0} {1} \n'.format(v['pa'], k))


def save_armature_info(object_id, output_dir, armature_info):
    filename = object_id + '.json'
    out_file = os.path.join(output_dir, filename)
    with open(out_file, 'w') as f:
        json.dump(armature_info, f)


def get_skel_rig_info(input_path, obj_id, output_skel_dir, output_rig_dir):
    clear()
    load(input_path)
    all_b_names = get_all_bone_names()
    armature_info = get_full_armature(all_b_names)
    save_armature_info(obj_id, output_skel_dir, armature_info)

    root_name = get_root_name()
    joint_info = get_joints_dict(root_name)
    rig_info_path = os.path.join(output_rig_dir, "{}.txt".format(obj_id))
    with open(rig_info_path, 'w') as file:
        record_info(root_name, joint_info, file)
    bpy.ops.object.mode_set(mode='OBJECT')


if __name__ == '__main__':
    clear()

    data_dir = 'data/raw_data'

    in_fbx_dir = os.path.join(data_dir, 'fbx')
    output_skel_dir = os.path.join(data_dir, 'skel')
    output_rig_dir = os.path.join(data_dir, 'rig')

    for fn in os.listdir(in_fbx_dir):
        if fn[0] == '.':
            continue
        id = fn.split('.')[0]
        input_path = os.path.join(in_fbx_dir, fn)
        get_skel_rig_info(input_path, id, output_skel_dir, output_rig_dir)
