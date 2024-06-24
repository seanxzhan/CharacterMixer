import os
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import List
from data_prep import segment
from anytree import AnyNode, RenderTree
from utils import kit, bbox, misc, visualize
from corresp.group_corresp import find_corresp_groups
from mixer_interp import Info, make_model_info, InterpInfo


def interp_aug_skel(aug_skel_nodes, num_interp, src_id, dest_id):
    """
    each node has the folllowing attributes
    - id
    - src_virt_ht_xlated
    - dest_virt_ht_xlated
    - src_virt_ht
    - dest_virt_ht
    - src_bbox
    - dest_bbox
    nodes are in ascending order
    """
    num_nodes = len(aug_skel_nodes)
    all_new_ht = []
    all_new_local_coords = []
    all_new_w_to_l = []

    delta_t = 1 / num_interp
    for i in range(num_interp + 1):
        t = i * delta_t

        new_ht = np.zeros((num_nodes, 2, 3), dtype=np.float32)
        new_local_coords = np.zeros((num_nodes, 5, 3), dtype=np.float32)

        for j, node in enumerate(aug_skel_nodes):
            new_virt_ht = kit.lerp(
                node.dest_virt_ht, node.src_virt_ht, t)
            new_ht[j] = new_virt_ht

        new_local_coords = kit.get_joints_local_coords(num_nodes, new_ht)

        new_w_to_l = kit.get_world_to_local(new_local_coords)

        all_new_ht.append(new_ht)
        all_new_local_coords.append(new_local_coords)
        all_new_w_to_l.append(new_w_to_l)

    return all_new_ht, all_new_local_coords, all_new_w_to_l


def get_aug_skel(src_root_idx, src_nodes: List[AnyNode],
                 dest_root_idx, dest_nodes: List[AnyNode],
                 corres_groups):
    # TODO: add xlation to this
    src_root = src_nodes[src_root_idx]
    dest_root = dest_nodes[dest_root_idx]

    src_xlation = src_root.xlation
    dest_xlation = dest_root.xlation

    # NOTE: each compound node should also maintain a hierarchy
    # root is the ancestor node
    # compound node tree's root xform should be hooked up to ancestor xform
    corres_groups_queue = copy.deepcopy(corres_groups)
    assigned_numbers = list(range(len(corres_groups_queue)))
    numbered_queue = list(zip(assigned_numbers, corres_groups_queue))
    ci = 0

    c_nodes = []
    for numbered_item in numbered_queue:
        assigned_number = numbered_item[0]
        cg = numbered_item[1]

        c_node = AnyNode(id=str(ci))

        src_g: List[int] = cg[0]
        dest_g: List[int] = cg[1]
        # the first node of each group determines the group

        src_g_nodes = [src_nodes[i] for i in src_g]
        dest_g_nodes = [dest_nodes[i] for i in dest_g]

        src_virt_ht, src_bboxes, src_subdiv_to_orig_xforms, src_subdiv_indices,\
            dest_virt_ht, dest_bboxes, dest_subdiv_to_orig_xforms, dest_subdiv_indices =\
            create_virtual_joints(corres_groups, src_g, dest_g, src_nodes, dest_nodes)
        c_node.all_src_virt_ht = src_virt_ht
        c_node.all_src_subdiv_bboxes = src_bboxes
        c_node.all_src_subdiv_to_orig = src_subdiv_to_orig_xforms
        c_node.all_src_subdiv_indices = src_subdiv_indices
        c_node.all_dest_virt_ht = dest_virt_ht
        c_node.all_dest_subdiv_bboxes = dest_bboxes
        c_node.all_dest_subdiv_to_orig = dest_subdiv_to_orig_xforms
        c_node.all_dest_subdiv_indices = dest_subdiv_indices

        # deal with vertex to void first
        if src_g[0] == -1:
            # locate the compound node cni that includes dest_g[0]'s parent
            # make dest_g[0] a compound node cnj, set cnj.parent = cni
            p = int(dest_nodes[dest_g[0]].parent.id)
            cn_idx = look_up_node_in_compound_tree(c_nodes, p, 1)
            if cn_idx == -1:
                print("ERROR: haven't dealt with this case")
                raise Exception("ERROR: haven't dealt with this case")
            c_node.src_ids = [-1]
            c_node.dest_ids = [int(x.id) for x in dest_g_nodes]
            c_node.parent = c_nodes[cn_idx]
            c_nodes.append(c_node)
            ci += 1
            continue
        if dest_g[0] == -1:
            # locate the compound node cni that includes src_g[0]'s parent
            # make src_g[0] a compound node cnj, set cnj.parent = cni
            p = int(src_nodes[src_g[0]].parent.id)
            cn_idx = look_up_node_in_compound_tree(c_nodes, p, 0)
            if cn_idx == -1:
                print("ERROR: haven't dealt with this case")
                raise Exception("ERROR: haven't dealt with this case")
            c_node.src_ids = [int(x.id) for x in src_g_nodes]
            c_node.dest_ids = [-1]
            c_node.parent = c_nodes[cn_idx]
            c_nodes.append(c_node)
            ci += 1
            continue

        src_g_parent: AnyNode = src_nodes[src_g[0]].parent
        dest_g_parent: AnyNode = dest_nodes[dest_g[0]].parent

        if src_g_parent is None and dest_g_parent is None:
            # root to root
            c_node.src_ids = [int(x.id) for x in src_g_nodes]
            c_node.dest_ids = [int(x.id) for x in dest_g_nodes]
            c_nodes.append(c_node)
            ci += 1
            continue
        
        if (src_g_parent is None) != (dest_g_parent is None):
            print("ERROR: root doesn't match to root!")
            raise Exception("ERROR: root doesn't match to root!")

        # neither has a None parent
        src_g_parent_id = int(src_g_parent.id)
        dest_g_parent_id = int(dest_g_parent.id)

        # group parent compound node id
        src_g_par_cn_id = look_up_node_in_compound_tree(
            c_nodes, src_g_parent_id, 0)
        dest_g_par_cn_id = look_up_node_in_compound_tree(
            c_nodes, dest_g_parent_id, 1)
        # print(cg)
        # print("src par cn id: ", src_g_par_cn_id, "dest par cn id: ", dest_g_par_cn_id)
        assert src_g_par_cn_id == dest_g_par_cn_id
        if src_g_par_cn_id is None:
            numbered_queue.append((assigned_number ,cg))
            continue
        c_node.src_ids = [int(x.id) for x in src_g_nodes]
        c_node.dest_ids = [int(x.id) for x in dest_g_nodes]            
        c_node.parent = c_nodes[src_g_par_cn_id]
        c_nodes.append(c_node)
        ci += 1

    print("---------- src ----------")
    for pre, _, node in RenderTree(src_nodes[src_root_idx]):
        print("%s%s" % (pre, node.id))
    print("---------- dest ----------")
    for pre, _, node in RenderTree(dest_nodes[dest_root_idx]):
        print("%s%s" % (pre, node.id))

    saved_c_nodes = copy.deepcopy(c_nodes)
    
    # one of src_xlation and dest_xlation is 0
    src_xlation = src_root.xlation
    dest_xlation = dest_root.xlation

    n_c_nodes = len(c_nodes)
    new_c_nodes = []
    for i in range(len(c_nodes)):
        inc = write_comp_node_attrs(
            saved_c_nodes, c_nodes[i], n_c_nodes,
            src_xlation, dest_xlation,
            new_c_nodes)
        n_c_nodes += inc

    print("---------- compound ----------")
    for pre, _, node in RenderTree(saved_c_nodes[0]):
        print("{}{}, src: {}, dest: {}, aug: {}".format(
            pre, node.id, node.src_ids, node.dest_ids, node.aug_ids))

    groups = get_augmented_groups(c_nodes)
    print("---------- augmented groups ----------")
    print(groups)

    print("---------- augmented ----------")
    all_c_nodes = c_nodes + new_c_nodes
    for pre, _, node in RenderTree(all_c_nodes[0]):
        print("%s%s, src: %s, dest: %s" % (
            pre, node.id, node.src_subdiv_index, node.dest_subdiv_index))

    return all_c_nodes, groups, saved_c_nodes


def get_augmented_groups(c_nodes):
    groups = []

    total_num = len(c_nodes)
    for n in c_nodes:
        lst = []
        idx = int(n.id)
        lst.append(idx)
        num_sub_bones = len(n.all_src_virt_ht) - 1
        for i in range(num_sub_bones):
            lst.append(total_num)
            total_num += 1
        groups.append((lst, ))
    
    return groups


def look_up_node_in_compound_tree(nodes: List[AnyNode], id: int, src_or_dest: int):
    # src_to_dest can either be 0 (src) or 1 (dest)
    for i, n in enumerate(nodes):
        g_list = n.src_ids if src_or_dest == 0 else n.dest_ids
        if id in g_list:
            return i
    return None


def make_bbox_around_root(verts_in_world, root_ht):
    local_coords = kit.get_joints_local_coords(1, root_ht[None, :])[0]
    world_to_local = kit.get_one_world_to_local(local_coords)
    verts_in_root = kit.transform_points(verts_in_world, world_to_local)
    box = bbox.get_bbox(verts_in_root, root_ht[0], world_to_local)
    return box, world_to_local


def mapped_ht_of_node_matching_void(mapping: bbox.Mappings, ht_in_world,
                                    bbox1: bbox.Bbox, bbox2: bbox.Bbox,
                                    w_to_l1, w_to_l2):
    """mapping is from 1 -> 2, ht_in_world belongs to 1
    """
    ht_in_root_local = kit.transform_points(
        ht_in_world, w_to_l1)
    ht_in_root_bbox = kit.transform_points(
        ht_in_root_local, bbox1.local_to_bbox)
    mapped_ht_in_root_bbox = mapping.send_all(
        ht_in_root_bbox)
    mapped_ht_in_root_local = kit.transform_points(
        mapped_ht_in_root_bbox, np.linalg.inv(bbox2.local_to_bbox))
    mapped_ht_in_world = kit.transform_points(
        mapped_ht_in_root_local, np.linalg.inv(w_to_l2))
    return mapped_ht_in_world


def write_bboxes(node: AnyNode, all_local_verts: List, xforms):
    """
    since these bounding boxes are local, it doesn't matter if all_local_verts
    are translated or not, as long as all_local_verts and xforms are consistent
    in this case, we are using the original vertices and frames
    original: not translated, could be either
        - gt vertices 
        - normalized ml vertices 
    """
    if node == None:
        return

    idx = int(node.id)
    # print(idx)
    box = bbox.get_bbox(all_local_verts[idx],
                        node.frame[0],
                        xforms[idx])
    node.bbox = box

    for child in node.children:
        write_bboxes(child, all_local_verts, xforms)


def get_traced_info(group: List[int], nodes: List[AnyNode]):
    """
    out: total traced bone length, each node's interval in % along the bones
    """
    # NOTE this assumes that group[i+1] is a child of group[i]
    total_traced_len = 0
    for idx in group:
        total_traced_len += nodes[idx].bone_len

    accum = 0
    node_percentage = []
    for idx in group:
        node_percentage.append(accum / total_traced_len)
        accum += nodes[idx].bone_len

    node_per1 = copy.deepcopy(node_percentage)
    node_per2 = node_percentage[1:]
    node_per2.append(1)
    nodes_intervals = list(zip(node_per1, node_per2))

    # [0, 10, 40]
    # [10, 40, 100] (delete first, append 100 to the end)
    # -> [(0, 10), (10, 40), (40, 100)]
    return total_traced_len, nodes_intervals


def look_up_node_corresp_in_groups(corres_groups, id: int, src_or_dest: int):
    """
    src_or_dest: if the id provided is src or dest
    """
    for pair in corres_groups:
        if pair[src_or_dest] == [id]:
            return pair[1 - src_or_dest][0]
    return None


def create_virtual_joints(corres_groups, src_g: List[int], dest_g: List[int],
                          src_nodes: List[AnyNode], dest_nodes: List[AnyNode]):
    """
    anything suffixed with root: must use frame_xlated
    """
    # two cases
    # group to null
    #   - group: copy over all attrs of the group
    #   - null: create 1 virtual joint using the group's head and bbox mapping
    # group to group

    # NOTE: only leaf compound nodes can be group to group / group to void 
    #       by construction

    if src_g == [-1]:
        dest_group_lead_ht = dest_nodes[dest_g[0]].frame_xlated[:2, :]

        dest_par = dest_nodes[dest_g[0]].parent
        src_corres_par_idx = look_up_node_corresp_in_groups(
            corres_groups, int(dest_par.id), 1)
        src_par = src_nodes[src_corres_par_idx]
        dest_to_src_par_bbox = bbox.get_bbox_mapping(
            src_par.bbox, dest_par.bbox)

        mapped_to_src_root = mapped_ht_of_node_matching_void(
            dest_to_src_par_bbox, dest_group_lead_ht,
            dest_par.bbox, src_par.bbox,
            dest_par.frame_xlated_w_to_l,
            src_par.frame_xlated_w_to_l)

        virtual_head = mapped_to_src_root[:1, :]   # (1, 3)
        virtual_ht = np.repeat(virtual_head, 2, axis=0)

        src_virtual_ht = [virtual_ht] * len(dest_g)
        src_bboxes = [None] * len(dest_g)
        src_subdiv_to_orig_xforms = [None] * len(dest_g)
        src_subdiv_indices = [None] * len(dest_g)
        dest_virtual_ht = []
        dest_bboxes = []
        dest_subdiv_to_orig_xforms = []
        dest_subdiv_indices = []
        for i in dest_g:
            dest_node = dest_nodes[i]
            dest_virtual_ht.append(dest_node.frame_xlated[:2, :])
            dest_bboxes.append(dest_node.bbox)
            dest_subdiv_to_orig_xforms.append(np.eye(4, 4, dtype=np.float32))
            dest_subdiv_indices.append(int(dest_node.id))

        return src_virtual_ht, src_bboxes, src_subdiv_to_orig_xforms, src_subdiv_indices,\
            dest_virtual_ht, dest_bboxes, dest_subdiv_to_orig_xforms, dest_subdiv_indices

    if dest_g == [-1]:
        src_group_lead_ht = src_nodes[src_g[0]].frame_xlated[:2, :]

        src_par = src_nodes[src_g[0]].parent
        dest_corres_par_idx = look_up_node_corresp_in_groups(
            corres_groups, int(src_par.id), 0)
        dest_par = dest_nodes[dest_corres_par_idx]
        src_to_dest_par_bbox = bbox.get_bbox_mapping(
            dest_par.bbox, src_par.bbox)

        mapped_to_dest_root = mapped_ht_of_node_matching_void(
            src_to_dest_par_bbox, src_group_lead_ht,
            src_par.bbox, dest_par.bbox,
            src_par.frame_xlated_w_to_l,
            dest_par.frame_xlated_w_to_l)

        virtual_head = mapped_to_dest_root[:1, :]   # (1, 3)
        virtual_ht = np.repeat(virtual_head, 2, axis=0)
        
        dest_virtual_ht = [virtual_ht] * len(src_g)
        dest_bboxes = [None] * len(src_g)
        dest_subdiv_to_orig_xforms = [None] * len(src_g)
        dest_subdiv_indices = [None] * len(src_g)
        src_virtual_ht = []
        src_bboxes = []
        src_subdiv_to_orig_xforms = []
        src_subdiv_indices = []
        for i in src_g:
            src_node = src_nodes[i]
            src_virtual_ht.append(src_node.frame_xlated[:2, :])
            src_bboxes.append(src_node.bbox)
            src_subdiv_to_orig_xforms.append(np.eye(4, 4, dtype=np.float32))
            src_subdiv_indices.append(int(src_node.id))

        return src_virtual_ht, src_bboxes, src_subdiv_to_orig_xforms, src_subdiv_indices,\
            dest_virtual_ht, dest_bboxes, dest_subdiv_to_orig_xforms, dest_subdiv_indices
    
    src_traced_len, src_intervals = get_traced_info(src_g, src_nodes)
    dest_traced_len, dest_intervals = get_traced_info(dest_g, dest_nodes)

    # for each bone, figure out hd gap and tl gap relative to traced_len itself
    src_g_nodes = [src_nodes[x] for x in src_g]
    dest_g_nodes = [dest_nodes[x] for x in dest_g]
    src_hd_tl_gap_perc = get_hd_tl_gap_perc(src_traced_len, src_g_nodes)
    dest_hd_tl_gap_perc = get_hd_tl_gap_perc(dest_traced_len, dest_g_nodes)

    src_onto_dest_perc = get_virt_relative(src_intervals, dest_intervals)
    dest_onto_src_perc = get_virt_relative(dest_intervals, src_intervals)

    dest_g_indices, dest_c_hd_per = gather_all_c_ind_hd_per(
        dest_intervals, src_onto_dest_perc)
    src_g_indices, src_c_hd_per = gather_all_c_ind_hd_per(
        src_intervals, dest_onto_src_perc)

    # print("src_g_indices: ", src_g_indices, " src_c_hd_per: ", src_c_hd_per)
    # print("dest_g_indices: ", dest_g_indices, " dest_c_hd_per: ", dest_c_hd_per)

    src_bboxes, src_subdiv_to_orig_xforms, src_subdiv_indices = \
        get_subdiv_bboxes_and_xforms(
            src_g_indices, src_c_hd_per, src_g, src_nodes,
            dest_hd_tl_gap_perc, dest_g_nodes)
    dest_bboxes, dest_subdiv_to_orig_xforms, dest_subdiv_indices =\
        get_subdiv_bboxes_and_xforms(
            dest_g_indices, dest_c_hd_per, dest_g, dest_nodes,
            src_hd_tl_gap_perc, src_g_nodes)

    dest_virtual_ht = get_virtual_ht_world_pos(
        dest_g_indices, dest_g, dest_nodes, dest_c_hd_per)
    src_virtual_ht = get_virtual_ht_world_pos(
        src_g_indices, src_g, src_nodes, src_c_hd_per)

    assert len(dest_virtual_ht) == len(src_virtual_ht)
    assert len(src_bboxes) == len(dest_bboxes)
    assert len(src_subdiv_to_orig_xforms) == len(dest_subdiv_to_orig_xforms)
    assert len(src_bboxes) == len(src_subdiv_to_orig_xforms)

    return src_virtual_ht, src_bboxes, src_subdiv_to_orig_xforms, src_subdiv_indices,\
        dest_virtual_ht, dest_bboxes, dest_subdiv_to_orig_xforms, dest_subdiv_indices


def get_hd_tl_gap_perc(traced_len, nodes: List[AnyNode]):
    hd_tl_gaps_pairs = []
    for n in nodes:
        # hd gap is just xneg
        # tl gap can be calculated by xpos - x position of brining tl to local
        hd_gap = n.bbox.scaffold.xneg / n.bone_len
        w_to_l = n.frame_xlated_w_to_l
        tl = n.frame_xlated[1]
        tl_in_local = kit.transform_one_point(tl, w_to_l)
        tl_gap = (n.bbox.scaffold.xpos - tl_in_local[0]) / n.bone_len
        hd_tl_gaps_pairs.append((hd_gap, tl_gap))
    return hd_tl_gaps_pairs


def get_subdiv_bboxes_and_xforms(g_indices, hd_traced_perc, group,
                                 all_nodes: List[AnyNode],
                                 hd_tl_gap_perc: List,
                                 the_other_g_nodes: List[AnyNode]):
    """
    returns [(bbox, bone_len, hd_traced_perc), ...]
    """
    g_indices_in_global = list(map(lambda x: group[x], g_indices))
    bone_to_hd_perc = {}
    for counter, idx in enumerate(g_indices_in_global):
        if idx not in bone_to_hd_perc.keys():
            bone_to_hd_perc[idx] = [hd_traced_perc[counter]]
        else:
            curr_lst = bone_to_hd_perc[idx]
            curr_lst.append(hd_traced_perc[counter])
            bone_to_hd_perc[idx] = curr_lst
    input_lst = []
    for k, v in bone_to_hd_perc.items():
        input_lst.append(
            (all_nodes[k].bbox, all_nodes[k].bone_len, v))

    all_bboxes = []
    all_xforms = []

    for inp in input_lst:
        # NOTE: this if statment is more specific, it's for 1 to many
        if len(inp[2]) > 1:
            # if we are splitting 1 bbox into multiple bboxes
            assert len(inp[2]) == len(hd_tl_gap_perc)
            bboxes, xforms = bbox.get_one_bone_subdiv_bboxes_and_xforms(
                inp[0], inp[1], inp[2], the_other_g_nodes)
        else:
            # keep the original bbox
            bboxes, xforms = bbox.get_one_bone_subdiv_bboxes_and_xforms(
                inp[0], inp[1], inp[2])
        all_bboxes += bboxes
        all_xforms += xforms

    return all_bboxes, all_xforms, g_indices_in_global


def get_virt_relative(intervals1, intervals2):
    """
    find which nodes from intervals2 that interval1's head + tail maps to
    where interval1 \in intervals1

    out: [(int, float)] * len(intervals1)
    int is the idx of the target node within the node group
    float is percentage along that bone
    """
    def head_is_in_which_interval(head, bounds):
        for idx, bound in enumerate(bounds):
            if head >= bound[0] and head < bound[1]:
                return idx

    def find_per(x, bounds):
        return (x - bounds[0]) / (bounds[1] - bounds[0])

    out = []
    for interval in intervals1:
        hd = interval[0]
        tl = interval[1]
        hd_idx = head_is_in_which_interval(hd, intervals2)
        hd_per = find_per(hd, intervals2[hd_idx])
        out.append((hd_idx, hd_per))
    return out


def gather_all_c_ind_hd_per(orig_intervals, mapped_info):
    """
    orig_intervals: [(float, float), ...], (perc, perc)
    mapped_info: [(int, float), ...], (node_idx, perc), all heads
    
    out: [int, ...], [float, ...]
    first: indices of nodes of the original group that the new hd mapped to
    second: percentage of heads mapped onto the original link of bones
    """
    # get all the heads
    orig_all_heads = [0] * len(orig_intervals)
    orig_all_indices = list(range(len(orig_intervals)))
    mapped_info = mapped_info[1:]
    if mapped_info == []:
        return orig_all_indices, orig_all_heads
    flat_mapped_info = list(map(list, zip(*mapped_info)))
    mapped_indices = flat_mapped_info[0]
    mapped_hd_per = flat_mapped_info[1]    
    all_indices = []
    all_hd_per = []
    for oi in orig_all_indices:
        all_hd_per.append(0)
        all_indices.append(oi)
        for j, mi in enumerate(mapped_indices):
            if mi == oi:
                all_hd_per.append(mapped_hd_per[j])
                all_indices.append(mi)
    return all_indices, all_hd_per


def get_virtual_ht_world_pos(g_hd_mapped_indices: List[int],
                             group: List[int],
                             all_nodes: List[AnyNode],
                             hd_perc: List[float]):
    """
    items from g_indices index into group
    items for group index into all_nodes
    """
    assert len(g_hd_mapped_indices) == len(hd_perc)
    all_hd = []
    for i in range(len(g_hd_mapped_indices)):
        idx_in_group = g_hd_mapped_indices[i]
        idx_in_all_nodes = group[idx_in_group]
        node = all_nodes[idx_in_all_nodes]
        perc = hd_perc[i]
        hd_pos = locate_pos_given_percent(node.frame_xlated[:2, :], perc)
        all_hd.append(hd_pos)
    num_hd = len(all_hd)
    all_pos = all_hd + [all_nodes[group[-1]].frame_xlated[1:2, :]]
    all_ht = []
    for i in range(num_hd):
        cc = np.concatenate([all_pos[i], all_pos[i+1]], axis=0)
        all_ht.append(cc)
    return all_ht


def locate_pos_given_percent(ht, perc):
    """
    out: (1, 3)
    """
    return (1 - perc) * ht[:1, :] + perc * ht[1:, :]


def write_comp_node_attrs(orig_c_nodes, comp_node: AnyNode, n_total_comp_nodes,
                          src_xlation, dest_xlation,
                          new_comp_nodes: List[AnyNode]):
    """
    out: number of new nodes created
    """
    # NOTE: when interpolating skeletons, only need to compute
    # local frames and world to local for intermediate skel
    # when t == 0 or t == 1, just use src and dest nodes information
    # this is to avoid confusion w group to void and group to group mappings
    # only leaf comp nodes can have group to group corresp
    if len(comp_node.children) != 0:
        # not a leaf node
        assert len(comp_node.all_src_virt_ht) == 1
        assert len(comp_node.all_dest_virt_ht) == 1
        assert len(comp_node.all_src_subdiv_bboxes) == 1
        assert len(comp_node.all_dest_virt_ht) == 1
        comp_node.src_virt_ht_xlated = comp_node.all_src_virt_ht[0]
        comp_node.dest_virt_ht_xlated = comp_node.all_dest_virt_ht[0]
        comp_node.src_virt_ht = comp_node.src_virt_ht_xlated - src_xlation
        comp_node.dest_virt_ht = comp_node.dest_virt_ht_xlated - dest_xlation
        # bbox should be in local
        comp_node.src_bbox = comp_node.all_src_subdiv_bboxes[0]
        comp_node.src_subdiv_to_orig_xform = comp_node.all_src_subdiv_to_orig[0]
        comp_node.src_subdiv_index = comp_node.all_src_subdiv_indices[0]
        comp_node.dest_bbox = comp_node.all_dest_subdiv_bboxes[0]
        comp_node.dest_subdiv_to_orig_xform = comp_node.all_dest_subdiv_to_orig[0]
        comp_node.dest_subdiv_index = comp_node.all_dest_subdiv_indices[0]
        orig_c_nodes[int(comp_node.id)].aug_ids = [int(comp_node.id)]
        return 0
    else:
        # leaf node
        # either 1 to 1, 1 to void, many to void, 1 to many, many to many
        assert len(comp_node.all_src_virt_ht) == len(comp_node.all_dest_virt_ht)
        assert len(comp_node.all_src_subdiv_bboxes) == len(comp_node.all_dest_subdiv_bboxes)
        assert len(comp_node.all_src_virt_ht) == len(comp_node.all_src_subdiv_bboxes)
        num_sub_nodes = len(comp_node.all_src_virt_ht) - 1
        # first, write attributes to the leaf node
        comp_node.src_virt_ht_xlated = comp_node.all_src_virt_ht[0]
        comp_node.dest_virt_ht_xlated = comp_node.all_dest_virt_ht[0]
        comp_node.src_virt_ht = comp_node.src_virt_ht_xlated - src_xlation
        comp_node.dest_virt_ht = comp_node.dest_virt_ht_xlated - dest_xlation
        # bbox should be in local
        comp_node.src_bbox = comp_node.all_src_subdiv_bboxes[0]
        comp_node.src_subdiv_to_orig_xform = comp_node.all_src_subdiv_to_orig[0]
        comp_node.src_subdiv_index = comp_node.all_src_subdiv_indices[0]
        comp_node.dest_bbox = comp_node.all_dest_subdiv_bboxes[0]
        comp_node.dest_subdiv_to_orig_xform = comp_node.all_dest_subdiv_to_orig[0]
        comp_node.dest_subdiv_index = comp_node.all_dest_subdiv_indices[0]
        aug_ids = [int(comp_node.id)]
        # second, make this leaf node into a branch node and 
        # create new bones linked up to this branch node
        for i in range(num_sub_nodes):
            new_id = n_total_comp_nodes + i
            aug_ids.append(new_id)
            new_node = AnyNode(id=str(new_id))
            src_ht = comp_node.all_src_virt_ht[1+i]
            dest_ht = comp_node.all_dest_virt_ht[1+i]
            new_node.src_virt_ht_xlated = src_ht
            new_node.dest_virt_ht_xlated = dest_ht
            new_node.src_virt_ht = new_node.src_virt_ht_xlated - src_xlation
            new_node.dest_virt_ht = new_node.dest_virt_ht_xlated - dest_xlation
            new_node.src_bbox = comp_node.all_src_subdiv_bboxes[1+i]
            new_node.dest_bbox = comp_node.all_dest_subdiv_bboxes[1+i]
            new_node.src_subdiv_to_orig_xform = comp_node.all_src_subdiv_to_orig[1+i]
            new_node.src_subdiv_index = comp_node.all_src_subdiv_indices[1+i]
            new_node.dest_subdiv_to_orig_xform = comp_node.all_dest_subdiv_to_orig[1+i]
            new_node.dest_subdiv_index = comp_node.all_dest_subdiv_indices[1+i]
            if i == 0:
                new_node.parent = comp_node
            else:
                new_node.parent = new_comp_nodes[-1]
            new_comp_nodes.append(new_node)
        orig_c_nodes[int(comp_node.id)].aug_ids = aug_ids
        return num_sub_nodes


def export_aug_skel(src_id, dest_id, interp_dir, num_interp,
                    ml=False, move_src=False, vis=True, pair_interp_dir=None,
                    stitch_imgs=False, out_dir=None):

    if pair_interp_dir is None:
        pair_interp_dir = os.path.join(interp_dir, 'interp_skel')
        misc.check_dir(pair_interp_dir)
    else:
        misc.check_dir(pair_interp_dir)

    src_info: Info = make_model_info(src_id)
    dest_info: Info = make_model_info(dest_id)

    src_package = segment.get_model_parts(
        src_info.path_to_obj, src_info.path_to_rig_info, src_info.path_to_skel,
        just_rig_info=True)
    dest_package = segment.get_model_parts(
        dest_info.path_to_obj, dest_info.path_to_rig_info, dest_info.path_to_skel,
        just_rig_info=True)

    src_joints_to_verts = src_package["joints_to_verts"]
    dest_joints_to_verts = dest_package["joints_to_verts"]
    src_joints_to_faces = src_package["joints_to_faces"]
    dest_joints_to_faces = dest_package["joints_to_faces"]
    src_coords = src_package["local_coords"]
    dest_coords = dest_package["local_coords"]
    src_verts = src_package["vertices"]
    dest_verts = dest_package["vertices"]
    src_faces = src_package["faces"]
    dest_faces = dest_package["faces"]
    src_w_to_l = src_package["world_to_local"]
    dest_w_to_l = dest_package["world_to_local"]

    # ----------------------- find corresp groups -----------------------

    # this is where src and dest nodes' frames get written
    corres_groups, trees, _, pairs = find_corresp_groups(
        src_info, dest_info,
        move_src=move_src,
        src_coords=src_coords, dest_coords=dest_coords, vis=vis)

    src_root_idx = trees["src_root_idx"]
    dest_root_idx = trees["dest_root_idx"]
    src_nodes = trees["src_nodes"]
    dest_nodes = trees["dest_nodes"]

    print("corresponding groups")
    print(corres_groups)

    # ----------------------- visualizing groups -----------------------

    if vis:
        # print("visualizing matched bones...")
        src_new_ht = kit.gather_heads_tails(
            src_nodes[src_root_idx], np.zeros((len(src_nodes), 2, 3)), 0)
        dest_new_ht = kit.gather_heads_tails(
            dest_nodes[dest_root_idx], np.zeros((len(dest_nodes), 2, 3)), 0)
        src_local_frames = kit.get_joints_local_coords(
            len(src_new_ht), src_new_ht)
        dest_local_frames = kit.get_joints_local_coords(
            len(dest_new_ht), dest_new_ht)

        info = True

        skel_rot = 30

        src_bone_path = os.path.join(
            pair_interp_dir, f'{src_info.model_id}_src.png')
        plt, colors = visualize.show_corresponding_bones(
            src_new_ht, corres_groups, 'src',
            text=info, rot=skel_rot)
        misc.save_fig(plt, '', src_bone_path, transparent=(not info))
        plt.close()

        dest_bone_path = os.path.join(
            pair_interp_dir, f'{dest_info.model_id}_dest.png')
        plt, _ = visualize.show_corresponding_bones(
            dest_new_ht, corres_groups, 'dest', custom_colors=colors,
            text=info, rot=skel_rot)
        misc.save_fig(plt, '', dest_bone_path, transparent=(not info))
        plt.close()

        src_patches_path = os.path.join(
            pair_interp_dir, f'{src_info.model_id}_src_patches_trimesh.png')
        visualize.save_corresponding_patches_trimesh(
            src_verts, src_faces, src_joints_to_verts, src_joints_to_faces, 'src',
            corres_groups, colors, src_patches_path)

        dest_patches_path = os.path.join(
            pair_interp_dir, f'{dest_info.model_id}_dest_patches_trimesh.png')
        visualize.save_corresponding_patches_trimesh(
            dest_verts, dest_faces, dest_joints_to_verts, dest_joints_to_faces, 'dest',
            corres_groups, colors, dest_patches_path)

        src_patches_path = os.path.join(
            pair_interp_dir, f'{src_info.model_id}_src_patches_matplot.png')
        plt, _ = visualize.show_corresponding_patches(
            src_verts, src_joints_to_faces, 'src',
            corres_groups, colors)
        misc.save_fig(plt, '', src_patches_path, transparent=True)
        plt.close()

        dest_patches_path = os.path.join(
            pair_interp_dir, f'{dest_info.model_id}_dest_patches_matplot.png')
        visualize.show_corresponding_patches(
            dest_verts, dest_joints_to_faces, 'dest',
            corres_groups, colors)
        misc.save_fig(plt, '', dest_patches_path, transparent=True)
        plt.close()

        if not stitch_imgs:
            plt = visualize.show_bones_and_local_coors(
                src_new_ht, src_local_frames)
            misc.save_fig(
                plt, '', os.path.join(
                    pair_interp_dir,
                    '{}_src_frames.png'.format(src_info.model_id)))
            plt.close()

            plt = visualize.show_bones_and_local_coors(
                dest_new_ht, dest_local_frames)
            misc.save_fig(
                plt, '', os.path.join(
                    pair_interp_dir,
                    '{}_dest_frames.png'.format(dest_info.model_id)))
            plt.close()
        
    if stitch_imgs:
        assert vis == True
        all_imgs = []
        all_imgs.append(src_bone_path)
        all_imgs.append(src_patches_path)
        all_imgs.append(dest_bone_path)
        all_imgs.append(dest_patches_path)
        # all_imgs.append(interp_bones_img_paths[1])
        # all_imgs.extend(interp_frames_img_paths)
        images = [Image.open(x) for x in all_imgs]
        widths, heights = zip(*(i.size for i in images))
        adj = 0
        total_width = sum(widths) - 3 * adj
        max_height = max(heights)
        new_im = Image.new('RGBA', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0] - adj
        
        new_im.save(os.path.join(out_dir, '{}_{}.png'.format(src_id, dest_id)))
        return corres_groups

    # ----------------------- get augmented skeleton -----------------------

    src_local_verts = kit.get_local_verts(
        src_verts, src_joints_to_verts, src_w_to_l)
    dest_local_verts = kit.get_local_verts(
        dest_verts, dest_joints_to_verts, dest_w_to_l)

    # print("writing src bboxes...")
    write_bboxes(
        src_nodes[src_root_idx], src_local_verts, src_w_to_l)
    # print("writing dest bboxes...")        
    write_bboxes(
        dest_nodes[dest_root_idx], dest_local_verts, dest_w_to_l)

    aug_skel_nodes, groups, comp_nodes = get_aug_skel(
        src_root_idx, src_nodes,
        dest_root_idx, dest_nodes,
        corres_groups)

    # ----------------------- interpolate aug skel -----------------------

    all_new_ht, all_new_local_coords, all_new_w_to_l =\
        interp_aug_skel(aug_skel_nodes, num_interp, src_id, dest_id)

    # ----------------------- visualize aug skel -----------------------

    if vis:
        # for rot in [0, 45, 90]:
        for rot in [45]:
            print("visualizing interpolation at {} degrees".format(rot))
            rot_dir = os.path.join(pair_interp_dir, 'rot{}'.format(rot))
            misc.check_dir(rot_dir)

            interp_bones_img_paths = []
            pbar = tqdm(total = len(all_new_ht))
            for i, ht_c in enumerate(zip(all_new_ht, all_new_local_coords)):
                plt = visualize.show_bones_and_local_coors(ht_c[0], ht_c[1])
                if colors is not None:
                    plt, _ = visualize.show_corresponding_bones(
                        ht_c[0], groups, 'src',
                        custom_colors=colors, rot=rot,
                        text=info)
                else:
                    plt = visualize.show_bones(ht_c[0], text=False)
                misc.save_fig(
                    plt, '',
                    os.path.join(rot_dir, 'interp_ht_{}.png'.format(i)),
                    transparent=(not info))
                plt.close()
                interp_bones_img_paths.append(
                    os.path.join(rot_dir, 'interp_ht_{}.png'.format(i)))
                pbar.update()
            pbar.close()

            def condition(name):
                return name[:9] == 'interp_ht'

            interp_img_filename = 'ht_interp.gif'
            misc.save_images_as_gif(rot_dir, condition, interp_img_filename)

            interp_frames_img_paths = []
            pbar = tqdm(total = len(all_new_ht))
            for i, ht_c in enumerate(zip(all_new_ht, all_new_local_coords)):
                plt = visualize.show_bones_and_local_coors(ht_c[0], ht_c[1])
                misc.save_fig(
                    plt, '',
                    os.path.join(rot_dir, 'interp_frames_{}.png'.format(i)),
                    transparent=False)
                plt.close()
                interp_frames_img_paths.append(
                    os.path.join(rot_dir, 'interp_frames_{}.png'.format(i)))
                pbar.update()
            pbar.close()

            def condition(name):
                return name[:9] == 'interp_fr'

            interp_img_filename = 'frames_interp.gif'
            misc.save_images_as_gif(rot_dir, condition, interp_img_filename)


    return InterpInfo(
        src_id=src_id,
        dest_id=dest_id,
        src_info=src_info,
        dest_info=dest_info,
        src_package=src_package,
        dest_package=dest_package,
        src_nodes=src_nodes,
        dest_nodes=dest_nodes,
        pairs=pairs,
        corres_groups=corres_groups,
        comp_nodes=comp_nodes,
        aug_nodes=aug_skel_nodes,
        aug_groups=groups,
        colors=colors if vis else None,
        all_interp_ht=all_new_ht,
        all_interp_frames=all_new_local_coords,
        all_interp_w_to_l=all_new_w_to_l)
