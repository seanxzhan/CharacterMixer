import numpy as np
from typing import List
from anytree import AnyNode
from corresp import find_corresp
from scipy import spatial, optimize


def get_all_descendents(node: AnyNode):
    if len(node.children) == 0:
        node.all_desc = []
        return []

    all_desc = []
    for child in node.children:
        all_desc.append(int(child.id))
        all_desc.extend(get_all_descendents(child))

    node.all_desc = all_desc
    return all_desc


def find_corresp_groups(src_info, dest_info, move_src=False,
                        src_coords=None, dest_coords=None,
                        vis=True, vis_dir=None):
    pairs, _, trees = find_corresp.get_corresp_list(
        src_info, dest_info, move_src=move_src,
        src_coords=src_coords, dest_coords=dest_coords)

    src_root_idx = trees["src_root_idx"]
    dest_root_idx = trees["dest_root_idx"]
    src_nodes = trees["src_nodes"]
    dest_nodes = trees["dest_nodes"]
    num_src = len(src_nodes)
    num_dest = len(dest_nodes)

    # logic: 1 to many correspondence happens when destination nodes
    #        are mapped to void (nothing in src matches)
    # rule
    # if any src maps to a dest branch vertex + if branch vertex's kids all map to void
    # then that src maps to the entire subtree with the branch vertex as root
    get_all_descendents(src_nodes[src_root_idx])
    get_all_descendents(dest_nodes[dest_root_idx])
    src_node_to_desc = {}
    for n in src_nodes:
        src_node_to_desc[int(n.id)] = n.all_desc
    dest_node_to_desc = {}
    for n in dest_nodes:
        dest_node_to_desc[int(n.id)] = n.all_desc
    
    which_src_maps_to_void = []
    which_dest_maps_to_void = []
    for pair in pairs:
        if pair[0] == -1:
            which_dest_maps_to_void.append(pair[1])
        if pair[1] == -1:
            which_src_maps_to_void.append(pair[0])

    def subdiv_group_pair(src_g, dest_g):
        # split groups even further, make sure there are no many to many corresp
        # worst case, 1 to many corresp
        if len(src_g) >= 2 and len(dest_g) >= 2:
            how_many_1_to_1 = min(len(src_g), len(dest_g)) - 1
            out = []
            for i in range(how_many_1_to_1):
                out.append(([src_g[i]], [dest_g[i]]))
            out.append((src_g[how_many_1_to_1:], dest_g[how_many_1_to_1:]))
            return out, 1
        else:
            return [(src_g, dest_g)], 0

    groups = []     # src to dest
    visited_src = []
    visited_dest = []
    for pair in pairs:
        if pair[0] in visited_src:
            continue
        if pair[1] in visited_dest:
            continue
        lst_of_src = []
        lst_of_dest = []
        if pair[0] == 0 and pair[1] == 0:
            groups.append(([0], [0]))

            src_branches = find_single_branches_for_node(src_nodes[pair[0]])
            dest_branches = find_single_branches_for_node(dest_nodes[pair[1]])
            src_void_branches = desc_which_branch_maps_to_void(
                src_branches, which_src_maps_to_void)
            dest_void_branches = desc_which_branch_maps_to_void(
                dest_branches, which_dest_maps_to_void)

            if len(src_void_branches) >= 1 and len(dest_void_branches) >= 1:
                srcs, dests = compare_branch_parent_octant(
                    src_void_branches, dest_void_branches, src_nodes, dest_nodes)
                assert len(srcs) == len(dests)
                for i in range(len(srcs)):
                    src_b = src_void_branches[srcs[i]]
                    dest_b = dest_void_branches[dests[i]]
                    for n in src_b:
                        visited_src.append(n)
                    for n in dest_b:
                        visited_dest.append(n)
                    groups.append((src_b, dest_b))

        if pair[1] == -1 and pair[0] not in visited_src:
            # if src is not void and dest is void,
            # the descendants of src are grouped together to map to void
            lst_of_src.append(pair[0])
            if int(src_nodes[pair[0]].parent.id) in which_src_maps_to_void:
                continue
            desc = src_node_to_desc[pair[0]]
            for d in desc:
                if d in which_src_maps_to_void:
                    visited_src.append(d)
                    lst_of_src.append(d)
            lst_of_dest = [-1]
            groups.append((lst_of_src, lst_of_dest))
            continue
        if pair[0] == -1 and pair[1] not in visited_dest:
            # if dest is not void and src is void,
            # the descendants of dest are grouped together to map to void
            lst_of_dest.append(pair[1])
            if int(dest_nodes[pair[1]].parent.id) in which_dest_maps_to_void:
                continue
            desc = dest_node_to_desc[pair[1]]
            for d in desc:
                if d in which_dest_maps_to_void:
                    visited_dest.append(d)
                    lst_of_dest.append(d)
            lst_of_src = [-1]
            groups.append((lst_of_src, lst_of_dest))
            continue
        if pair[0] != 0 and pair[1] != 0:
            # if both are not void
            lst_of_src.append(pair[0])
            lst_of_dest.append(pair[1])

            src_descs = src_node_to_desc[pair[0]]
            dest_descs = dest_node_to_desc[pair[1]]

            if desc_all_map_to_void(src_descs, which_src_maps_to_void):
                for d in src_descs:
                    visited_src.append(d)
                    lst_of_src.append(d)
            if desc_all_map_to_void(dest_descs, which_dest_maps_to_void):
                for d in dest_descs:
                    visited_dest.append(d)
                    lst_of_dest.append(d)

            out, flag = subdiv_group_pair(lst_of_src, lst_of_dest)
            for o in out:
                groups.append(o)

            if flag == 1:
                continue

            src_branches = find_single_branches_for_node(src_nodes[pair[0]])
            dest_branches = find_single_branches_for_node(dest_nodes[pair[1]])
            src_void_branches = desc_which_branch_maps_to_void(
                src_branches, which_src_maps_to_void)
            dest_void_branches = desc_which_branch_maps_to_void(
                dest_branches, which_dest_maps_to_void)

            if len(src_void_branches) >= 1 and len(dest_void_branches) >= 1:
                srcs, dests = compare_branch_parent_octant(
                    src_void_branches, dest_void_branches, src_nodes, dest_nodes)
                assert len(srcs) == len(dests)
                for i in range(len(srcs)):
                    src_b = src_void_branches[srcs[i]]
                    dest_b = dest_void_branches[dests[i]]
                    for n in src_b:
                        visited_src.append(n)
                    for n in dest_b:
                        visited_dest.append(n)
                    groups.append((src_b, dest_b))

    if not vis:
        return groups, trees, None, pairs
    else:
        return groups, trees, None, pairs
    

def check_if_group_valid(group: List[int], nodes: List[AnyNode]):
    for i, idx in enumerate(group):
        if i != 0 and group != [-1]:
            assert nodes[idx] in nodes[group[i-1]].children


def desc_all_map_to_void(desc, descs_to_void):
    return set(desc).issubset(set(descs_to_void))


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def find_single_branches_for_node(node: AnyNode):
    """ given a node, find all the branches where each node on the branch only
    has one or no children

    OI: 1
        RI: 2 | 3 | 4 | 5
        RO: [2, 7, 11] | [3, 8, 12] | [...] | [...]
    OO = [RO]
    """
    branches = []
    for child in node.children:
        pot_branch = []
        if len(child.children) == 0 or len(child.children) == 1:
            pot_branch.append([int(child.id)] + find_single_branches_for_node(child))
        else:
            continue
        pot_branch = list(flatten(pot_branch))
        branches.append(pot_branch)
    return branches


def desc_which_branch_maps_to_void(branches, nodes_to_void):
    branches_mapped_to_single = []
    for b in branches:
        if desc_all_map_to_void(b, nodes_to_void):
            branches_mapped_to_single.append(b)
    return branches_mapped_to_single


def compare_branch_parent_octant(src_branches, dest_branches,
                                 src_nodes, dest_nodes):
    src_ances = [x[0] for x in src_branches]
    dest_ances = [x[0] for x in dest_branches]
    # print("src ancestors: ", [src_nodes[x].id for x in src_ances])
    # print("dest ancestors: ", [dest_nodes[x].id for x in dest_ances])
    cost_mat = np.zeros((len(src_ances), len(dest_ances)))
    for si, src in enumerate(src_ances):
        for di, dest in enumerate(dest_ances):
            s = src_nodes[src]
            d = dest_nodes[dest]
            x = find_corresp.octant_one_axis(s, d, 0)
            y = find_corresp.octant_one_axis(s, d, 1)
            z = find_corresp.octant_one_axis(s, d, 2)
            dist = spatial.distance.cosine(s.dir_to_root, d.dir_to_root) - 1
            cost_mat[si, di] = x + y + z + dist
    # print(cost_mat)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_mat)
    # print(row_ind, col_ind)
    return row_ind, col_ind


def desc_all_match(src_desc, dest_desc, pairs):
    # descendents all match if sets of src and dest descs are the same
    lists = list(zip(*pairs))
    src_lst = lists[0]
    dest_lst = lists[1]
    matching_src_indices = []
    for d in src_desc:
        matching_src_indices.append(src_lst.index(d))
    matched_nodes_src_to_dest = [dest_lst[i] for i in matching_src_indices]
    matching_dest_indices = []
    for d in dest_desc:
        matching_dest_indices.append(dest_lst.index(d))
    matched_nodes_dest_to_src = [src_lst[i] for i in matching_dest_indices]
    return set(src_desc) == set(matched_nodes_dest_to_src) and\
        set(dest_desc) == set(matched_nodes_src_to_dest)
