import copy
import numpy as np
from data_prep import segment
from utils import kit, misc, visualize 
from anytree import AnyNode, Walker
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from mixer_interp import Info


# we kinda need jumps to root
class FindSteps:
    def __init__(self):
        """Given a nested tuple, find the number of elements.
        """
        self.count = 0

    def get_num_steps(self):
        return self.count

    def count_path_steps(self, path):
        for p in path:
            if isinstance(p, tuple):
                self.count_path_steps(p)
            else:
                self.count += 1


def jumps_to_root(joint_idx, root_joint_idx, nodes):
    root = nodes[root_joint_idx]
    for count, node in enumerate(nodes):
        if node.id == str(joint_idx):
            joint = nodes[count]
    w = Walker()
    path = w.walk(root, joint)
    fs = FindSteps()
    fs.count_path_steps(path)
    return fs.get_num_steps() - 1


class Corresp:
    def __init__(self, src_root_idx, src_nodes, dest_root_idx, dest_nodes):
        self.src_root: AnyNode = src_nodes[src_root_idx]
        self.dest_root: AnyNode = dest_nodes[dest_root_idx]
        self.num_src = len(src_nodes)
        self.num_dest = len(dest_nodes)
        # src node to dest node cost
        self.A = np.zeros((self.num_src, self.num_dest), dtype=np.float32)
        # visited
        self.A_v = np.zeros((self.num_src, self.num_dest), dtype=np.float32)
        # src node to null
        self.B = np.zeros((self.num_src, self.num_src), dtype=np.float32)
        # visited
        self.B_v = np.zeros((self.num_src, self.num_src), dtype=np.float32)
        # dest node to null
        self.C = np.zeros((self.num_dest, self.num_dest), dtype=np.float32)
        # visited
        self.C_v = np.zeros((self.num_dest, self.num_dest), dtype=np.float32)

        self.min_num_bones = min(self.num_src, self.num_dest)
        self.num_bone_diff = abs(self.num_src - self.num_dest)

        if self.min_num_bones >= 30:
            self.alpha = 5
        else:
            self.alpha =\
                -0.05 * self.min_num_bones +\
                0.15 * self.num_bone_diff +\
                1.5
            self.alpha = round(self.alpha)

        self.use_add_branch_to_branch = True
        

    def get_corresp(self):
        self.compute_cost(self.src_root, self.dest_root)
        M = np.zeros((self.num_src + self.num_dest,
                      self.num_src + self.num_dest), dtype=np.float32)
        M[:self.num_src, :self.num_dest] = self.A
        M[self.num_src:, :self.num_dest] = self.C
        M[:self.num_src, self.num_dest:] = self.B

        row_ind, col_ind = linear_sum_assignment(M)
        min_cost = M[row_ind, col_ind].sum()
        return row_ind, col_ind, min_cost

    def compute_cost(self, src_node: AnyNode, dest_node: AnyNode):
        # all matrices are initialized as zeros
        if src_node != None:
            src_id = int(src_node.id)
        if dest_node != None:
            dest_id = int(dest_node.id)

        if src_node == None and dest_node == None:
            return

        # if src is null, dest is not null
        #   leaf to void cost if dest is leaf
        #   branch to void cost if dest is branch
        if src_node == None and dest_node != None:
            # query if already visited
            if self.C_v[dest_id, dest_id] == 1:
                return self.C[dest_id, dest_id]

            if is_leaf(dest_node):
                leaf_to_void = self.leaf_to_void_cost(dest_node)
                self.C[:, dest_id] = leaf_to_void
                self.C_v[dest_id, dest_id] = 1
                return leaf_to_void
            else:
                branch_to_void = 0
                for child in dest_node.children:
                    branch_to_void += self.compute_cost(None, child)
                self.C[:, dest_id] = branch_to_void
                self.C_v[dest_id, dest_id] = 1
                return branch_to_void

        # if src is not null, dest is null
        #   leaf to void cost if src is leaf
        #   brach to void cost if src is branch
        if src_node != None and dest_node == None:
            # query if already visited
            if self.B_v[src_id, src_id] == 1:
                return self.B[src_id, src_id]

            if is_leaf(src_node):
                leaf_to_void = self.leaf_to_void_cost(src_node)
                self.B[src_id, :] = leaf_to_void
                self.B_v[src_id, src_id] = 1
                return leaf_to_void
            else:
                branch_to_void = 0
                for child in src_node.children:
                    branch_to_void += self.compute_cost(child, None)
                self.B[src_id, :] = branch_to_void
                self.B_v[src_id, src_id] = 1
                return branch_to_void

        # if src is not null, dest is not null

        # query if already visited
        if self.A_v[src_id, dest_id] == 1:
            return self.A[src_id, dest_id]

        # if src is leaf, dest is leaf
        if is_leaf(src_node) and is_leaf(dest_node):
            leaf_to_leaf = leaf_to_leaf_cost(src_node, dest_node)
            self.A[src_id, dest_id] = leaf_to_leaf
            self.A_v[src_id, dest_id] = 1
            return leaf_to_leaf

        # if src is leaf, dest is not leaf (branch vertex)
        if is_leaf(src_node) and not is_leaf(dest_node):
            # construct two sets
            src_list = [src_node]
            dest_list = dest_node.children
            branch_to_leaf = self.hungary_cost(src_list, dest_list)
            self.A[src_id, dest_id] = branch_to_leaf
            self.A_v[src_id, dest_id] = 1
            return branch_to_leaf

        # if src is not leaf (branch vertex), dest is leaf
        if not is_leaf(src_node) and is_leaf(dest_node):
            # construct two sets
            src_list = src_node.children
            dest_list = [dest_node]
            branch_to_leaf = self.hungary_cost(src_list, dest_list)
            self.A[src_id, dest_id] = branch_to_leaf
            self.A_v[src_id, dest_id] = 1
            return branch_to_leaf

        # if src is not leaf (branch vertex), dest is not leaf (branch vertex)
        if not is_leaf(src_node) and not is_leaf(dest_node):
            # three possibilities, take the minimum 
            # a: {(children of src, children of dest)}
            # b: {(children of src, dest)}
            # c: {(src            , children of dest)}
            src_list = src_node.children
            dest_list = dest_node.children
            cost_a = self.hungary_cost(src_list, dest_list)
            cost_b = self.hungary_cost(src_list, [dest_node])
            cost_c = self.hungary_cost([src_node], dest_list)
            branch_to_branch = np.min([cost_a, cost_b, cost_c])
     
            # additional cost for branch to branch
            if self.use_add_branch_to_branch:
                branch_to_branch +=\
                    1 * (spatial.distance.cosine(
                        src_node.dir_to_root, dest_node.dir_to_root) - 1)

            self.A[src_id, dest_id] = branch_to_branch
            self.A_v[src_id, dest_id] = 1
            return branch_to_branch
        
        return -1

    def hungary_cost(self, set1: list, set2: list, print_M=False):
        """compute the sum of the pairwise costs
        """
        n_set1 = len(set1)
        n_set2 = len(set2)
        h_A = np.zeros((n_set1, n_set2), dtype=np.float32)
        h_B = np.zeros((n_set1, n_set1), dtype=np.float32)
        h_C = np.zeros((n_set2, n_set2), dtype=np.float32)

        # compute node to node
        for i in range(n_set1):
            n_i = set1[i]
            for j in range(n_set2):
                n_j = set2[j]
                h_A[i, j] = self.compute_cost(n_i, n_j)

        # compute None to node
        for j in range(n_set2):
            n_j = set2[j]
            h_C[:, j] = self.compute_cost(None, n_j)
        
        # compute node to None
        for i in range(n_set1):
            n_i = set1[i]
            h_B[i, :] = self.compute_cost(n_i, None)

        M = np.zeros((n_set1 + n_set2, n_set1 + n_set2), dtype=np.float32)
        M[:n_set1, :n_set2] = h_A
        M[n_set1:, :n_set2] = h_C
        M[:n_set1, n_set2:] = h_B
        row_ind, col_ind = linear_sum_assignment(M)
        cost = M[row_ind, col_ind].sum()
        return cost
    
    def leaf_to_void_cost(self, node: AnyNode):
        return node.norm_bone_len + self.alpha * node.norm_bone_len


def octant_one_axis(node1: AnyNode, node2: AnyNode, axis):
    # different signs
    condition1 = node1.rel_pos[axis] * node2.rel_pos[axis] < 0
    # one is close to zero but the other is not
    condition2 = misc.x_within_n(node1.rel_pos[axis], 0) != \
                 misc.x_within_n(node2.rel_pos[axis], 0)
    if condition1 or condition2:
        return 1
    return 0


def leaf_to_leaf_cost(node1: AnyNode, node2: AnyNode):
    bone_len_diff = abs(node1.norm_bone_len - node2.norm_bone_len)
    dir_vec_diff = spatial.distance.cosine(
        node1.dir_in_parent, node2.dir_in_parent)   # distance
    pos_diff = np.linalg.norm(node1.rel_pos - node2.rel_pos)
    jumps_diff = abs(node1.jumps_to_root - node2.jumps_to_root)
    # octant
    x = octant_one_axis(node1, node2, 0)
    y = octant_one_axis(node1, node2, 1)
    z = octant_one_axis(node1, node2, 2)
    bl = 1
    dv = 1
    po = 1
    jtr = 1
    oct = 1

    # all heu
    cost = bl * bone_len_diff +\
        po * pos_diff +\
        dv * dir_vec_diff +\
        jtr * jumps_diff +\
        oct * (x + y + z)

    return cost


def is_leaf(node: AnyNode):
    return len(node.children) == 0


def write_attributes(node: AnyNode,
                     max_bone_len,
                     xlation=np.array([0, 0, 0]),
                     root_idx=-1,
                     nodes=[]):
    """Use existing frame attrib to write bone_len and dir_in_parent
    """
    if node == None:
        return

    if node.parent is None:
        node.xlation = xlation

    assert node.frame is not None

    node.frame_w_to_l = kit.get_one_world_to_local(node.frame)

    node.frame_xlated = copy.deepcopy(node.frame)
    node.frame_xlated[0, :] += xlation
    node.frame_xlated[1, :] += xlation 

    node.frame_xlated_w_to_l = kit.get_one_world_to_local(node.frame_xlated)

    head = node.frame_xlated[0, :]
    tail = node.frame_xlated[1, :]
    # normalizaed bone length
    node.bone_len = np.linalg.norm(head - tail)
    node.norm_bone_len = node.bone_len / max_bone_len
    node.pos = head
    # node.pos = (head + tail) / 2
    node.rel_pos = node.pos - nodes[root_idx].pos
    pos_diff = node.pos - nodes[root_idx].pos
    node.dist_to_root = np.linalg.norm(pos_diff)
    node.dir_to_root = kit.normalize(pos_diff) if node.dist_to_root != 0\
        else np.array([0, 0, 0], dtype=np.float32)
    node.jumps_to_root = jumps_to_root(int(node.id), root_idx, nodes)
    node.num_children = len(node.children)

    if node.parent is None:
        dir_in_parent = np.array([0, 0, 0], dtype=np.float32)
    else:
        xform_parent = kit.get_one_world_to_local(node.parent.frame_xlated)
        head_in_parent = kit.transform_one_point(head, xform_parent)
        tail_in_parent = kit.transform_one_point(tail, xform_parent)
        dir_in_parent = kit.normalize(tail_in_parent - head_in_parent)
    node.dir_in_parent = dir_in_parent
    
    for child in node.children:
        write_attributes(child, max_bone_len, xlation, root_idx, nodes)


def get_root_idx_and_nodes(info: Info, local_coords=None):
    obj_path = info.path_to_obj
    rig_path = info.path_to_rig_info
    skel_path = info.path_to_skel

    rig_info = segment.get_model_parts(
        obj_path, rig_path, skel_path, just_rig_info=True)
    num_src = len(rig_info["local_coords"])
    root_idx, nodes, _, _, _ = segment.get_skel_tree(
        num_src, rig_path, local_coords=local_coords)

    return root_idx, nodes


def get_corresp_list(src_info, dest_info, move_src=True,
                     src_coords=None, dest_coords=None):
    vis_ht = False

    src_root_idx, src_nodes = get_root_idx_and_nodes(src_info, src_coords)
    dest_root_idx, dest_nodes = get_root_idx_and_nodes(dest_info, dest_coords)

    src_root = src_nodes[src_root_idx]
    dest_root = dest_nodes[dest_root_idx]
    src_root_head = src_root.frame[0, :]
    dest_root_head = dest_root.frame[0, :]
    src_to_dest_xlation = dest_root_head - src_root_head

    num_src = len(src_nodes)
    num_dest = len(dest_nodes)

    src_ht = kit.gather_heads_tails(
        src_root, np.zeros((num_src, 2, 3), dtype=np.float32), 0)
    src_max_bone_len = np.max(np.linalg.norm(
        src_ht[:, 0, :] - src_ht[:, 1, :], axis=-1))
    dest_ht = kit.gather_heads_tails(
        dest_root, np.zeros((num_dest, 2, 3), dtype=np.float32), 0)
    dest_max_bone_len = np.max(np.linalg.norm(
        dest_ht[:, 0, :] - dest_ht[:, 1, :], axis=-1))

    # call write attributes
    write_attributes(src_nodes[src_root_idx],
                     src_max_bone_len,
                     xlation=int(move_src) * src_to_dest_xlation,
                     root_idx=src_root_idx,
                     nodes=src_nodes)
    write_attributes(dest_nodes[dest_root_idx],
                     dest_max_bone_len,
                     xlation=int(not move_src) * -src_to_dest_xlation,
                     root_idx=dest_root_idx,
                     nodes=dest_nodes)

    if vis_ht:
        src_ht = kit.gather_heads_tails(
            src_root, np.zeros((num_src, 2, 3), dtype=np.float32), 1)
        plt = visualize.show_bones(src_ht, text=True)
        misc.save_fig(plt, '', 'tmp/ht_src.png')
        plt.close()
        dest_ht = kit.gather_heads_tails(
            dest_root, np.zeros((num_dest, 2, 3), dtype=np.float32), 1)
        plt = visualize.show_bones(dest_ht, text=True)
        misc.save_fig(plt, '', 'tmp/ht_dest.png')
        plt.close()


    corresp = Corresp(src_root_idx, src_nodes, dest_root_idx, dest_nodes)
    row_ind, col_ind, cost = corresp.get_corresp()

    row_ind = row_ind.tolist()
    col_ind = col_ind.tolist()
    row_ind = list(map(lambda x: -1 if x >= num_src else x, row_ind))
    col_ind = list(map(lambda x: -1 if x >= num_dest else x, col_ind))

    pairs = list(zip(row_ind, col_ind))
    pairs = list(filter(lambda x: x[0] != -1 or x[1] != -1, pairs))

    print("--------")
    print(pairs)
    print("--------")

    trees = {
        'src_root_idx': src_root_idx,
        'dest_root_idx': dest_root_idx,
        'src_nodes': src_nodes,
        'dest_nodes': dest_nodes
    }

    return pairs, cost, trees

