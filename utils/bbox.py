import copy
import numpy as np
from utils import kit
from typing import List
from anytree import AnyNode
from dataclasses import dataclass


EPSILON = 0.015


class Scaffold():
    def __init__(self,
        xneg: np.float32,
        xpos: np.float32,
        yneg: np.float32,
        ypos: np.float32,
        zneg: np.float32,
        zpos: np.float32,
        center: np.ndarray):
        self.xneg = xneg
        self.xpos = xpos
        self.yneg = yneg
        self.ypos = ypos
        self.zneg = zneg
        self.zpos = zpos
        self.center_in_local = center

    def tolist(self) -> np.ndarray:
        return np.array([
            self.xneg, self.xpos,
            self.yneg, self.ypos,
            self.zneg, self.zpos])

    def get_min_max(self, origin_in_local, scale=1):
        oil = origin_in_local.tolist()
        """oil: origin in local
        <minx> <miny> <minz> <maxx> <maxy> <maxz>
        """
        return [
            oil[0] - self.xneg * scale,
            oil[1] - self.yneg * scale,
            oil[2] - self.zneg * scale,
            oil[0] + self.xpos * scale,
            oil[1] + self.ypos * scale,
            oil[2] + self.zpos * scale
        ]


@dataclass
class Bbox:
    # origin coordinates are in the part's local space
    origin_in_local: np.ndarray
    scaffold: Scaffold
    local_to_bbox: np.ndarray


class Mappings:
    # mapping from 1 bounding box to the other
    # contains a mapping for each octant, each mapping is 4x4
    def __init__(self,
        ppp: np.ndarray,
        npp: np.ndarray,
        pnp: np.ndarray,
        nnp: np.ndarray,
        ppn: np.ndarray,
        npn: np.ndarray,
        pnn: np.ndarray,
        nnn: np.ndarray):
        self.ppp = ppp
        self.npp = npp
        self.pnp = pnp
        self.nnp = nnp
        self.ppn = ppn
        self.npn = npn
        self.pnn = pnn
        self.nnn = nnn

    def send(self, point) -> np.ndarray:
        if point[0] >= 0 and point[1] >= 0 and point[2] >= 0:
            return kit.transform_one_point(point, self.ppp)
        if point[0] <= 0 and point[1] >= 0 and point[2] >= 0:
            return kit.transform_one_point(point, self.npp)
        if point[0] >= 0 and point[1] <= 0 and point[2] >= 0:
            return kit.transform_one_point(point, self.pnp)
        if point[0] <= 0 and point[1] <= 0 and point[2] >= 0:
            return kit.transform_one_point(point, self.nnp)
        if point[0] >= 0 and point[1] >= 0 and point[2] <= 0:
            return kit.transform_one_point(point, self.ppn)
        if point[0] <= 0 and point[1] >= 0 and point[2] <= 0:
            return kit.transform_one_point(point, self.npn)
        if point[0] >= 0 and point[1] <= 0 and point[2] <= 0:
            return kit.transform_one_point(point, self.pnn)
        if point[0] <= 0 and point[1] <= 0 and point[2] <= 0:
            return kit.transform_one_point(point, self.nnn)
    
    def quad_to_index(self, x, y, z):
        """x, y, z \in {0, 1}, 
        0 means negative, 1 means positive
        """
        return x*1 + y*2 + z*4

    def send_all(self, points) -> np.ndarray:
        """
        points: (n, 3)

        out: (n, 3)
        """
        # partitions points into octants
        octants = (points > 0) @ 2**np.arange(3)
        ppp_indices = np.where(octants == self.quad_to_index(1, 1, 1))[0]
        npp_indices = np.where(octants == self.quad_to_index(0, 1, 1))[0]
        pnp_indices = np.where(octants == self.quad_to_index(1, 0, 1))[0]
        nnp_indices = np.where(octants == self.quad_to_index(0, 0, 1))[0]
        ppn_indices = np.where(octants == self.quad_to_index(1, 1, 0))[0]
        npn_indices = np.where(octants == self.quad_to_index(0, 1, 0))[0]
        pnn_indices = np.where(octants == self.quad_to_index(1, 0, 0))[0]
        nnn_indices = np.where(octants == self.quad_to_index(0, 0, 0))[0]
        result = np.zeros_like(points, dtype=np.float32)
        # TODO: this can be faster with a batch dimension
        result[ppp_indices] = kit.transform_points(points[ppp_indices], self.ppp)
        result[npp_indices] = kit.transform_points(points[npp_indices], self.npp)
        result[pnp_indices] = kit.transform_points(points[pnp_indices], self.pnp)
        result[nnp_indices] = kit.transform_points(points[nnp_indices], self.nnp)
        result[ppn_indices] = kit.transform_points(points[ppn_indices], self.ppn)
        result[npn_indices] = kit.transform_points(points[npn_indices], self.npn)
        result[pnn_indices] = kit.transform_points(points[pnn_indices], self.pnn)
        result[nnn_indices] = kit.transform_points(points[nnn_indices], self.nnn)
        return result


# given a local part, find the bounding box and transformation
# the origin in local is the corresponding joint position in local
def get_bbox(vertices, joint_in_world, part_world_to_local):
    origin_in_local = kit.transform_one_point(
        joint_in_world, part_world_to_local)

    # vertices shape should be n x 3
    xmin = np.min(vertices[:, 0])
    xmax = np.max(vertices[:, 0])
    ymin = np.min(vertices[:, 1])
    ymax = np.max(vertices[:, 1])
    zmin = np.min(vertices[:, 2])
    zmax = np.max(vertices[:, 2])

    assert xmax > xmin
    assert ymax > ymin
    assert zmax > zmin

    xmid = (xmax + xmin) / 2
    ymid = (ymax + ymin) / 2
    zmid = (zmax + zmin) / 2
    center_in_local = np.array([xmid, ymid, zmid])

    local_to_bbox = np.eye(4, 4)
    local_to_bbox[:3, 3] = origin_in_local - center_in_local

    xpos = xmax - xmid
    xneg = xmid - xmin
    ypos = ymax - ymid
    yneg = ymid - ymin
    zpos = zmax - zmid
    zneg = zmid - zmin

    assert xpos > 0
    assert xneg > 0
    assert ypos > 0
    assert yneg > 0
    assert zpos > 0
    assert zneg > 0

    scaffold = Scaffold(xneg, xpos, yneg, ypos, zneg, zpos, center_in_local)

    return Bbox(origin_in_local, scaffold, local_to_bbox)


def get_zero_bbox():
    local_to_bbox = np.eye(4, 4)
    center = np.array([0, 0, 0])
    scaffold = Scaffold(0, 0, 0, 0, 0, 0, center)
    origin_in_local = np.array([0, 0, 0], dtype=np.float32)
    return Bbox(origin_in_local, scaffold, local_to_bbox)


def reconstruct_bbox(origin_in_local, scaffold: Scaffold):
    local_to_bbox = np.eye(4, 4)
    # local_to_bbox[:3, 3] = -origin_in_local
    local_to_bbox[:3, 3] = origin_in_local-scaffold.center_in_local
    return Bbox(origin_in_local, scaffold, local_to_bbox)


# given two bounding boxes, find mapping from one to another
def get_bbox_mapping(b_dest: Bbox, b_src: Bbox) -> Mappings:
    # we need a mapping from b_src to b_dest
    def get_one_octant_mapping(x1, y1, z1, x2, y2, z2):
        # print(x1, x2)
        xscale = x1 / x2
        yscale = y1 / y2
        zscale = z1 / z2
        mapping = np.eye(4, 4)
        mapping[0][0] = xscale
        mapping[1][1] = yscale
        mapping[2][2] = zscale
        # print(mapping)
        return mapping

    dest_scaf: Scaffold = b_dest.scaffold
    src_scaf: Scaffold = b_src.scaffold

    src_to_dest = Mappings(
        get_one_octant_mapping(
            dest_scaf.xpos, dest_scaf.ypos, dest_scaf.zpos,
            src_scaf.xpos, src_scaf.ypos, src_scaf.zpos),
        get_one_octant_mapping(
            dest_scaf.xneg, dest_scaf.ypos, dest_scaf.zpos,
            src_scaf.xneg, src_scaf.ypos, src_scaf.zpos),
        get_one_octant_mapping(
            dest_scaf.xpos, dest_scaf.yneg, dest_scaf.zpos,
            src_scaf.xpos, src_scaf.yneg, src_scaf.zpos),
        get_one_octant_mapping(
            dest_scaf.xneg, dest_scaf.yneg, dest_scaf.zpos,
            src_scaf.xneg, src_scaf.yneg, src_scaf.zpos),
        get_one_octant_mapping(
            dest_scaf.xpos, dest_scaf.ypos, dest_scaf.zneg,
            src_scaf.xpos, src_scaf.ypos, src_scaf.zneg),
        get_one_octant_mapping(
            dest_scaf.xneg, dest_scaf.ypos, dest_scaf.zneg,
            src_scaf.xneg, src_scaf.ypos, src_scaf.zneg),
        get_one_octant_mapping(
            dest_scaf.xpos, dest_scaf.yneg, dest_scaf.zneg,
            src_scaf.xpos, src_scaf.yneg, src_scaf.zneg),
        get_one_octant_mapping(
            dest_scaf.xneg, dest_scaf.yneg, dest_scaf.zneg,
            src_scaf.xneg, src_scaf.yneg, src_scaf.zneg))

    return src_to_dest


def interp_bbox(b_dest: Bbox, b_src: Bbox, t,
                interp_joint, interp_world_to_local):
    # dest bbox should become more and more like src bbox
    # interpolate scaffolds
    origin_in_local = kit.transform_one_point(
        interp_joint, interp_world_to_local)
    new_scaffold = list(map(lambda x, y: kit.lerp(x, y, t),
                            b_dest.scaffold.tolist(),
                            b_src.scaffold.tolist()))
    new_center_in_local = kit.lerp(
        b_dest.scaffold.center_in_local,
        b_src.scaffold.center_in_local, t)

    new_scaffold = Scaffold(
        new_scaffold[0], new_scaffold[1], new_scaffold[2],
        new_scaffold[3], new_scaffold[4], new_scaffold[5],
        new_center_in_local)
    return reconstruct_bbox(origin_in_local, new_scaffold)


def get_one_bone_subdiv_bboxes_and_xforms(bbox: Bbox, bone_len,
                                          head_traced_perc: List[float],
                                          the_other_g_nodes: List[AnyNode]=None):
    """
    
    returns
    bboxes: subdivided bboxes 
    xforms: xform mats that send points from subdiv bbox to orig bbox
    """
    if len(head_traced_perc) == 1:
        assert the_other_g_nodes is None
    else:
        assert the_other_g_nodes is not None

    bboxes = []
    xforms = []

    if len(head_traced_perc) == 1:
        bboxes.append(copy.deepcopy(bbox))

        xform = np.eye(4, 4, dtype=np.float32)
        xforms.append(xform) 

        return bboxes, xforms

    all_bone_seg_len = []       # new_virtual_bones_len
    complete = copy.deepcopy(head_traced_perc)
    complete.append(1.0)
    for i in range(len(head_traced_perc)):
        all_bone_seg_len.append(
            (complete[i+1] - complete[i]) * bone_len)

    # NOTE: this shouldn't be zero
    all_many_bboxes_x_len_sum = 0
    for n in the_other_g_nodes:
        box: Bbox = n.bbox
        # all_many_bboxes_x_len_sum += box.scaffold.xpos + box.scaffold.xneg
        all_many_bboxes_x_len_sum += box.scaffold.ypos + box.scaffold.yneg

    all_many_bboxes_x_len_portion = []
    all_many_bboxes_bbox_center_over_bone_len = []
    for n in the_other_g_nodes:
        box: Bbox = n.bbox
        # x_len = box.scaffold.xpos + box.scaffold.xneg
        x_len = box.scaffold.ypos + box.scaffold.yneg
        all_many_bboxes_x_len_portion.append(x_len / all_many_bboxes_x_len_sum)
        # assert box.scaffold.center_in_local[1] > 0
        c_o_bl = box.scaffold.center_in_local[1] / n.bone_len
        all_many_bboxes_bbox_center_over_bone_len.append(c_o_bl)

    # assert bbox.scaffold.center_in_local[1] > 0
    
    # orig_bbox_x_len = bbox.scaffold.xpos + bbox.scaffold.xneg
    orig_bbox_x_len = bbox.scaffold.ypos + bbox.scaffold.yneg

    for i in range(len(head_traced_perc)):
        origin_in_local = np.array([0, 0, 0], dtype=np.float32)
        
        this_one_virt_bone_len = all_bone_seg_len[i]
        this_one_x_len_portion = all_many_bboxes_x_len_portion[i]
        this_one_c_o_bl = all_many_bboxes_bbox_center_over_bone_len[i]

        new_x_len = this_one_x_len_portion * orig_bbox_x_len

        new_bbox_center_x = this_one_c_o_bl * this_one_virt_bone_len
        new_bbox_center = np.array([
                bbox.scaffold.center_in_local[0],
                new_bbox_center_x,
                bbox.scaffold.center_in_local[2]])
        
        local_to_bbox = np.eye(4, 4, dtype=np.float32)
        local_to_bbox[:3, 3] = origin_in_local - new_bbox_center

        new_yneg = new_x_len / 2
        new_ypos = new_x_len / 2

        scaffold = Scaffold(
            bbox.scaffold.xneg, bbox.scaffold.xpos,
            new_yneg, new_ypos,
            bbox.scaffold.zneg, bbox.scaffold.zpos,
            new_bbox_center
        )

        new_bbox = reconstruct_bbox(origin_in_local, scaffold)
        bboxes.append(new_bbox)

        xforms.append(local_to_bbox)
    return bboxes, xforms
