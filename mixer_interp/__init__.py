import os

raw_data_dir = 'data/raw_data'
part_data_dir = 'data/parts'

class Info:
    def __init__(self, **kwargs):
        self.model_id = -1
        self.path_to_obj = ''
        self.path_to_rig_info = ''
        self.path_to_skel = ''
        self.path_to_sdf = ''
        self.dir_to_vox_parts = ''

        # get a list of all predefined values directly from __dict__
        allowed_keys = list(self.__dict__.keys())

        # Update __dict__ but only for keys that have been predefined 
        # (silently ignore others)
        self.__dict__.update((key, value) for key, value in kwargs.items() 
                             if key in allowed_keys)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor: {}".format(rejected_keys))


def make_model_info(model, res=128):
    path_to_obj = os.path.join(
        raw_data_dir, 'obj', '{}.obj'.format(model))
    path_to_rig_info = os.path.join(
        raw_data_dir, 'rig', '{}.txt'.format(model))
    path_to_skel = os.path.join(
        raw_data_dir, 'skel', '{}.json'.format(model))
    path_to_sdf = os.path.join(
        part_data_dir, 'sdf', '{}_sdf'.format(res), '{}.npz'.format(model))
    dir_to_vox_parts = os.path.join(
        part_data_dir, 'vox', '{}_vox_c'.format(res), str(model))
    return Info(
        model_id=model,
        path_to_obj=path_to_obj,
        path_to_rig_info=path_to_rig_info,
        path_to_skel=path_to_skel,
        path_to_sdf=path_to_sdf,
        dir_to_vox_parts=dir_to_vox_parts)


class InterpInfo:
    def __init__(self, **kwargs):
        self.src_id = -1
        self.dest_id = -1
        self.src_info = None
        self.dest_info = None
        self.src_package = None
        self.dest_package = None
        self.src_nodes = []
        self.dest_nodes = []
        self.pairs = None
        self.corres_groups = []
        self.comp_nodes = []
        self.aug_nodes = []
        self.aug_groups = []
        self.colors = []
        self.all_interp_ht = None
        self.all_interp_frames = None
        self.all_interp_w_to_l = None

        # get a list of all predefined values directly from __dict__
        allowed_keys = list(self.__dict__.keys())

        # Update __dict__ but only for keys that have been predefined 
        # (silently ignore others)
        self.__dict__.update((key, value) for key, value in kwargs.items() 
                             if key in allowed_keys)

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError(
                "Invalid arguments in constructor: {}".format(rejected_keys))