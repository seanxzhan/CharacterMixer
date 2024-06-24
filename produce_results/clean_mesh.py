import os
import pymeshlab
import argparse
from tqdm import tqdm
from utils import misc
from produce_results import make_t_steps

def clean_mesh_mlab(input_pathname, output_pathname):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_pathname)
    ms.meshing_snap_mismatched_borders()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_t_vertices()
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pymeshlab.Percentage(40))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=10)
    ms.meshing_surface_subdivision_butterfly()
    ms.save_current_mesh(output_pathname)


if __name__ == '__main__':
    out_dir = os.path.join('results', 'interp_sdf')

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
    seq_id = args.seq_id
    num_interp = args.num_interp
    interp_step = args.interp_step
    num_frames = args.frame_end - args.frame_start + 1
    fix_t = bool(args.fix_t)

    if fix_t:
        lst_of_interp_steps = [interp_step] * num_frames
    if not fix_t:
        if seq_id == 0:
            _, lst_of_interp_steps, num_interp =\
                make_t_steps.make_t_pose(num_frames)
        else:
            if (src_id, dest_id) in [(432, 1299), (1379, 14035)]:
                _, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_inf_loop(num_frames)
                
            if (src_id, dest_id) == (2097, 2091):
                _, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_2097_2091(num_frames)

            if (src_id, dest_id) == (4010, 1919):
                _, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_4010_1919(num_frames)
                
            if (src_id, dest_id) == (12852, 12901):
                _, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_12852_12901(num_frames)
                
            if (src_id, dest_id) == (16880, 16827):
                _, lst_of_interp_steps, num_interp =\
                    make_t_steps.make_16880_16827(num_frames)

        interp_step = 'x'

    pair_interp_dir = os.path.join(out_dir, '{}_{}'.format(src_id, dest_id))
    seq_dir = os.path.join(pair_interp_dir, 'seq_id_{}_{}'.format(seq_id, res))
    vox_cleaned_mesh_dir = os.path.join(seq_dir, f'cleaned_vox_{interp_step}')
    misc.check_dir(vox_cleaned_mesh_dir)
    gt_src_cleaned_mesh_dir = os.path.join(seq_dir, f'cleaned_{src_id}_{interp_step}')
    misc.check_dir(gt_src_cleaned_mesh_dir)
    gt_dest_cleaned_mesh_dir = os.path.join(seq_dir, f'cleaned_{dest_id}_{interp_step}')
    misc.check_dir(gt_dest_cleaned_mesh_dir)

    print("cleaning meshes...")
    for i in tqdm(range(num_frames)):
        frame = i + 1
        interp_step = lst_of_interp_steps[i]

        if seq_id != 0:
            posed_dir = os.path.join(seq_dir, f'posed_id_{frame}_{res}')
            expt_dir = os.path.join(
                posed_dir,
                f'step_{interp_step}_{num_interp}_{frame}_{num_frames}')
        else:
            posed_dir = os.path.join(seq_dir, f'posed_id_0_{res}')
            expt_dir = os.path.join(
                posed_dir,
                f'step_{interp_step}_{num_interp}_0_0')
        
        vox_in_path = os.path.join(expt_dir, 'posed_gt.obj')
        gt_src_in_path = os.path.join(posed_dir, 'gt_recon', f'gt_{src_id}_recon_parts_{res}.obj')
        gt_dest_in_path = os.path.join(posed_dir, 'gt_recon', f'gt_{dest_id}_recon_parts_{res}.obj')
        vox_out_path = os.path.join(
            vox_cleaned_mesh_dir,
            f'{src_id}_{dest_id}_seq_{seq_id}_frame_{frame}_vox_{res}.obj')
        gt_src_out_path = os.path.join(
            gt_src_cleaned_mesh_dir,
            f'{src_id}_{dest_id}_seq_{seq_id}_frame_{frame}_{src_id}_{res}.obj')
        gt_dest_out_path = os.path.join(
            gt_dest_cleaned_mesh_dir,
            f'{src_id}_{dest_id}_seq_{seq_id}_frame_{frame}_{dest_id}_{res}.obj')
        clean_mesh_mlab(vox_in_path, vox_out_path)
        clean_mesh_mlab(gt_src_in_path, gt_src_out_path)
        clean_mesh_mlab(gt_dest_in_path, gt_dest_out_path)
