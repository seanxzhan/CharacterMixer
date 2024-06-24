import re
import os
import numpy as np
from PIL import Image
from utils import binvox_rw
import colorsys


def load_voxels(vox_path):
    with open(vox_path, 'rb') as f:
        voxel_model_dense = binvox_rw.read_as_3d_array(f)
        voxels = voxel_model_dense.data.astype(int)
    return voxels


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def save_fig(plt, title, img_path, rotate=True, transparent=True):
    plt.title(title)
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=150, 
                transparent=transparent)
    if rotate:
        im = Image.open(img_path)
        im = im.rotate(90)
        im.save(img_path)


def save_images_as_gif(dir_, condition, interp_filename):
    frames = []
    imgs = sorted_alphanumeric(os.listdir(dir_))
    for i in imgs:
        if condition(i):
            new_frame = Image.open(os.path.join(dir_, i))
            frames.append(new_frame)

    out_gif = os.path.join(dir_, interp_filename)
    frames[0].save(out_gif, format='GIF', append_images=frames[1:],
                    save_all=True, duration=300, loop=0)
    print("saved to:", out_gif)


def save_images_as_gif_paths(paths, interp_filepath):
    frames = []
    for p in paths:
        new_frame = Image.open(p)
        frames.append(new_frame)

    frames[0].save(interp_filepath, format='GIF', append_images=frames[1:],
                   save_all=True, duration=300, loop=0)
    print("saved to:", interp_filepath)


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    result = []
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        result.append(l[si:si+(d+1 if i < r else d)])
    return result


def renorm(x):
    d1 = x.copy()
    diff = np.max(d1) - np.min(d1)
    d1 = (d1 - np.min(d1)) * (1 / diff)
    d1 = 1 - d1
    return d1


def x_within_n(x, n, epsilon=0.05):
    return abs(x - n) <= epsilon


def write_obj(new_vertices, faces, output_path):
    out_file = open(output_path, 'w')

    for i in range(len(new_vertices)):
        new_line = "v {:.6f} {:.6f} {:.6f}\n".format(
            new_vertices[i][0], new_vertices[i][1], new_vertices[i][2])
        out_file.write(new_line)
    
    for i in range(len(faces)):
        new_line = "f {} {} {}\n".format(
            faces[i][0]+1, faces[i][1]+1, faces[i][2]+1)
        out_file.write(new_line)
    
    out_file.close()


def check_obj_attrs(obj):
    attrs = dir(obj)
    print(attrs)


def hex2rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def increase_saturation(rgb, percent):
    # convert RGB values to HSV (hue, saturation, value) format
    hsv = colorsys.rgb_to_hsv(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    # increase the saturation by a percentage
    hsv = (hsv[0], hsv[1] + percent / 100.0, hsv[2])
    # convert back to RGB format
    rgb = tuple(max(0, min(int(x * 255), 255)) for x in colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2]))
    return rgb


def hex2rgb_sat(h, percent):
    color = hex2rgb(h)
    return increase_saturation(color, percent)


def get_tab_20():
    return [
        hex2rgb('9edae5'),
        hex2rgb('17becf'),
        hex2rgb('dbdb8d'),
        hex2rgb('bcbd22'),
        hex2rgb('c7c7c7'),
        hex2rgb('7f7f7f'),
        hex2rgb('f7b6d2'),
        hex2rgb('e377c2'),
        hex2rgb('c49c94'),
        hex2rgb('8c564b'),
        hex2rgb('c5b0d5'),
        hex2rgb('9467bd'),
        hex2rgb('ff9896'),
        hex2rgb('d62728'),
        hex2rgb('98df8a'),
        hex2rgb('2ca02c'),
        hex2rgb('ffbb78'),
        hex2rgb('ff7f0e'),
        hex2rgb('aec7e8'),
        hex2rgb('1f77b4'),
    ]


def get_tab_20_saturated(percent):
    return [
        hex2rgb_sat('9edae5', percent),
        hex2rgb_sat('17becf', percent),
        hex2rgb_sat('dbdb8d', percent),
        hex2rgb_sat('bcbd22', percent),
        hex2rgb_sat('c7c7c7', percent),
        hex2rgb_sat('7f7f7f', percent),
        hex2rgb_sat('f7b6d2', percent),
        hex2rgb_sat('e377c2', percent),
        hex2rgb_sat('c49c94', percent),
        hex2rgb_sat('8c564b', percent),
        hex2rgb_sat('c5b0d5', percent),
        hex2rgb_sat('9467bd', percent),
        hex2rgb_sat('ff9896', percent),
        hex2rgb_sat('d62728', percent),
        hex2rgb_sat('98df8a', percent),
        hex2rgb_sat('2ca02c', percent),
        hex2rgb_sat('ffbb78', percent),
        hex2rgb_sat('ff7f0e', percent),
        hex2rgb_sat('aec7e8', percent),
        hex2rgb_sat('1f77b4', percent),
    ]