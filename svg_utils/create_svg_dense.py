import argparse
import json
import drawSvg as draw
# import math
import base64
# import cv2
import numpy as np
# import pickle
import os
# import copy
from io import BytesIO
from PIL import Image as PILImage

from .utils import *


class Image(draw.DrawingParentElement):
    TAG_NAME = 'image'
    def __init__(self, href, width, height, x=0, y=0, **kwargs):
        super().__init__(href=href, width=width, height=height, x=x, y=y, ** kwargs)


class Group(draw.DrawingParentElement):
    TAG_NAME = 'g'
    def __init__(self, **kwargs):
        super().__init__(** kwargs)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--video_dir', required=True, type=str,
    help='Directory containing video data.')
parser.add_argument(
    '--frame_rate', default=24, type=int, 
    help='Output SVG frame rate.')    

arg = parser.parse_args()


def add_svg_obj(curr_shape, shape_id, shape_file_path, frame_size, animation_duration, frame_rate):
    animation_duration_s = animation_duration / frame_rate  # in seconds
    width = 0
    height = 0

    im = PILImage.open(shape_file_path)
    width, height = im.size

    # rotate and scale into canonical form
    if 'canonical' in curr_shape:
        sx, sy, theta = curr_shape['canonical'][0]
        im = im.resize((int(width * sx), int(height * sy)))
        im = im.rotate(np.rad2deg(theta), PILImage.BILINEAR, expand=0)

    # create a SVG image in base64 format
    buf = BytesIO()
    im.save(buf, format="PNG")
    encoded_string = base64.b64encode(buf.getvalue())

    image = Image('data:image/png;base64,' + encoded_string.decode('utf-8'), width=width,
                    height=height, x=-width/2, y=-height/2)
    
    animate_canon_rotation = []
    animate_canon_scale = []
    animate_scale = []
    animate_rotation = []
    animate_translation = []
    animate_kx = []
    animate_ky = []
    animate_display = []
    animate_z = []       

    if 'canonical' in curr_shape:
        append_to_transform_value_arr("scale", animate_canon_scale, [curr_shape['canonical'][0][0], curr_shape['canonical'][0][1]], frame_size)
        append_to_transform_value_arr("rotate", animate_canon_rotation, [curr_shape['canonical'][0][2]], frame_size)

    # constructing transform info at every frame
    # populate frames before recorded frames. frame starts at 1
    for j in range(0, curr_shape['time'][0] - 1):
        append_to_transform_value_arr("scale", animate_scale, [curr_shape['sx'][0], curr_shape['sy'][0]], frame_size)
        append_to_transform_value_arr("shear_x", animate_kx, [curr_shape['kx'][0]], frame_size)
        append_to_transform_value_arr("shear_y", animate_ky, [curr_shape['ky'][0]], frame_size)
        append_to_transform_value_arr("rotate", animate_rotation, [curr_shape['theta'][0]], frame_size)
        append_to_transform_value_arr("translation", animate_translation, [curr_shape['cx'][0], curr_shape['cy'][0]], frame_size)
        append_to_transform_value_arr("display", animate_display, ['none'], frame_size)
        append_to_transform_value_arr("z", animate_z, [curr_shape['z'][0]], frame_size)
        
    
    # populate defined frames
    append_to_transform_value_arr("scale", animate_scale, [curr_shape['sx'][0], curr_shape['sy'][0]], frame_size)
    append_to_transform_value_arr("shear_x", animate_kx, [curr_shape['kx'][0]], frame_size)
    append_to_transform_value_arr("shear_y", animate_ky, [curr_shape['ky'][0]], frame_size)
    append_to_transform_value_arr("rotate", animate_rotation, [curr_shape['theta'][0]], frame_size)
    append_to_transform_value_arr("translation", animate_translation, [curr_shape['cx'][0], curr_shape['cy'][0]], frame_size)
    append_to_transform_value_arr("display", animate_display, ['inline'], frame_size)
    append_to_transform_value_arr("z", animate_z, [curr_shape['z'][0]], frame_size)
    
    # print(shape_id, curr_shape['time'])
    for j in range(1, len(curr_shape['time'])):
        # if there is a skip in frame, interpolate (j > 0 already)
        frame_diff = curr_shape['time'][j] - curr_shape['time'][j - 1]
        skipped_sx = generate_values_of_skipped_frames(curr_shape, frame_diff, 'sx', j)
        skipped_sy = generate_values_of_skipped_frames(curr_shape, frame_diff, 'sy', j)
        skipped_kx = generate_values_of_skipped_frames(curr_shape, frame_diff, 'kx', j)
        skipped_ky = generate_values_of_skipped_frames(curr_shape, frame_diff, 'ky', j)
        skipped_theta = generate_values_of_skipped_frames(curr_shape, frame_diff, 'theta', j)
        skipped_cx = generate_values_of_skipped_frames(curr_shape, frame_diff, 'cx', j)
        skipped_cy = generate_values_of_skipped_frames(curr_shape, frame_diff, 'cy', j)
        skipped_z = generate_values_of_skipped_frames(curr_shape, frame_diff, 'z', j)
        
        for k in range(frame_diff - 1):
            append_to_transform_value_arr("scale", animate_scale, [skipped_sx[k], skipped_sy[k]], frame_size)
            append_to_transform_value_arr("shear_x", animate_kx, [skipped_kx[k]], frame_size)
            append_to_transform_value_arr("shear_y", animate_ky, [skipped_ky[k]], frame_size)
            append_to_transform_value_arr("rotate", animate_rotation, [skipped_theta[k]], frame_size)
            append_to_transform_value_arr("translation", animate_translation, [skipped_cx[k], skipped_cy[k]], frame_size)
            append_to_transform_value_arr("display", animate_display, ['inline'], frame_size)
            append_to_transform_value_arr("z", animate_z, [skipped_z[k]], frame_size)
            
        append_to_transform_value_arr("scale", animate_scale, [curr_shape['sx'][j], curr_shape['sy'][j]], frame_size)
        append_to_transform_value_arr("shear_x", animate_kx, [curr_shape['kx'][j]], frame_size)
        append_to_transform_value_arr("shear_y", animate_ky, [curr_shape['ky'][j]], frame_size)
        append_to_transform_value_arr("rotate", animate_rotation, [curr_shape['theta'][j]], frame_size)
        append_to_transform_value_arr("translation", animate_translation, [curr_shape['cx'][j], curr_shape['cy'][j]], frame_size)
        append_to_transform_value_arr("display", animate_display, ['inline'], frame_size)
        append_to_transform_value_arr("z", animate_z, [curr_shape['z'][j]], frame_size)
            
    # populate frames after. frame starts at 1
    # print(curr_shape['time'][-1], animation_duration, len(animate_display))
    for j in range(curr_shape['time'][-1], animation_duration):
        append_to_transform_value_arr("scale", animate_scale, [curr_shape['sx'][-1], curr_shape['sy'][-1]], frame_size)
        append_to_transform_value_arr("shear_x", animate_kx, [curr_shape['kx'][-1]], frame_size)
        append_to_transform_value_arr("shear_y", animate_ky, [curr_shape['ky'][-1]], frame_size)
        append_to_transform_value_arr("rotate", animate_rotation, [curr_shape['theta'][-1]], frame_size)
        append_to_transform_value_arr("translation", animate_translation, [curr_shape['cx'][-1], curr_shape['cy'][-1]], frame_size)
        append_to_transform_value_arr("display", animate_display, ['none'], frame_size)
        append_to_transform_value_arr("z", animate_z, [curr_shape['z'][-1]], frame_size)
            
    assert(len(animate_scale) == animation_duration)
    assert(len(animate_z) == animation_duration)
    shape_name = curr_shape['shape'].split('/')[-1].split('.')[0]
    
    # display
    if (curr_shape['z'][0]) == -1:  # used to get rid of shapes
        animate_display = np.repeat('none', len(animate_display))
    append_animate_transform(image, 'display', animation_duration_s, animate_display, shape_name)
    
    # shear x and y
    append_animate_transform(image, 'skewY', animation_duration_s, animate_ky, shape_name)
    append_animate_transform(image, 'skewX', animation_duration_s, animate_kx, shape_name)

    # rotation
    append_animate_transform(image, 'rotate', animation_duration_s, animate_rotation, shape_name)

    # scale x and y
    append_animate_transform(image, 'scale', animation_duration_s, animate_scale, shape_name)
    
    if 'canonical' in curr_shape:
        # canonical scale x and y
        append_animate_transform(image, 'scale', animation_duration_s, animate_canon_scale, shape_name)
        # canonical rotation
        append_animate_transform(image, 'rotate', animation_duration_s, animate_canon_rotation, shape_name)
    
    # z-index
    append_animate_transform(image, 'z-index', animation_duration_s, animate_z, shape_name)

    # translation
    g1 = Group(
        id=shape_id,
        data_start_frame_index = curr_shape['time'][0] - 1, 
        data_end_frame_index = curr_shape['time'][-1] - 1)
    g1.append(image)
    append_animate_transform(g1, 'translate', animation_duration_s, animate_translation, shape_name)
    
    return g1


def create_svg(motion_file, frame_height, frame_width):
    animation_duration = calc_animation_duration(motion_file)

    # assuming background doesn't change for now
    if 'bg_color' in motion_file['-1']:
        bg = [str(i) for i in motion_file['-1']['bg_color'][0]]
    else:
        bg = [str(i) for i in motion_file['-1']['bg'][0]]
    temp = bg[2]
    bg[2] = bg[0]
    bg[0] = temp
    background_color = 'rgb(' + ",".join(bg) + ')'

    d = draw.Drawing(frame_width,
                     frame_height,
                     origin=(0, -frame_height),
                     displayInline=False,
                     id="svg_id",
                     data_duration=animation_duration,
                     data_frame_rate=arg.frame_rate
                     )

    d.append(draw.Rectangle(0, -frame_height, frame_width,
             frame_height, fill=background_color, id="background"))

    # list of shapes in g tag
    gs = []
    
    # start at 1 to skip -1 (meta info)
    # for i in range(len(motion_file)-1):
    # curr_shape = motion_file[str(i)]
    for i in motion_file:
        if i == '-1':
            continue
        curr_shape = motion_file[i]
        shape_file_path = os.path.join(arg.video_dir, 'shapes', i +'.png')
        if 'shape' in curr_shape:
            shape_file_path = curr_shape['shape']
        shape_id = "shape_" + str(i)
        g1 = add_svg_obj(curr_shape, shape_id, shape_file_path, [frame_width, frame_height], animation_duration, arg.frame_rate)
        gs.append(g1)

    # one way to do this is to find the highest z value, rank and sort the objects based on that, and render that way
    shape_avg_z_s = rank_z_index(motion_file)
    # gs = np.array(gs)[np.argsort(shape_avg_z_s)]
    gs = [gs[i] for i in np.argsort(shape_avg_z_s)]

    # make an object for the background
    if 'bg_img' in motion_file['-1'] and motion_file['-1']['bg_img'] is not None:
        base64_string = convert_image_to_base64(motion_file['-1']['bg_img'])
    else:
        buf = BytesIO()
        bg_img = PILImage.fromarray(np.zeros((frame_height, frame_width, 4), dtype=np.uint8))
        bg_img.save(buf, format='PNG')
        encoded_string = base64.b64encode(buf.getvalue())
        base64_string = 'data:image/png;base64,' + encoded_string.decode('utf-8')
    image = Image(
        base64_string, width=frame_width, height=frame_height, x=-frame_width/2, y=-frame_height/2)
    append_animate_transform(image, 'z-index', animation_duration / arg.frame_rate, np.repeat('0', animation_duration), 'bg')
    g1 = Group(
        id='shape_bg',
        data_start_frame_index = 0, 
        data_end_frame_index = animation_duration - 1)
    g1.append(image)
    append_animate_transform(g1, 'translate', animation_duration / arg.frame_rate, np.repeat(f'{frame_width/2} {frame_height/2}', animation_duration), 'bg')
    gs.insert(0, g1)
        
    return [d, gs]


def rank_z_index(motion_file):
    shape_avg_z_s = []
    if "time" in motion_file["-1"]:
        global_frames = motion_file["-1"]["time"]
        # build z index organized by frame -> object
        # for every frame
        global_zs = {}
        for i in range(len(global_frames)): 
            curr_global_frame = global_frames[i]
            # print("curr_global_frame: {}".format(curr_global_frame))
            
            # for every object
            curr_zs = {}
            for j in motion_file:
                if j == '-1':
                    continue
            # for j in range(len(motion_file)-1):
                curr_shape = motion_file[str(j)]
                curr_shape_frames = np.array(curr_shape['time'])
                npwhere = np.where(curr_shape_frames == curr_global_frame)
                
                # if there is a recorded transform at this frame
                if (len(npwhere[0]) != 0):
                    frame_index = npwhere[0][0]
                    curr_zs[str(j)] = curr_shape['z'][frame_index]
        
            global_zs[str(curr_global_frame)] = curr_zs

        json_zs = json.dumps(global_zs, indent=4)
        with open(os.path.join(arg.video_dir, "z-index.json"), "w") as outfile:
            outfile.write(json_zs)
        
        # avg z value not counting 5.0
        # for i in range(len(motion_file)-1):
        for i in motion_file:
            if i == '-1':
                continue
            curr_shape = motion_file[str(i)]
            curr_shape_z = np.array([float(z) for z in curr_shape['z'] if not float(z) == 5.0])
            if len(curr_shape_z) == 0:
                shape_avg_z_s.append(5.0)
            else:
                shape_avg_z_s.append(np.average(curr_shape_z))
                
    return np.array(shape_avg_z_s)

    

def generate_values_of_skipped_frames(curr_shape, frame_diff, t_type, index):
    if frame_diff == 1:
        return [curr_shape[t_type][index]]
    
    # res = np.linspace(curr_shape[t_type][index - 1], curr_shape[t_type][index], num = frame_diff + 1)
    
    # not interpolating frames
    res = np.repeat(curr_shape[t_type][index - 1], frame_diff + 1)
    
    res = res[1:-1]
    res = np.append(res, curr_shape[t_type][index])

    return res
        

def append_animate_transform(image, type, dur, values_list, shape_name):
    
    value_string = ";".join(values_list)
    id = "_".join([shape_name, type])
    
    if type == 'display' or type == 'z-index':
        image.appendAnim(draw.Animate(
            type,
            dur,
            value_string,
            id=id,
            attributeType="XML",
            calcMode="discrete",
            fill="freeze",
        ))
    else:
        image.appendAnim(draw.AnimateTransform(
            type,
            dur,
            value_string,
            id=id,
            attributeType="XML",
            calcMode="discrete",
            fill="freeze",
            additive="sum"
        ))


def main():
    motion_file = json.load(open(os.path.join(arg.video_dir, 'motion_file.json'), 'rb'))

    output_file_name = 'motion_file.svg'
    output_path = os.path.join(arg.video_dir, output_file_name)
    print(f"Saving to: {output_path}")

    frame_height = motion_file['-1']['height']
    frame_width = motion_file['-1']['width']

    d, gs = create_svg(motion_file, frame_height, frame_width)
    
    for g in gs:
        d.append(g)
        
    d.saveSvg(output_path)


if __name__ == '__main__':
    main()
