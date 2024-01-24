import argparse
import json
import drawSvg as draw
import math
import base64
import cv2
import numpy as np
import pickle
import os
import copy
from PIL import Image as Img
from io import BytesIO

from adjust_aspect_ratio import *
# from retime import *
from svg_utils.utils import *


class Image(draw.DrawingParentElement):
    TAG_NAME = 'image'

    def __init__(self, href, width, height, x=0, y=0, **kwargs):
        # Other init logic...
        # Keyword arguments to super().__init__() correspond to SVG node
        # arguments: stroke_width=5 -> stroke-width="5"
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
    '--resize', type=int,
    help='Whether to resize the svg given a resize.json file.'
)
    
arg = parser.parse_args()
frame_rate = 60


def create_svg(motion_file, frame_height, frame_width):

    animation_duration = calc_animation_duration(motion_file)
    animation_duration_s = animation_duration / frame_rate  # in seconds

    # assuming background doesn't change for now
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
                     data_frame_rate=frame_rate
                     )

    d.append(draw.Rectangle(0, -frame_height, frame_width,
             frame_height, fill=background_color, id="background"))

    # list of shapes in g tag
    gs = []
    
    # start at 1 to skip -1 (meta info)
    for i in motion_file:
        if i == '-1':
            continue
        curr_shape = motion_file[str(i)]
        shape_file_path = os.path.join(arg.video_dir, 'shapes', str(i) +'.png')
        
        # im = cv2.imread(shape_file_path)
        # height = im.shape[0]
        # width = im.shape[1]
        im = Img.open(shape_file_path)
        width, height = im.size

        # rotate and scale into canonical form
        # cv2.imshow('im_before', cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        if 'canonical' in curr_shape:
            sx, sy, theta = curr_shape['canonical'][0]
            im = im.resize((int(width * sx), int(height * sy)))
            im = im.rotate(np.rad2deg(theta), Img.BILINEAR, expand=0)
        # cv2.imshow('im_after', cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

        # create a SVG image in base64 format
        # with open(shape_file_path, "rb") as image_file:
        #     encoded_string = base64.b64encode(image_file.read())
        buf = BytesIO()
        im.save(buf, format="PNG")
        encoded_string = base64.b64encode(buf.getvalue())

        image = Image('data:image/png;base64,' + encoded_string.decode('utf-8'), width=width,
                      height=height, x=-width/2, y=-height/2)

        # adding animation attributes
        shape_animation_duration = curr_shape['time'][-1] - curr_shape['time'][0] + 1
        shape_animation_duration_s = shape_animation_duration / frame_rate
        dur = str(shape_animation_duration_s) + "s"

        # animation start and end adjustment
        start_delay = curr_shape['time'][0]  # in frame
        start_delay_dur = str((start_delay - 1) / frame_rate) + "s"
        
        animate_canon_rotation = []
        animate_canon_scale = []
        animate_scale = []
        animate_rotation = []
        animate_translation = []
        animate_kx = []
        animate_ky = []
        animate_z = []  # not implemented
        # one way to do this is to find the highest z value, rank the objects based on that, and render that way
        
        frame_size = [frame_width, frame_height]

        if 'canonical' in curr_shape:
            append_to_transform_value_arr("scale", animate_canon_scale, [curr_shape['canonical'][0][0], curr_shape['canonical'][0][1]], frame_size)
            append_to_transform_value_arr("rotate", animate_canon_rotation, [curr_shape['canonical'][0][2]], frame_size)
        append_to_transform_value_arr("scale", animate_scale, [curr_shape['sx'][0], curr_shape['sy'][0]], frame_size)
        append_to_transform_value_arr("rotate", animate_rotation, [curr_shape['theta'][0]], frame_size)
        append_to_transform_value_arr("shear_x", animate_kx, [curr_shape['kx'][0]], frame_size)
        append_to_transform_value_arr("shear_y", animate_ky, [curr_shape['ky'][0]], frame_size)
        append_to_transform_value_arr("translation", animate_translation, [curr_shape['cx'][0], curr_shape['cy'][0]], frame_size)
        
        for j in range(1, len(curr_shape['time'])):
            
            # if there is a skip in frame, interpolate (j > 0 already)
            # if curr_shape['time'][j] != cano_frame_num:
            frame_diff = curr_shape['time'][j] - curr_shape['time'][j - 1]
            skipped_sx = generate_values_of_skipped_frames(curr_shape, frame_diff, 'sx', j)
            skipped_sy = generate_values_of_skipped_frames(curr_shape, frame_diff, 'sy', j)
            skipped_theta = generate_values_of_skipped_frames(curr_shape, frame_diff, 'theta', j)
            skipped_kx = generate_values_of_skipped_frames(curr_shape, frame_diff, 'kx', j)
            skipped_ky = generate_values_of_skipped_frames(curr_shape, frame_diff, 'ky', j)
            skipped_cx = generate_values_of_skipped_frames(curr_shape, frame_diff, 'cx', j)
            skipped_cy = generate_values_of_skipped_frames(curr_shape, frame_diff, 'cy', j)

            assert(frame_diff == len(skipped_sx))
            
            for k in range(frame_diff):
                append_to_transform_value_arr("scale", animate_scale, [skipped_sx[k], skipped_sy[k]], frame_size)
                append_to_transform_value_arr("rotate", animate_rotation, [skipped_theta[k]], frame_size)
                append_to_transform_value_arr("shear_x", animate_kx, [skipped_kx[k]], frame_size)
                append_to_transform_value_arr("shear_y", animate_ky, [skipped_ky[k]], frame_size)
                append_to_transform_value_arr("translation", animate_translation, [skipped_cx[k], skipped_cy[k]], frame_size)

        shape_name = curr_shape['shape'].split('/')[-1].split('.')[0]
        
        # display
        key_value, key_times = generate_display_key_times(start_delay, shape_animation_duration, animation_duration, frame_rate)
        
        image.appendAnim(draw.Animate(
            'display',
            str(animation_duration_s) + "s",
            ";".join(key_value),
            keyTimes=";".join(key_times),
            calcMode="discrete",
            fill="freeze",
            id="_".join([shape_name, "display"])
        ))


        # shear x and y
        append_animate_transform(image, 'skewY', dur, animate_ky, shape_name, start_delay_dur)
        append_animate_transform(image, 'skewX', dur, animate_kx, shape_name, start_delay_dur)

        # rotation
        append_animate_transform(image, 'rotate', dur, animate_rotation, shape_name, start_delay_dur)

        # scale x and y
        append_animate_transform(image, 'scale', dur, animate_scale, shape_name, start_delay_dur)
        
        if 'canonical' in curr_shape:
            # canonical scale x and y
            append_animate_transform(image, 'scale', dur, animate_canon_scale, shape_name, start_delay_dur)
            # canonical rotation
            append_animate_transform(image, 'rotate', dur, animate_canon_rotation, shape_name, start_delay_dur)
        
        # translation
        # g1 = Group(transform_origin="center")
        g1 = Group(id="shape_" + str(i),
                   data_start_frame=curr_shape['time'][0], 
                   data_end_frame=curr_shape['time'][-1])
        g1.append(image)
        append_animate_transform(
            g1, 'translate', dur, animate_translation, shape_name, start_delay_dur)

        # d.append(g1)
        gs.append(g1)
        
    return [d, gs]


def generate_values_of_skipped_frames(curr_shape, frame_diff, t_type, index):
    if frame_diff == 1:
        return [curr_shape[t_type][index]]
    
    res = np.linspace(curr_shape[t_type][index - 1], curr_shape[t_type][index], num = frame_diff + 1)
    res = res[1:-1]
    res = np.append(res, curr_shape[t_type][index])

    return res
        

def append_animate_transform(image, type, dur, values_list, shape_name, start_delay_dur):
    image.appendAnim(draw.AnimateTransform(
        type,
        dur,
        ";".join(values_list),
        attributeType="XML",
        id="_".join([shape_name, type]),
        begin=start_delay_dur,
        additive="sum",
        calcMode="discrete",
        fill="freeze"
    ))


def main():

    # Load in computed template info.
    # motion_file = json.load(open(arg.motion_file, 'rb'))
    motion_file = json.load(
        open(os.path.join(arg.video_dir, 'motion_file_full.json'), 'rb'))

    output_file_name = 'motion_file.svg'
    output_path = os.path.join(arg.video_dir, output_file_name)

    frame_height = motion_file['-1']['height']
    frame_width = motion_file['-1']['width']

    d, gs = create_svg(motion_file, frame_height, frame_width)
    
    if arg.resize == 1:
        resize_file = json.load(
            open(os.path.join(arg.video_dir, 'resize.json'), 'rb'))
        
        svg_groups, group_nums = group_shapes(motion_file, resize_file, d, gs)
        resize_aspect_raio(motion_file, resize_file, d, svg_groups, group_nums, [684, 512])
    else:
        for g in gs:
            d.append(g)
    
    d.saveSvg(output_path)

    # # find bounding box of extreme points
    # extreme_bounding_box = find_extreme_poses(arg.video_dir, motion_file)

    # # cropping
    # crop_sizes = [[2, 1], [1, 2], [16, 9], [9, 16], [3, 2], [2, 3], [1, 1]]
    # for crop_size in crop_sizes:
    #     cropped_d = crop_svg(copy.deepcopy(d), extreme_bounding_box, [
    #                          frame_width, frame_height], crop_size)
    #     cropped_d.saveSvg(os.path.join(
    #         arg.video_dir, 'motion_file_{}_{}.svg'.format(crop_size[0], crop_size[1])))

    # # bounding box visualization
    # d.append(draw.Rectangle(
    #     extreme_bounding_box[0], -
    #         (extreme_bounding_box[1] + extreme_bounding_box[3]),
    #     extreme_bounding_box[2], extreme_bounding_box[3],
    #     fill='transparent',
    #     stroke_width=3,
    #     stroke="rgb(0, 0, 255)",
    #     z_index="999"
    # ))
    # d.saveSvg(os.path.join(arg.video_dir, 'motion_file_bounds.svg'))


if __name__ == '__main__':
    main()
