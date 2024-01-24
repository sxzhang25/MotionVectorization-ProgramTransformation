import librosa
# import soundfile as sf
# from xml.dom import minidom
import math
import numpy as np
from easing_functions import *
import base64
from PIL import Image
import io
import cv2


def lerp(t, v0, v1):
  return (1 - t) * v0 + t * v1


# https://gist.github.com/th0ma5w/9883420
# https: // spicyyoghurt.com/tools/easing-functions
# t = 0 - Animation is just started. Zero time has passed
# b = 200 - The starting position of the objects x-coordinate is 200
# c = 300 - The object has to move 300 to the right, ending at 500
# d = 1 - The object has one second to perform this motion from 200 to 500
def easeInQuart(t, b, c, d):
	t /= d
	return c*t*t*t*t + b

# a < 1 speed up, a > 1 slowing down
def linearSpeed(t, a):
    return t / a


utils_animate_keys = ['display_animate', 'z_animate', 
                'translate_transform','scale_transform', 
                'rotate_transform', 'skewX_transform', 'skewY_transform']

easing_function_mapping = {
    'QuinticEaseIn': QuinticEaseIn,
    'QuinticEaseOut': QuinticEaseOut,
    'QuinticEaseInOut': QuinticEaseInOut,
    'QuadEaseIn': QuadEaseIn,
    'QuadEaseOut': QuadEaseOut,
    'QuadEaseInOut': QuadEaseInOut,
    'CubicEaseIn': CubicEaseIn,
    'CubicEaseOut': CubicEaseOut,
    'ExponentialEaseIn': ExponentialEaseIn,
    'LinearInOut': LinearInOut
}


def extract_beat_times(audio_file_path, dur):
    y, sr = librosa.load(audio_file_path, duration=dur)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # y, sr = librosa.load(audio_file_path, duration=dur)
    
    # onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    # pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    # beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    # times = librosa.times_like(pulse, sr=sr)
    
    # beat_times = times[beats_plp]
    # beat_frames = librosa.time_to_frames(beat_times)
    
    # y_beats = librosa.clicks(frames=beat_frames, sr=sr, length=len(y))
    # sf.write('../outputs/music/levitating_instrumental_clicks_only.wav', y_beats, sr)
    return beat_times


def convert_image_to_base64(shape_file_path):
    with open(shape_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    base64_string = 'data:image/png;base64,' + encoded_string.decode('utf-8')
    
    return base64_string
    
def calc_animation_duration(motion_file):
    animation_duration = 0  # in frames
    if "time" in motion_file["-1"]:
        times = motion_file["-1"]["time"]
        animation_duration = times[-1]
        
    elif "bg" in motion_file["-1"]:
        animation_duration = len(motion_file['-1']['bg'])
        
    return animation_duration

def get_object_animation_duration(curr_svg_shape):
    return len(parseValuesFromAnimateTransform(curr_svg_shape['scale_transform']))


def append_to_transform_value_arr(type, arr_transform, values, frame_size=[1,1]):
    '''
    Convert numeric values into an array of strings to be joined by ";" for the values field
    '''
    if type == "scale":
        arr_transform.append(" ".join([str(values[0]), str(values[1])]))
    elif type == "rotate":
        arr_transform.append(str(math.degrees(values[0])))
    elif type == "rotate_degree":
        arr_transform.append(str(values[0]))
    elif type == "shear_x":
        arr_transform.append(str(math.degrees(math.atan(values[0]))))
    elif type == "shear_x_atan":
        arr_transform.append(str(values[0]))
    elif type == "shear_y":
        arr_transform.append(str(math.degrees(math.atan(values[0]))))
    elif type == "shear_y_atan":
        arr_transform.append(str(values[0]))
    elif type == "translation":
        curr_centroid_x = values[0] * frame_size[0]
        curr_centroid_y = values[1] * frame_size[1]
        arr_transform.append(" ".join([str(curr_centroid_x), str(curr_centroid_y)]))
    elif type =="display":
        arr_transform.append(values[0])
    elif type =="z":
        arr_transform.append(str(values[0]))


def pad_obj_transform_values(curr_svg_shape, pad_len, frame_rate):
    for transform_key in [
        'z_animate', 
        'translate_transform', 
        'scale_transform', 
        'rotate_transform', 
        'skewX_transform', 
        'skewY_transform',
        'display_animate'
    ]:
        transform_values = parseValuesFromAnimate(curr_svg_shape[transform_key])
        last_value = transform_values[-1]
        for i in range(pad_len):
            if transform_key == 'display_animate':
                transform_values.append('none')
            else:
                transform_values.append(last_value)
        curr_svg_shape[transform_key].setAttribute('values', ";".join(transform_values))
        curr_svg_shape[transform_key].setAttribute('dur', str(len(transform_values) / frame_rate))
           

def write_retimed_values(curr_svg_shape, eased_frames, frame_size, frame_rate, no_offset=False):
    '''
    Given an array of easef frames, retrieve transform values at index
    '''
    
    eased_z_raw = interpolate_frame_values(
        parseValuesFromAnimate(curr_svg_shape['z_animate']), eased_frames, 'z', no_offset)
    eased_display_raw = interpolate_frame_values(
        parseValuesFromAnimate(curr_svg_shape['display_animate']), eased_frames, 'display', no_offset)
    eased_translation_raw = interpolate_frame_values(
        parseValuesFromAnimateTransform(curr_svg_shape['translate_transform']), eased_frames, 'translate', no_offset)
    eased_scale_raw = interpolate_frame_values(
        parseValuesFromAnimateTransform(curr_svg_shape['scale_transform']), eased_frames, 'scale', no_offset)
    eased_rotate_raw = interpolate_frame_values(
        parseValuesFromAnimateTransform(curr_svg_shape['rotate_transform']), eased_frames, 'rotate', no_offset)
    eased_skewX_raw = interpolate_frame_values(
        parseValuesFromAnimateTransform(curr_svg_shape['skewX_transform']), eased_frames, 'skewX', no_offset)
    eased_skewY_raw = interpolate_frame_values(
        parseValuesFromAnimateTransform(curr_svg_shape['skewY_transform']), eased_frames, 'skewY', no_offset)
    
    eased_z = []
    eased_display = []
    eased_translation = []
    eased_scale = []
    eased_rotate = []
    eased_skewX = []
    eased_skewY = []
    for j in range(len(eased_translation_raw)):
        append_to_transform_value_arr('z', eased_z, [eased_z_raw[j]])
        append_to_transform_value_arr('display', eased_display,[eased_display_raw[j]])
        append_to_transform_value_arr('translation', eased_translation,[eased_translation_raw[j][0], eased_translation_raw[j][1]])
        append_to_transform_value_arr('scale', eased_scale,[eased_scale_raw[j][0], eased_scale_raw[j][1]])
        append_to_transform_value_arr('rotate_degree', eased_rotate,[eased_rotate_raw[j]])
        append_to_transform_value_arr('shear_x_atan', eased_skewX,[eased_skewX_raw[j]])
        append_to_transform_value_arr('shear_y_atan', eased_skewY,[eased_skewY_raw[j]])

    curr_svg_shape['z_animate'].setAttribute('values', ";".join(eased_z))
    curr_svg_shape['z_animate'].setAttribute('dur', str(len(eased_z) / frame_rate))

    curr_svg_shape['display_animate'].setAttribute('values', ";".join(eased_display))
    curr_svg_shape['display_animate'].setAttribute('dur', str(len(eased_display) / frame_rate))
    
    curr_svg_shape['translate_transform'].setAttribute('values', ";".join(eased_translation))
    curr_svg_shape['translate_transform'].setAttribute('dur', str(len(eased_translation) / frame_rate))

    curr_svg_shape['scale_transform'].setAttribute('values', ";".join(eased_scale))
    curr_svg_shape['scale_transform'].setAttribute('dur', str(len(eased_scale) / frame_rate))

    curr_svg_shape['rotate_transform'].setAttribute('values', ";".join(eased_rotate))
    curr_svg_shape['rotate_transform'].setAttribute('dur', str(len(eased_rotate) / frame_rate))

    curr_svg_shape['skewX_transform'].setAttribute('values', ";".join(eased_skewX))
    curr_svg_shape['skewX_transform'].setAttribute('dur', str(len(eased_skewX) / frame_rate))

    curr_svg_shape['skewY_transform'].setAttribute('values', ";".join(eased_skewY))
    curr_svg_shape['skewY_transform'].setAttribute('dur', str(len(eased_skewY) / frame_rate))


def interpolate_frame_values(original_values, eased_frames, type, no_offset=False):
    res = []
    start_frame_id = int(eased_frames[0]) # might not be neccessary but leaving it here for now
    if no_offset == True:
        start_frame_id = 0

    for i in range(len(eased_frames)):
        curr_frame = float(eased_frames[i])

        # offset by starting frame
        prev_frame = math.floor(curr_frame) - start_frame_id
        next_frame = math.ceil(curr_frame) - start_frame_id
        prev_frame = min(prev_frame, len(original_values) - 1)
        next_frame = min(next_frame, len(original_values) - 1)
        # if prev_frame >= len(original_values) or next_frame >= len(original_values):
        #     break

        # denom = next_frame - prev_frame
        # if denom == 0:
        #     alpha = 0
        # else:
        #     alpha = (curr_frame - start_frame_id - prev_frame) / \
        #         (next_frame - prev_frame)
        alpha = 0

        if type == 'translate' or type == 'scale':
            prev_v1 = float(original_values[prev_frame][0])
            next_v1 = float(original_values[next_frame][0])
            v1 = lerp(alpha, prev_v1, next_v1)

            prev_v2 = float(original_values[prev_frame][1])
            next_v2 = float(original_values[next_frame][1])
            v2 = lerp(alpha, prev_v2, next_v2)
            res.append((v1, v2))
            
        elif type == 'display':
            prev_v = original_values[prev_frame]
            next_v = original_values[next_frame]
            if prev_v == 'inline' or next_v == 'inline':
                res.append('inline')
            else:
                res.append('none')
            
        elif type == 'rotate':
            prev_v = float(original_values[prev_frame])
            next_v = float(original_values[next_frame])
            v = prev_v
            if alpha > 0.5:
                v = next_v
            res.append(v)
        
        else:
            prev_v = float(original_values[prev_frame])
            next_v = float(original_values[next_frame])
            v = lerp(alpha, prev_v, next_v)
            res.append(v)

    return res


def set_animate_values(animate, values):
    '''Set the values in an animate attribute with an array of numbers.'''
    animate.setAttribute("values", values)


def get_obj_id(obj):
    return obj['group'].getAttribute('id')


def get_obj_png(obj):
    base64_str = obj['image'].getAttribute('href')[len('data:image/png;base64,'):]
    img_pil = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    img = np.array(img_pil)
    img[:, :, :3] = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    return img


def get_obj_size(obj):
    width = int(obj['image'].getAttribute('width'))
    height = int(obj['image'].getAttribute('height'))
    return [width, height]


######## SVG Parsing ########

def getAnimateTransformById(dom, id):
    transforms = dom.getElementsByTagName('animateTransform')
    for transform in transforms:
        if transform.getAttribute("id") == id:
            return transform

    return None

def getAnimateById(dom, id):
    animates = dom.getElementsByTagName('animate')
    for animate in animates:
        if animate.getAttribute("id") == id:
            return animate

    return None
    
def getSVGAnimationMetadata(dom):
    svg_element = dom.getElementsByTagName('svg')[0]
    return int(svg_element.getAttribute("data-duration")), int(svg_element.getAttribute("data-frame-rate"))

def setSVGAnimationMetadata(dom, dur, frame_rate):
    svg_element = dom.getElementsByTagName('svg')[0]
    svg_element.setAttribute("data-duration", str(dur))
    svg_element.setAttribute("data-frame-rate", str(frame_rate))

def getSVGSize(dom):
    svg_element = dom.getElementsByTagName('svg')[0]
    return int(svg_element.getAttribute("width")), int(svg_element.getAttribute("height"))

def parseValuesFromAnimateTransform(animateTransform, frame_sizes=[0,0]):
    values = animateTransform.getAttribute("values").split(';')
    type = animateTransform.getAttribute("type")
    if type == 'translate':
        # values = [(float(x.split(' ')[0]) / frame_sizes[0],
        #            float(x.split(' ')[1]) / frame_sizes[1]) for x in values]
        values = [(float(x.split(' ')[0]),
                   float(x.split(' ')[1])) for x in values]

    elif type == 'scale':
        values = [(float(x.split(' ')[0]), float(x.split(' ')[1]))
                  for x in values]
    else:
        values = [float(x) for x in values]
    
    return values

def parseValuesFromAnimate(animate):
    return animate.getAttribute("values").split(';')


def getBackgroundImg(dom):
    bg = {}
    
    # group elements
    groups = dom.getElementsByTagName('g')
    for group in groups:
        id = group.getAttribute('id')
        if id == 'shape_bg':
            # create a dict if not already there. if there, append
            bg["group"] = group
            image_element =group.getElementsByTagName("image")[0]
            bg["image"] = image_element
            break
    z_animate = getAnimateById(dom, 'bg_z-index')
    bg['z_animate'] = z_animate
    t_transform = getAnimateTransformById(dom, 'translate')
    bg['translate_transform'] = t_transform
    return bg


def parseSVGInfo(dom, frame_sizes=None):
    shapes = {}
    
    # group elements
    groups = dom.getElementsByTagName('g')
    for group in groups:
        id = group.getAttribute('id')
        if id != None and id.split('_')[0] == 'shape' and id.split('_')[1] != 'bg':
            key = id.split('_')[1]
            # create a dict if not already there. if there, append
            shapes.setdefault(key, {})["group"] = group
            
            image_element =group.getElementsByTagName("image")[0]
            shapes.setdefault(key, {})["image"] = image_element

    # values
    for key in shapes:
        
        z_animate = getAnimateById(dom, str(key) + '_z-index')
        shapes.setdefault(key, {})['z_animate'] = z_animate
        
        d_animate = getAnimateById(dom, str(key) + '_display')
        shapes.setdefault(key, {})['display_animate'] = d_animate       
        
        t_transform = getAnimateTransformById(dom, '_'.join([str(key), 'translate']))
        shapes.setdefault(key, {})['translate_transform'] = t_transform
        
        s_transform = getAnimateTransformById(dom, '_'.join([str(key), 'scale']))
        shapes.setdefault(key, {})['scale_transform'] = s_transform
        
        r_transform = getAnimateTransformById(dom, '_'.join([str(key), 'rotate']))
        shapes.setdefault(key, {})['rotate_transform'] = r_transform
        
        kx_transform = getAnimateTransformById(dom, '_'.join([str(key), 'skewX']))
        shapes.setdefault(key, {})['skewX_transform'] = kx_transform

        ky_transform = getAnimateTransformById(dom, '_'.join([str(key), 'skewY']))
        shapes.setdefault(key, {})['skewY_transform'] = ky_transform
    
    return shapes


def parse_retime_config(retime_config, i):
    str_i = str(i)
    if str_i in retime_config:
        return retime_config[str_i]
    
    return retime_config['-1']


# def parseAnimateValues(dom, shape_id):
#     animate_id = str(shape_id) + "_display"
#     animate = getAnimateById(dom, animate_id)
#     key_times = animate.getAttribute("keyTimes").split(';')
#     values = animate.getAttribute("values").split(';')
#     key_times = [float(x) for x in key_times]

#     return animate, values, key_times

# def parseTransformValues(dom, shape_id, transform_type, frame_sizes):
#     transform_id = str(shape_id) + "_" + transform_type
#     animateTransform = getAnimateTransformById(dom, transform_id)
#     values = animateTransform.getAttribute("values").split(';')

#     if transform_type == 'translate':
#         values = [(float(x.split(' ')[0]) / frame_sizes[0], float(x.split(' ')[1]) / frame_sizes[1]) for x in values]

#     elif transform_type == 'scale':
#         values = [(float(x.split(' ')[0]), float(x.split(' ')[1]))
#                   for x in values]
#     else:
#         values = [float(x) for x in values]
#     return animateTransform

######## Edit SVGs ##########
def update_shape_g_id(group_node, new_id_number):
    group_node.setAttribute('id', '_'.join(['shape', str(new_id_number)]))

def update_animate_ids(curr_svg_shape, new_id_number):
    for key in utils_animate_keys:
        if curr_svg_shape[key]:
            curr_id = curr_svg_shape[key].getAttribute('id').split('_')
            new_id = '_'.join([str(new_id_number), curr_id[1]])
            curr_svg_shape[key].setAttribute('id', new_id)

def replace_img(curr_svg_shape, img_file_path):
    curr_svg_shape['image'].setAttribute(
        'href', convert_image_to_base64(img_file_path))

def hide_shape(curr_svg_shape):
    values = curr_svg_shape['display_animate'].getAttribute(
        'values').split(';')
    new_values = ['none' for i in range(len(values))]
    curr_svg_shape['display_animate'].setAttribute(
        'values', ';'.join(new_values))

def replace_rotation(curr_svg_shape, start_frame:int, end_frame:int, start_degree:float, end_degree:float):
    rotate_values = parseValuesFromAnimateTransform(curr_svg_shape['rotate_transform'])
    new_rotate_values_raw = np.linspace(start_degree, end_degree, end_frame-start_frame+1)
    for i in range(start_frame, end_frame+1):
        rotate_values[i] = new_rotate_values_raw[i-start_frame]
    
    new_rotate_values = []
    for i in range(len(rotate_values)):
        append_to_transform_value_arr('rotate_degree', new_rotate_values, [rotate_values[i]])

    curr_svg_shape['rotate_transform'].setAttribute('values', ";".join(new_rotate_values))

def trim_time(curr_svg_shape, start_index, end_index, frame_rate, svg_dom):    
    for key in utils_animate_keys:
        values = curr_svg_shape[key].getAttribute('values').split(';')
        values = values[start_index:end_index+1]
        curr_svg_shape[key].setAttribute('values', ';'.join(values))
        curr_svg_shape[key].setAttribute('dur', str(len(values) / frame_rate))

    # metadata
    setSVGAnimationMetadata(svg_dom, len(values), frame_rate)

def pad_time(curr_svg_shape, prev_padding_duration, after_padding_duration, prev_vis: str, after_vis: str, frame_rate, svg_dom):
    for key in utils_animate_keys:
        curr_values = np.array(curr_svg_shape[key].getAttribute('values').split(';'))
        prev_padding_values = np.repeat(curr_values[0], prev_padding_duration)
        after_padding_values = np.repeat(curr_values[-1], after_padding_duration)
        
        if key == 'display_animate':
            prev_padding_values = np.repeat(prev_vis, prev_padding_duration)
            after_padding_values = np.repeat(after_vis, after_padding_duration)
            
        new_values = np.concatenate((prev_padding_values, curr_values, after_padding_values))
        curr_svg_shape[key].setAttribute('values', ';'.join(new_values))
        curr_svg_shape[key].setAttribute('dur', str(len(new_values) / frame_rate))
        
        # not sure
        setSVGAnimationMetadata(svg_dom, len(new_values), frame_rate)                 


######## Aspect Ratio ########
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


######## Might not need ########

def generate_display_key_times(shape_gloabal_start_frame, shape_animation_duration, global_animation_duration, frame_rate):
    keyTimes = []
    keyValues = []

    # frames to seconds
    shape_gloabal_start_index = shape_gloabal_start_frame - 1
    shape_gloabal_start_time = shape_gloabal_start_index / frame_rate
    global_animation_time = global_animation_duration / frame_rate

    is_start_delayed = shape_gloabal_start_index > 0
    is_end_early = (shape_gloabal_start_index +
                    shape_animation_duration) < global_animation_duration

    time = ((shape_gloabal_start_frame + shape_animation_duration -
            1) / frame_rate) / global_animation_time

    if is_start_delayed:
        keyTimes.append(str(0))
        keyValues.append('none')

        keyTimes.append(str(shape_gloabal_start_time / global_animation_time))
        keyValues.append('inline')

    else:  # if start at 0
        keyTimes.append(str(0))
        keyValues.append('inline')

    if is_end_early:
        keyTimes.append(str(time))
        keyValues.append('none')

    return keyValues, keyTimes


def update_animate_display(curr_svg_shape, new_shape_global_start_frame, new_shape_global_end_frame, global_animation_frames, frame_rate):
    new_shape_frames = new_shape_global_end_frame - new_shape_global_start_frame + 1

    # if new end frame is longer
    if (new_shape_global_end_frame > global_animation_frames):
        global_animation_frames = new_shape_global_end_frame
        new_global_animation_dur = str(
            global_animation_frames / frame_rate) + "s"
        curr_svg_shape["display_animate"].setAttribute(
            'dur', new_global_animation_dur)
        # print("new global animation frames: {}".format(global_animation_frames))

    new_values, new_key_times = generate_display_key_times(
        new_shape_global_start_frame, new_shape_frames, global_animation_frames, frame_rate)

    curr_svg_shape["display_animate"].setAttribute(
        'keyTimes', ";".join(new_key_times))
    curr_svg_shape["display_animate"].setAttribute(
        'values', ";".join(new_values))

    new_key_times = [float(x) for x in new_key_times]
    curr_svg_shape["display_key"] = new_key_times
    curr_svg_shape["display_values"] = new_values

    curr_svg_shape['group'].setAttribute(
        'data-start-frame', str(new_shape_global_start_frame))
    curr_svg_shape['group'].setAttribute(
        'data-end-frame', str(new_shape_global_end_frame))


def update_all_animate_display(svg_object, new_global_animation_frames, frame_rate):
    for i in range(len(svg_object)):
        curr_svg_shape = svg_object[str(i)]
        curr_shape_global_start_frame = int(
            curr_svg_shape['group'].getAttribute('data-start-frame'))
        curr_shape_global_end_frame = int(
            curr_svg_shape['group'].getAttribute('data-end-frame'))
        update_animate_display(curr_svg_shape, curr_shape_global_start_frame,
                               curr_shape_global_end_frame, new_global_animation_frames, frame_rate)
        new_global_animation_dur = str(
            new_global_animation_frames / frame_rate) + "s"
        curr_svg_shape["display_animate"].setAttribute(
            'dur', new_global_animation_dur)


def update_animateTransform_delay(curr_svg_shape, new_shape_global_start_frame, frame_rate):
    new_shape_global_start_index = new_shape_global_start_frame - 1
    new_start_delay_dur = str(new_shape_global_start_index / frame_rate) + "s"

    curr_svg_shape['translate_transform'].setAttribute(
        'begin', new_start_delay_dur)
    curr_svg_shape['scale_transform'].setAttribute(
        'begin', new_start_delay_dur)
    curr_svg_shape['rotate_transform'].setAttribute(
        'begin', new_start_delay_dur)
    curr_svg_shape['skewX_transform'].setAttribute(
        'begin', new_start_delay_dur)
    curr_svg_shape['skewY_transform'].setAttribute(
        'begin', new_start_delay_dur)


def find_last_global_end_frame(svg_object):
    new_global_animation_frames = 0
    for i in range(len(svg_object)):
        curr_shape_global_end_frame = int(
            svg_object[str(i)]['group'].getAttribute('data-end-frame'))
        if (curr_shape_global_end_frame > new_global_animation_frames):
            new_global_animation_frames = curr_shape_global_end_frame

    return new_global_animation_frames
