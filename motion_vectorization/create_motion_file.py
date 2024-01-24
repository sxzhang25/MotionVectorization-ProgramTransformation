import numpy as np
import os
import argparse
import pickle
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from .utils import get_numbers


parser = argparse.ArgumentParser()
parser.add_argument(
  '--video_name', required=True, type=str, 
  help='Name of video to process.')
parser.add_argument(
  '--video_dir', default='videos', type=str,
  help='Directory containing video data.')
parser.add_argument(
  '--output_dir', default='outputs', type=str, 
  help='Directory to save outputs.')
parser.add_argument(
  '--suffix', default=None, type=str, 
  help='Suffix for output video names.')
parser.add_argument(
  '--bg_file', type=str, default=None, 
  help='Background file.')
parser.add_argument(
  '--skip_figs', action='store_true', default=False, 
  help='Skip generating figures.')
parser.add_argument(
  '--config', type=str, default=None, 
  help='Config file.')
arg = parser.parse_args()


if arg.config is not None:
  configs_file = arg.config

  if not os.path.exists(configs_file):
    print('[WARNING] Configs file not found! Using default.json instead.')
    configs_file = 'motion_vectorization/config/default.json'

  configs = json.load(open(configs_file, 'r'))
  parser.set_defaults(**configs)
  arg = parser.parse_args()


def main():
  # Create output directories.
  shapes_folder = os.path.join(arg.output_dir, f'{arg.video_name}_{arg.suffix}', 'shapes')
  print('shapes_folder:', shapes_folder)
  frame_folder = os.path.join(arg.video_dir, arg.video_name, 'rgb')
  frame_idxs = get_numbers(frame_folder)

  # Read all shapes.
  shapes = {}
  for filename in os.listdir(shapes_folder):
    if os.path.splitext(filename)[1] == '.png':
      shape_idx = int(os.path.splitext(filename)[0])
      shape = cv2.imread(os.path.join(shapes_folder, filename), cv2.IMREAD_UNCHANGED)
      shapes[shape_idx] = shape
  print('\nShapes:', shapes.keys())

  # shape_bank_name = 'shape_bank.pkl' if arg.suffix is None else f'shape_bank_{arg.suffix}.pkl'
  shape_bank = pickle.load(open(os.path.join(arg.output_dir, f'{arg.video_name}_{arg.suffix}', 'shape_bank.pkl'), 'rb'))

  # Take first frame.
  frame = cv2.imread(os.path.join(arg.video_dir, arg.video_name, 'rgb', '001.png'))

  # Define the codec and create VideoWriter object.
  w = frame.shape[1]
  h = frame.shape[0]

  motion_file = {}
  max_frame = 0
  if -1 in shape_bank:
    motion_file[-1] = {
      'name': arg.video_name,
      'width': w,
      'height': h,
      'bg_color': [c.tolist() for c in shape_bank[-1]],
      'bg_img': arg.bg_file,
      'time': [int(i) for i in frame_idxs],
    }
  for i, shape_idx in tqdm(enumerate(shape_bank)):
    shape_idx = int(shape_idx)
    if shape_idx < 0:
      continue
    shape_info = shape_bank[shape_idx]
    # print(shape_idx, len(shape_info))
    # if len(shape_info) == 0:
    #   continue
    # for shape_idx_info in shape_info:
    #   print(shape_idx_info['t'], '/', len(frame_idxs))
    shape_path = os.path.join(shapes_folder, f'{shape_idx}.png')
    shape = cv2.imread(shape_path, cv2.IMREAD_UNCHANGED)
    shape_data = {
      'shape': shape_path,
      'size': (shape.shape[1], shape.shape[0]),
      'centroid': shape_info[0]['centroid'],
      'time': [int(frame_idxs[shape_idx_info['t']]) for shape_idx_info in shape_info],
      'sx': [shape_idx_info['h'][0] for shape_idx_info in shape_info],
      'sy': [shape_idx_info['h'][1] for shape_idx_info in shape_info],
      'cx': [shape_idx_info['centroid'][0] / w for shape_idx_info in shape_info],
      'cy': [shape_idx_info['centroid'][1] / h for shape_idx_info in shape_info],
      'theta': [shape_idx_info['h'][4] for shape_idx_info in shape_info],
      'kx': [shape_idx_info['h'][5] for shape_idx_info in shape_info],
      'ky': [shape_idx_info['h'][6] for shape_idx_info in shape_info],
      'z': [float(shape_idx_info['h'][7]) for shape_idx_info in shape_info]
    }
    # last_idx = motion_file[-1]['time'].index(shape_data['time'][-1])
    # if last_idx < len(motion_file[-1]['time']) - 1:
    #   shape_data['time'].append(motion_file[-1]['time'][last_idx + 1])
    #   shape_data['sx'].append(shape_data['sx'][-1])
    #   shape_data['sy'].append(shape_data['sy'][-1])
    #   shape_data['cx'].append(shape_data['cx'][-1])
    #   shape_data['cy'].append(shape_data['cy'][-1])
    #   shape_data['theta'].append(shape_data['theta'][-1])
    #   shape_data['kx'].append(shape_data['kx'][-1])
    #   shape_data['ky'].append(shape_data['ky'][-1])
    #   shape_data['z'].append(shape_data['z'][-1])
    motion_file[shape_idx] = shape_data
    max_frame = max(max_frame, frame_idxs[shape_info[-1]['t']])
    
  with open(os.path.join(arg.output_dir, f'{arg.video_name}_{arg.suffix}', 'motion_file.json'), 'w', encoding='utf-8') as handle:
    json.dump(motion_file, handle, ensure_ascii=False, indent=4)
  
  # Create figures.
  # if not arg.skip_figs:
  #   for shape_idx in motion_file:
  #     if shape_idx == -1:
  #       continue
      
  #     img = shapes[shape_idx]
  #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #     # t0 = motion_file[shape_idx]["time"][0]
  #     # t1 = motion_file[shape_idx]["time"][-1]

  #     fig, ax = plt.subplots(9, figsize=(6, 15))
  #     ax[0].imshow(img)
  #     ax[0].axis('off')
  #     ax[0].set_title(shape_idx)

  #     ax[1].plot(motion_file[shape_idx]["time"], [x * w for x in motion_file[shape_idx]["cx"]], c='r')
  #     ax[2].plot(motion_file[shape_idx]["time"], [y * h for y in motion_file[shape_idx]["cy"]], c='r')
  #     ax[3].plot(motion_file[shape_idx]["time"], motion_file[shape_idx]["sx"], c='r')
  #     ax[4].plot(motion_file[shape_idx]["time"], motion_file[shape_idx]["sy"], c='r')
  #     ax[5].plot(motion_file[shape_idx]["time"], [np.rad2deg(t) for t in motion_file[shape_idx]["theta"]], c='r')
  #     ax[6].plot(motion_file[shape_idx]["time"], [t for t in motion_file[shape_idx]["kx"]], c='r')
  #     ax[7].plot(motion_file[shape_idx]["time"], [t for t in motion_file[shape_idx]["ky"]], c='r')
  #     ax[8].scatter(motion_file[shape_idx]["time"], motion_file[shape_idx]["z"], c='r')
      
  #     ax[1].set_title('cx')
  #     ax[1].set(xlabel='time', ylabel='pixels')

  #     ax[2].set_title('cy')
  #     ax[2].set(xlabel='time', ylabel='pixels')

  #     ax[3].set_title('sx')
  #     ax[3].set(xlabel='time', ylabel='scale')

  #     ax[4].set_title('sy')
  #     ax[4].set(xlabel='time', ylabel='scale')

  #     ax[5].set_title('theta')
  #     ax[5].set(xlabel='time', ylabel='deg')

  #     ax[6].set_title('kx')
  #     ax[6].set(xlabel='time', ylabel='shear')

  #     ax[7].set_title('ky')
  #     ax[7].set(xlabel='time', ylabel='shear')

  #     ax[8].set_title('z-index')
  #     ax[8].set(xlabel='time', ylabel='z')

  #     plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.75, wspace=0.4)
  #     plt.savefig(os.path.join(arg.output_dir, f'{arg.video_name}_{arg.suffix}', f'shape_{shape_idx}_info.png'))
  #     plt.close()


if __name__ == '__main__':
  main()