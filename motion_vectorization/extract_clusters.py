import argparse
import os
import json
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')
from scipy import stats
import torch
from tqdm import tqdm
from skimage.feature import canny
from scipy import signal

from .utils import *
from .visualizer import Visualizer

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser()
# Video and directory information.
parser.add_argument(
  '--video_file', type=str, required=True, 
  help='Name of the video to process.')
parser.add_argument(
  '--video_dir', default='videos', 
  help='Directory containing videos.')
parser.add_argument(
  '--output_dir', default='outputs', type=str, 
  help='Directory to save outputs.')
parser.add_argument(
  '--start_frame', default=1, type=int, 
  help='The frame to start at.')
parser.add_argument(
  '--max_frames', default=-1, type=int, 
  help='The maximum number of frames to process. If set to -1, then process all frames.')
parser.add_argument(
  '--bg_file', type=str, default=None, 
  help='Background file.')
parser.add_argument(
  '--bg_thresh', type=float, default=0.1, 
  help='Threshold RGB distance to be counted as a background pixel.')
parser.add_argument(
  '--lo_thresh', type=float, default=30, 
  help='Threshold RGB distance to be counted as a background pixel.')
parser.add_argument(
  '--hi_thresh', type=float, default=100, 
  help='Threshold RGB distance to be counted as a background pixel.')
parser.add_argument(
  '--max_radius', type=int, default=3, 
  help='The radius of the largest trapped ball.')
parser.add_argument(
  '--add_contrast', action='store_true', default=False, 
  help='If true, we add contast to the L channel.')
parser.add_argument(
  '--link', action='store_true', default=False, 
  help='If true, link broken edges.')
parser.add_argument(
  '--min_cluster_size', default=50, type=int, 
  help='The minimum number of samples allowed in a cluster.')
parser.add_argument(
  '--min_density', default=0.15, type=int, 
  help='The minimum number of samples allowed in a cluster.')
parser.add_argument(
  '--min_dim', default=5, type=int, 
  help='The minimum dimension of a valid cluster.')

parser.add_argument(
  '--config', type=str, default=None, 
  help='Config file.')
arg = parser.parse_args()

arg = parser.parse_args()
video_name = os.path.splitext(arg.video_file.split('/')[-1])[0]
if arg.config is not None:
  configs_file = arg.config

  if not os.path.exists(configs_file):
    print('[WARNING] Configs file not found! Using default.json instead.')
    configs_file = 'motion_vectorization/config/default.json'

  configs = json.load(open(configs_file, 'r'))
  parser.set_defaults(**configs)
  arg = parser.parse_args()    
print('Configs:')
for arg_name, arg_val in vars(arg).items():
  print(f'  {arg_name}:\t{arg_val}')

def main():
  # Read folders.
  frame_folder = os.path.join(arg.video_dir, video_name, 'rgb')
  frame_idxs = get_numbers(frame_folder)

  # Create output directories.
  if not os.path.exists(arg.output_dir):
    os.makedirs(arg.output_dir)
  labels_folder = os.path.join(arg.video_dir, video_name, 'labels')
  comps_folder = os.path.join(arg.video_dir, video_name, 'comps')
  fgbg_folder = os.path.join(arg.video_dir, video_name, 'fgbg')
  edges_folder = os.path.join(arg.video_dir, video_name, 'edges')
  clusters_folder = os.path.join(arg.video_dir, video_name, 'clusters')
  for folder in [
    labels_folder,
    comps_folder,
    fgbg_folder,
    edges_folder,
    clusters_folder
  ]:
    if not os.path.exists(folder):
      os.makedirs(folder)

  # Take first frame.
  frame_idx = frame_idxs[0]
  frame = cv2.imread(os.path.join(frame_folder, f'{frame_idx:03d}.png'))

  for t in tqdm(range(0, min(len(frame_idxs), arg.max_frames))):
    if t < arg.start_frame:
      continue
    frame_idx = frame_idxs[t]
    if os.path.exists(os.path.join(frame_folder, f'{frame_idx:03d}.png')):
      frame = cv2.imread(os.path.join(frame_folder, f'{frame_idx:03d}.png'))
    else:
      break

    # Get foreground clusters in current frame.
    frame_blurred = frame.copy()
    for i in range(1):
      frame_blurred = cv2.bilateralFilter(frame, 9, 25, 25)
    frame_blurred_lab = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2LAB)
    frame_lab = cv2.cvtColor(cv2.bilateralFilter(frame, 9, 25, 25), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(frame_lab)
    # Applying CLAHE to L-channel
    if arg.add_contrast:
      clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
      l = clahe.apply(l)
    fg_bg = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    if arg.bg_file is not None:
      background = cv2.cvtColor(cv2.imread(arg.bg_file), cv2.COLOR_BGR2LAB) / 255.0
      fg_bg = np.where(np.linalg.norm(frame_lab / 255.0 - background, axis=-1) > arg.bg_thresh, fg_bg, 0)
      kernel = np.ones((3, 3), dtype=np.uint8)
      fg_bg = cv2.morphologyEx(fg_bg, cv2.MORPH_OPEN, kernel)
    else:
      lab_mode, _ = np.array(stats.mode(np.reshape(frame_blurred_lab, (-1, 3)), axis=0))
      fg_bg = np.where(np.linalg.norm(frame_lab / 255.0 - lab_mode / 255.0, axis=-1) > arg.bg_thresh, fg_bg, 0)
    edges_l = np.uint8(canny(l, low_threshold=arg.lo_thresh, high_threshold=arg.hi_thresh))
    edges_a = np.uint8(canny(a, low_threshold=arg.lo_thresh, high_threshold=arg.hi_thresh))
    edges_b = np.uint8(canny(b, low_threshold=arg.lo_thresh, high_threshold=arg.hi_thresh))
    edges = np.max(np.stack([edges_l, edges_a, edges_b]), axis=0)
    # Link broken edges.
    if arg.link:
      for kernel in [
        np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8), 
        np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.uint8), 
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8), 
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.uint8)
      ]:
        links = signal.convolve2d(edges, kernel, mode='same')
        links = np.uint8(links==2)
        edges = np.maximum(edges, links)
    cv2.imwrite(os.path.join(edges_folder, f'{frame_idx:03d}.png'), 255 * edges)

    # BEGIN CANNY EDGE CLUSTERING
    fg_labels, fg_labels_vis = compute_clusters_floodfill(fg_bg, edges, max_radius=arg.max_radius, min_cluster_size=arg.min_cluster_size, min_density=arg.min_density, min_dim=arg.min_dim)
    fg_bg[fg_labels<0] = 0
    # END CANNY EDGE CLUSTERING
    _, fg_comps = cv2.connectedComponents(fg_bg)
    # BEGIN CONNECTED COMPS CLUSTERING
    # for comp_idx in np.unique(fg_comps)[1:]:
    #   comp_size = np.sum(fg_comps==comp_idx)
    #   if comp_size < arg.min_cluster_size:
    #     fg_comps[fg_comps==comp_idx] = 0
    # fg_comps = fg_comps - 1
    # # Compress ord of labels.
    # idx = -1
    # for l in np.unique(fg_comps):
    #   fg_comps[fg_comps==l] = idx
    #   idx += 1
    # fg_bg[fg_comps<0] = 0
    # fg_labels_vis = Visualizer.show_labels(fg_comps + 1)
    # END CONNECTED COMPS CLUSTERING

    np.save(os.path.join(labels_folder, f'{frame_idx:03d}.npy'), fg_labels) # fg_labels for CE and fg_comps for COMP
    np.save(os.path.join(comps_folder, f'{frame_idx:03d}.npy'), fg_comps)
    np.save(os.path.join(fgbg_folder, f'{frame_idx:03d}.npy'), fg_bg)
    cv2.imwrite(os.path.join(clusters_folder, f'{frame_idx:03d}.png'), fg_labels_vis)

if __name__ == '__main__':
  main()
