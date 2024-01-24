import argparse
import collections
import os
import json
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import time
import datetime
from scipy import ndimage
from skimage.transform import AffineTransform
from skimage.measure import ransac
import torch.nn.functional as F
import networkx as nx

from . import compositing
from .utils import *
from .dataloader import DataLoader


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
  '--suffix', default=None, type=str, 
  help='Suffix for output video names.')
parser.add_argument(
  '--config', type=str, default=None, 
  help='Config file.'
)
parser.add_argument(
  '--date', default=None, type=str, 
  help='Date of experiment')

# Video processing.
parser.add_argument(
  '--frame_rate', default=24, type=int, 
  help='The output frame rate.')
parser.add_argument(
  '--subsample_rate', default=1, type=int, 
  help='Rate at which to subsample video frames.')
parser.add_argument(
  '--max_frames', default=-1, type=int, 
  help='The maximum number of frames to process. If set to -1, then process all frames.')
parser.add_argument(
  '--start_frame', default=1, type=int, 
  help='The frame to start at.')

# Clustering.
parser.add_argument(
  '--scale', default=0.3, type=float, 
  help='The scale factor to multiply the frames (for computational purposes).')
parser.add_argument(
  '--bg_thresh', type=float, default=0.1, 
  help='Threshold RGB distance to be counted as a background pixel.')
parser.add_argument(
  '--min_cluster_size', default=50, type=int, 
  help='The minimum number of samples allowed in a cluster.')
parser.add_argument(
  '--median_radius', '-mr', default=5, type=int, 
  help='The radius for median blurring.')
parser.add_argument(
  '--cluster_eps', default=5, type=int, 
  help='DBSCAN eps.')
parser.add_argument(
  '--cluster_min_samples', default=10, type=int, 
  help='DBSCAN min samples.')
parser.add_argument(
  '--bg_file', type=str, default=None, 
  help='Background file.')

# Shape matching.
parser.add_argument(
  '--init_weight', default=0.0, type=float, 
  help='The weight for rotation deviation during initialization.')
parser.add_argument(
  '--init_thresh', default=0.01, type=float, 
  help='The RGB difference threshold for rotation initialization.')
parser.add_argument(
  '--num_match_shapes', default=5, type=int, 
  help='The number of closest shapes to check for matches.')
parser.add_argument(
  '--match_radius', default=120.0, type=float, 
  help='Maximum distance to be check as a match.')
parser.add_argument(
  '--main_match_thresh', default=0.5, type=float, 
  help='Threshold to be considered as a main match.')
parser.add_argument(
  '--fallback_match_thresh', default=0.5, type=float, 
  help='Threshold to be considered as a fallback match.')
parser.add_argument(
  '--match_method', default='hu', type=str, 
  help='Shape matching similarity method.')

# Optimization.
parser.add_argument(
  '--min_opt_size', default=200, type=int, 
  help='Size of optimization.')
parser.add_argument(
  '--blur_kernel', default=3, type=int, 
  help='Blur kernel size for visual loss.')
parser.add_argument(
  '--lr', default=0.01, type=float, 
  help='Learning rate for optimization.')
parser.add_argument(
  '--p_weight', default=0.1, type=float, 
  help='Weight for params loss.')
parser.add_argument(
  '--use_z', action='store_true', default=True, 
  help='If true, use z-layering.')
parser.add_argument(
  '--n_steps', default=50, type=int, 
  help='Number of steps to optimize for individual shape matching.')
parser.add_argument(
  '--use_gpu', action='store_true', default=False, 
  help='Use GPU')
parser.add_argument(
  '--bleed', default=50, type=int, 
  help='Margin outside of frame.')
parser.add_argument(
  '--all_joint', action='store_true', default=False,
  help='Optimize all shapes jointly.')
parser.add_argument(
  '--overlap_thresh', type=float, default=0.9, 
  help='Threshold definition of containment.')
parser.add_argument(
  '--manual_init', action='store_true', default=False,
  help='Use template matching for initialization.')
parser.add_argument(
  '--use_k', action='store_true', default=False, 
  help='Use skew.')
parser.add_argument(
  '--use_r', action='store_true', default=False, 
  help='Use rotation.')
parser.add_argument(
  '--use_s', action='store_true', default=False, 
  help='Use scale.')
parser.add_argument(
  '--use_t', action='store_true', default=False, 
  help='Use translation.')
parser.add_argument(
  '--init_r', action='store_true', default=False, 
  help='Initalize rotation.')
parser.add_argument(
  '--init_s', action='store_true', default=False, 
  help='Initalize scale.')
parser.add_argument(
  '--init_t', action='store_true', default=False, 
  help='Initalize translation.')

# Shape processing.
parser.add_argument(
  '--drop_thresh', type=float, default=0.005, 
  help='Threshold on RGB loss to drop a shape.')
parser.add_argument(
  '--snap_thresh', default=3, type=float, 
  help='Snap to 0 for all movements less than this percentage of shape width/height.')
parser.add_argument(
  '--scale_thresh', default=0.03, type=float, 
  help='Snap to 0 for all scales less than this percentage of shape width/height.')
parser.add_argument(
  '--angle_thresh', default=5.0, type=float, 
  help='Snap to 0 for all rotations less than this angle.')

# Debugging.
parser.add_argument(
  '--debug', action='store_true', default=False, 
  help='If true, visualize intermediate outputs.')
parser.add_argument(
  '--verbose', action='store_true', default=False, 
  help='If true, print intermediate messages and outputs.')
parser.add_argument(
  '--show', action='store_true', default=False, 
  help='If true, visualize final outputs.')
parser.add_argument(
  '--show_optim', action='store_true', default=False, 
  help='If true, visualize optimization.')
parser.add_argument(
  '--show_recon', action='store_true', default=False, 
  help='If true, visualize the video reconstruction.')
parser.add_argument(
  '--checkpoint', action='store_true', default=False,
  help='Start from most recently saved checkpoint.'
)

d = datetime.date.today().strftime("%d%m%y")
now = datetime.datetime.now()
print(now)

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

if arg.date is not None:
  d = arg.date

np.random.seed(0)
torch.manual_seed(0)

def main():
  # Use GPU if available.
  device = 'cuda' if (arg.use_gpu and torch.cuda.is_available()) else 'cpu'
  print('device:', device)

  # Read folders.
  frame_folder = os.path.join(arg.video_dir, video_name, 'rgb')
  flow_folder = os.path.join(arg.video_dir, video_name, 'flow', 'forward')
  frame_idxs = get_numbers(frame_folder)

  # Create dataloader.
  video_dir = os.path.join(arg.video_dir, video_name)
  dataloader = DataLoader(video_dir, max_frames=arg.max_frames)

  # Create output directories.
  if not os.path.exists(arg.output_dir):
    os.makedirs(arg.output_dir)
  video_folder = os.path.join(arg.output_dir, f'{video_name}_{arg.suffix}')
  recon_folder = os.path.join(video_folder, 'outputs', 'recon')
  diff_folder = os.path.join(video_folder, 'outputs', 'diff')
  debug_opt_folder = os.path.join(video_folder, 'debug', 'opt_opt')
  debug_targets_folder = os.path.join(video_folder, 'debug', 'opt_targets')
  debug_elements_folder = os.path.join(video_folder, 'debug', 'opt_elements')
  debug_rotate_folder = os.path.join(video_folder, 'debug', 'opt_rotate')
  for folder in [
    video_folder, 
    recon_folder, 
    diff_folder, 
    debug_opt_folder, 
    debug_targets_folder, 
    debug_elements_folder,
    debug_rotate_folder,
  ]:
    if not os.path.exists(folder):
      os.makedirs(folder)

  # Save arguments.
  with open(os.path.join(video_folder, 'arg.pkl'), 'wb') as handle:
    pickle.dump(vars(arg), handle)
  
  # Load tracking information.
  time_bank = pickle.load(open(os.path.join(video_folder, 'time_bank.pkl'), 'rb'))

  # Read all shapes.
  shapes = {}
  for filename in os.listdir(os.path.join(video_folder, 'shapes')):
    shape_idx = int(os.path.splitext(filename)[0])
    shape = cv2.imread(os.path.join(video_folder, 'shapes', filename), cv2.IMREAD_UNCHANGED)
    shapes[shape_idx] = shape
  print('[NOTE] \nShapes:', shapes.keys())
  
  t = arg.start_frame
  frame = cv2.imread(os.path.join(frame_folder, f'{frame_idxs[t]:03d}.png'))
  frame_height, frame_width, _ = frame.shape
  frame_width_s = int(arg.scale * frame_width)
  frame_height_s = int(arg.scale * frame_height)

  # Coordinate grids.
  x = np.linspace(0, frame_width - 1, frame_width)
  y = np.linspace(0, frame_height - 1, frame_height)
  spatial = np.stack(np.meshgrid(x, y), axis=-1)
  x_s = np.linspace(0, frame_width_s - 1, frame_width_s)
  y_s = np.linspace(0, frame_height_s - 1, frame_height_s)

  # Load background job.
  bg_img = None
  if arg.bg_file is not None:
    bg_img = cv2.imread(arg.bg_file)  

  active_shapes = []  # List of shape indices which appeared in the previous frame.
  shape_params = collections.defaultdict(
    lambda: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Shape indices mapped to their params.
  if arg.checkpoint:
    shape_bank = pickle.load(open(os.path.join(video_folder, 'shape_bank.pkl'), 'rb'))
    last_frame = 0
    for shape_idx in shape_bank:
      if shape_idx == -1:
        continue
      last_frame = max(shape_bank[shape_idx][-1]['t'], last_frame)
      arg.start_frame = min(arg.start_frame, last_frame) + 1
  else:
    shape_bank = collections.defaultdict(list)
  optim_err = collections.defaultdict(int)  # Maps shape indices to last optimization error.
  label_mapping = {}  # Maps shape indices in this frame to shape indices across entire video.
  for t, frame_idx in enumerate(dataloader.frame_idxs):
    if t < arg.start_frame:
      continue
    print(f'\nFRAME {frame_idx} ({t}) CLUSTERS\n========================')
    frame_t0 = time.perf_counter()
    frame, _, _, fg_comps, flow, _ = dataloader.load_data(t)
    label_mapping.clear()
    bgr_mode = time_bank['bgr'][t]
    shape_bank[-1].append(bgr_mode)

    # Create background image.
    if arg.bg_file is None:
      bg_img = np.full((frame_height, frame_width, 3), np.array(bgr_mode))

    # Get all shape masks for this frame.
    if len(time_bank['shapes'][t]) < 1:
      fg_layers_bleed = -1 * np.ones((1, frame_height + 2 * arg.bleed, frame_width + 2 * arg.bleed), dtype=np.int32)
    else:
      fg_layers_bleed = []
      for i, active_shape_idx in enumerate(time_bank['shapes'][t]):
        # fg_layer_i = -1 * np.ones((1, frame_height + 2 * arg.bleed, frame_width + 2 * arg.bleed), dtype=np.int32)
        # e_min_x, e_min_y, e_max_x, e_max_y = time_bank['shapes'][t][active_shape_idx]['coords']
        # fg_layer_i[e_min_y + arg.bleed:e_max_y + arg.bleed, e_min_x + arg.bleed:e_max_x + arg.bleed] = time_bank['shapes'][t][active_shape_idx]['shape']
        # fg_layers_bleed.append(fg_layer_i)
        fg_layers_bleed.append(time_bank['shapes'][t][active_shape_idx]['mask'])
      fg_layers_bleed = np.stack(fg_layers_bleed)
    fg_layers = fg_layers_bleed[
      :, arg.bleed:fg_layers_bleed.shape[1] - arg.bleed, arg.bleed:fg_layers_bleed.shape[2] - arg.bleed]
    fg_comp_to_label, _ = get_comp_label_map(fg_comps, np.max(fg_layers, axis=0))
    active_shapes = sorted(list(time_bank['shapes'][t].keys()))
    valid_active_shapes = []
    for idx in active_shapes:
      if idx in shapes:
        valid_active_shapes.append(idx)
    active_shapes = valid_active_shapes
    active_shapes_lidx = {active_shape_idx: i for i, active_shape_idx in enumerate(active_shapes)}
    print('[VARS] Active shapes:', active_shapes)
    print('[VARS] fg_comp_to_label:', fg_comp_to_label)
    if len(active_shapes) > 32:
      device = 'cuda' if (arg.use_gpu and torch.cuda.is_available()) else 'cpu'
      print(f'[NOTE] Switched to cuda for {len(active_shapes)} shapes')
    else:
      device = 'cpu'

    if len(active_shapes) > 0:
      # Get initial shape elements for optimization.
      elements = []
      centroids = []
      tx_init = []
      ty_init = []
      cxs = []
      cys = []
      shape_crops = []
      for active_shape_idx in active_shapes:
        shape_crop = shapes[active_shape_idx]
        shape_crops.append(shape_crop)
        if 'centroid' in time_bank['shapes'][t][active_shape_idx]:
          target_cx, target_cy = time_bank['shapes'][t][active_shape_idx]['centroid']
        else:
          fg_layer_mask = np.max(np.uint8(fg_layers_bleed==active_shape_idx), axis=0)
          target_cx, target_cy = get_shape_centroid(fg_layer_mask)
          target_cx -= arg.bleed
          target_cy -= arg.bleed
        # Be careful to place shapes completely within the bounds of the frame.
        cx = max(shape_crop.shape[1] / 2, min(target_cx, frame.shape[1] - shape_crop.shape[1] / 2))
        cy = max(shape_crop.shape[0] / 2, min(target_cy, frame.shape[0] - shape_crop.shape[0] / 2))
        cxs.append(cx)
        cys.append(cy)
        # Save all the current shape centroids.
        centroids.append([target_cx, target_cy])
        # Also save any translations required to satisfy this constraint so that we can incorporate
        # it into our translation initialization later.
        tx_init.append((target_cx - cx))
        ty_init.append((target_cy - cy))
      elements = compositing.place_shape(
        shape_crops,
        cxs, cys,
        [1.0] * len(shape_crops), # sx 
        [1.0] * len(shape_crops), # sy
        [0.0] * len(shape_crops), # theta
        [0.0] * len(shape_crops), # kx
        [0.0] * len(shape_crops), # ky
        frame_width, 
        frame_height,
        bg=np.tile(np.transpose(bg_img / 255.0, (2, 0, 1)), [len(shape_crops), 1, 1, 1]),
        keep_alpha=True,
        device=device
      )

      # Construct optimization groups.
      G = nx.Graph()
      G.add_nodes_from([f'c{comp_idx}' for comp_idx in list(np.unique(fg_comps)[1:])], bipartite=0)
      G.add_nodes_from([f's{shape_idx}' for shape_idx in active_shapes], bipartite=1)
      for comp_idx in fg_comp_to_label:
        for shape_idx in fg_comp_to_label[comp_idx]:
          if shape_idx in active_shapes:
            G.add_edges_from([(f'c{comp_idx}', f's{shape_idx}')])
      opt_groups = []
      for cc in nx.connected_components(G):
        s_count = 0
        for v in cc:
          if v[0] == 's':
            s_count += 1
        if s_count > 0:
          opt_groups.append(cc)
      
      # Collect all component groups and ground truth.
      print('[NOTE] Collecting component groups...')
      comp_t0 = time.perf_counter()
      target_to_element = collections.defaultdict(list)  # Map element position index to recon position index.
      element_regions = [None] * len(active_shapes)
      element_sizes = [None] * len(active_shapes)
      recon_regions = []
      bg_regions = []
      sx_init = [None] * len(active_shapes)
      sy_init = [None] * len(active_shapes)
      theta_init = [None] * len(active_shapes)
      sx_prev = [None] * len(active_shapes)
      sy_prev = [None] * len(active_shapes)
      theta_prev = [None] * len(active_shapes)
      kx_prev = [None] * len(active_shapes)
      ky_prev = [None] * len(active_shapes)
      z_init = [None] * len(active_shapes)
      target_bounds = []
      centroids_opt = [[0, 0] for i in range(len(active_shapes))]  
      for c, opt_group in enumerate(opt_groups):
        comp_shapes = []
        comp_shape_lidxs = []
        for v in opt_group:
          if v[0] == 's':
            active_shape_idx = int(v[1:])
            # if c in target_to_element:  # DEBUG HERE!
            # print(c, 'in target_to_element:', (c in target_to_element))
            # print(active_shape_idx, 'in active_shapes_lidx:', (active_shape_idx in active_shapes_lidx))
            # if active_shape_idx in active_shapes_lidx:
            target_to_element[c].append(active_shapes_lidx[active_shape_idx])
            comp_shapes.append(active_shape_idx)
            comp_shape_lidxs.append(active_shapes_lidx[active_shape_idx])

        comp_alpha_bin = np.zeros((elements.shape[2], elements.shape[3]), dtype=np.uint8)
        for active_shape_idx in comp_shapes:
          l_mask = np.max(np.uint8(fg_layers==active_shape_idx), axis=0)
          pad_b = max(0, comp_alpha_bin.shape[0] - l_mask.shape[0])
          pad_r = max(0, comp_alpha_bin.shape[1] - l_mask.shape[1])
          l_mask = np.pad(l_mask, ((0, pad_b), (0, pad_r)))
          comp_alpha_bin[l_mask==1] = 1
        comp_alpha = cv2.GaussianBlur(comp_alpha_bin, (3, 3), 0)

        total_alpha = torch.tensor(comp_alpha, dtype=torch.float64, device=elements.device)
        for active_shape_idx in comp_shape_lidxs:
          shape_alpha = (elements[active_shape_idx, 3, :, :])
          pad_b = max(0, total_alpha.shape[0] - shape_alpha.shape[0])
          pad_r = max(0, total_alpha.shape[1] - shape_alpha.shape[1])
          shape_alpha = F.pad(shape_alpha, (0, pad_b, 0, pad_r))
          total_alpha = torch.maximum(total_alpha, shape_alpha)
        r_min_x = max(int(torch.min(torch.where(total_alpha>0)[1])) - 10, 0)
        r_max_x = min(int(torch.max(torch.where(total_alpha>0)[1])) + 10, spatial.shape[1])
        r_min_y = max(int(torch.min(torch.where(total_alpha>0)[0])) - 10, 0)
        r_max_y = min(int(torch.max(torch.where(total_alpha>0)[0])) + 10, spatial.shape[0])
        target_bounds.append([r_min_x, r_min_y, r_max_x, r_max_y])
        pad_b = max(0, comp_alpha.shape[0] - frame.shape[0])
        pad_r = max(0, comp_alpha.shape[1] - frame.shape[1])
        frame_recon = np.pad(frame, ((0, pad_b), (0, pad_r), (0, 0)))
        recon_region = (1 - comp_alpha[..., None]) * bg_img + comp_alpha[..., None] * frame_recon
        recon_region = np.concatenate([recon_region, np.uint8(255 * comp_alpha[..., None])], axis=-1)
        recon_region = recon_region[r_min_y:r_max_y, r_min_x:r_max_x, :]
        recon_regions.append(recon_region)
        bg_regions.append(bg_img[r_min_y:r_max_y, r_min_x:r_max_x])

        for active_shape_idx in comp_shapes:
          lidx = active_shapes_lidx[active_shape_idx]
          shape_region = elements[lidx, :, r_min_y:r_max_y, r_min_x:r_max_x]
          element_regions[lidx] = shape_region
          element_sizes[lidx] = [r_max_x - r_min_x, r_max_y - r_min_y]
          adjust = (max(element_sizes[lidx])) / 2
          tx_init[lidx] /= adjust
          ty_init[lidx] /= adjust
          centroids_opt[lidx][0] = (cxs[lidx] - r_min_x)
          centroids_opt[lidx][1] = (cys[lidx] - r_min_y)
        
        for active_shape_idx in comp_shapes:
          lidx = active_shapes_lidx[active_shape_idx]
          z_init[lidx] = shape_params[active_shape_idx][7]
          sx_prev[lidx] = shape_params[active_shape_idx][0]
          sy_prev[lidx] = shape_params[active_shape_idx][1]
          theta_prev[lidx] = shape_params[active_shape_idx][4]
          kx_prev[lidx] = shape_params[active_shape_idx][5]
          ky_prev[lidx] = shape_params[active_shape_idx][6]
          # If the shape was previously seen, compute an affine transform from the flow. Then add it to
          # the previously computed shape params.
          fallback = optim_err[active_shape_idx] > 0.025 or arg.all_joint
          if arg.verbose:
            if fallback:
              print(f'[NOTE] Prev shape {active_shape_idx} error ({optim_err[active_shape_idx]:.4f}) is too high, reinitializing!')
          if False:#active_shape_idx in shape_bank.keys() and not fallback and not arg.manual_init and t > arg.start_frame:
            xy = np.stack(np.where(prev_fg_layers==active_shape_idx)[1:], axis=1)[:, ::-1]
            if xy.shape[0] < 1:  # This line is to account for the correction gui getting rid of shapes.
              continue
            centroid = np.mean(xy, axis=0)[None, ...]
            disp_xy = xy + flow[xy[:, 1], xy[:, 0]]
            xy = xy - centroid
            disp_xy = disp_xy - centroid
            spacing = xy.shape[0] // min(xy.shape[0], 500)
            idxs = np.arange(0, xy.shape[0], spacing)
            affine_model, inliers = ransac(
              (xy[idxs, :], disp_xy[idxs, :]), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100)
            affine_model.estimate(xy[idxs, :], disp_xy[idxs, :])
            theta = shape_params[active_shape_idx][4] 
            # if arg.init_r:
            #   theta += affine_model.rotation
            curr_crop_alpha = np.max(np.uint8(fg_layers==active_shape_idx), axis=0)
            t_min_x, t_min_y, t_max_x, t_max_y = get_shape_coords(curr_crop_alpha)
            curr_crop_alpha = curr_crop_alpha[t_min_y:t_max_y, t_min_x:t_max_x]
            curr_crop_unrot = ndimage.rotate(curr_crop_alpha, np.rad2deg(theta), order=0)
            t_min_x, t_min_y, t_max_x, t_max_y = get_shape_coords(curr_crop_unrot)
            shape_mask = shapes[active_shape_idx][:, :, 3]
            s_min_x, s_min_y, s_max_x, s_max_y = get_shape_coords(shape_mask)
            sx = (t_max_x - t_min_x) / (s_max_x - s_min_x)
            sy = (t_max_y - t_min_y) / (s_max_y - s_min_y)
            # if arg.init_s:
            #   sx_init[lidx] = sx
            #   sy_init[lidx] = sy
            # else:
            #   sx_init[lidx] = shape_params[active_shape_idx][0]
            #   sy_init[lidx] = shape_params[active_shape_idx][1]
            sx_init[lidx] = shape_params[active_shape_idx][0]
            sy_init[lidx] = shape_params[active_shape_idx][1]
            theta_init[lidx] = theta
          # If the shape has never been seen before (not in shape_bank), use brute force search 
          # initialization.
          else:
            shape_crop = shapes[active_shape_idx] / 255.0
            # e_min_coords, e_max_coords = time_bank['shapes'][t][active_shape_idx]['coords']
            # e_min_x, e_min_y = [e_min_coords[0] + arg.bleed, e_min_coords[1] + arg.bleed]
            # e_max_x, e_max_y = [e_max_coords[0] + arg.bleed, e_max_coords[1] + arg.bleed]
            e_min_x, e_min_y, e_max_x, e_max_y = get_shape_coords(np.max(np.uint8(fg_layers_bleed==active_shape_idx), axis=0))
            target_crop_mask = np.max(np.uint8(fg_layers_bleed==active_shape_idx), axis=0)[..., None]
            frame_pad = np.pad(frame, ((arg.bleed, arg.bleed), (arg.bleed, arg.bleed), (0, 0)))
            target_crop = (frame_pad * target_crop_mask)[e_min_y:e_max_y, e_min_x:e_max_x] / 255.0
            target_crop_mask = target_crop_mask[e_min_y:e_max_y, e_min_x:e_max_x]
            target_crop = np.dstack([target_crop, target_crop_mask])
            p_weight = arg.init_weight if active_shape_idx in shape_bank.keys() else 0.0
            over_mask = np.max(np.uint8(fg_layers_bleed==-2), axis=0)
            over_mask = over_mask[e_min_y:e_max_y, e_min_x:e_max_x]
            theta, sx, sy, rot_vis = init_rot_scale(
              shape_crop, target_crop, shape_params[active_shape_idx][4], bgr_mode / 255.0, over_mask=over_mask, p_weight=arg.init_weight, init_thresh=arg.init_thresh)
            if arg.init_r:
              theta_init[lidx] = np.deg2rad(-theta)
            else:
              theta_init[lidx] = shape_params[active_shape_idx][4]
            if arg.init_s:
              sx_init[lidx] = sx
              sy_init[lidx] = sy
            else:
              sx_init[lidx] = shape_params[active_shape_idx][0]
              sy_init[lidx] = shape_params[active_shape_idx][1]
            cv2.imwrite(os.path.join(debug_rotate_folder, f'{frame_idx:03d}_p{active_shape_idx}.png'), rot_vis)
      comp_t1 = time.perf_counter()
      print(f'[TIME] Collecting comps took {comp_t1 - comp_t0:.2f}s')

      # Do the optimization jointly over all shapes against the current frame.
      print('[NOTE] Optimizing...')
      p_weight = 0.0 if t == 0 else arg.p_weight
      _, params_shape, layer_z, _, render_shape, target_shape, _, _, side_by_side, on_top, losses = optimize(
        element_regions, centroids_opt, recon_regions, target_to_element,
        np.array(sx_init), np.array(sy_init), np.array(theta_init), np.array(tx_init), np.array(ty_init), np.array(z_init), 
        0, 0, bg_regions, use_k=arg.use_k, use_r=arg.use_r, use_s=arg.use_s, use_t=arg.use_t, rgb_weight=10,
        sx_prev=np.array(sx_prev), sy_prev=np.array(sy_prev), theta_prev=np.array(theta_prev), kx_prev=np.array(kx_prev), ky_prev=np.array(ky_prev),
        use_z=arg.use_z, blur_kernel=arg.blur_kernel, lr=arg.lr, n_steps=arg.n_steps, min_size=arg.min_opt_size, p_weight=p_weight, debug=arg.show_optim, device=device)
      
      target_bounds_elements = {}
      for c in target_to_element:
        r_min_x, r_min_y, r_max_x, r_max_y = target_bounds[c]
        for elem in target_to_element[c]:
          target_bounds_elements[elem] = target_bounds[c]
        target_height = r_max_y - r_min_y
        target_width = r_max_x - r_min_x
        target_rgb = target_shape[c] / 255.0
        element_zs = [layer_z[e] for e in target_to_element[c]]
        render_order = np.argsort(element_zs).tolist()
        output_rgb = np.pad(
          bg_regions[c], 
          (
            (0, max(target_width, target_height) - bg_regions[c].shape[0]), 
            (0, max(target_width, target_height) - bg_regions[c].shape[1]), 
            (0, 0)
          )
        ) / 255.0
        for o in render_order:
          shape_alpha = render_shape[target_to_element[c][o]][:, :, 3:4]
          shape_rgb = render_shape[target_to_element[c][o]][:, :, :3]
          output_rgb = shape_alpha * shape_rgb + (1 - shape_alpha) * output_rgb
        output_rgb = output_rgb[:target_height, :target_width, :3]
        target_rgb = target_rgb[:target_height, :target_width, :3]
        for i, e in enumerate(target_to_element[c]):
          output_mask = render_shape[e][:target_height, :target_width, 3]
          over_mask = np.zeros_like(output_mask)
          ro_idx = render_order.index(i)
          for o in render_order[ro_idx + 1:]:
            o_alpha = render_shape[target_to_element[c][o]][:target_height, :target_width, 3]
            over_mask = np.maximum(over_mask, o_alpha)
          output_mask = np.where(over_mask>0, np.zeros_like(output_mask), output_mask)
          target_mask = np.max(np.uint8(fg_layers==active_shapes[e]), axis=0)[r_min_y:r_max_y, r_min_x:r_max_x]
          loss_mask = np.maximum(target_mask, output_mask)[..., None]
          loss = np.mean((loss_mask * (output_rgb - target_rgb))**2.0)
          optim_err[active_shapes[e]] = loss

      frame_opt_folder = os.path.join(debug_opt_folder, f'f{frame_idx:03d}')
      for comp_idx, (sbs, ot) in enumerate(zip(side_by_side, on_top)):
        frame_comp_folder = os.path.join(frame_opt_folder, f'c{comp_idx:02d}')
        if not os.path.exists(frame_comp_folder):
          os.makedirs(frame_comp_folder)
        save_frames(sbs, frame_comp_folder, suffix='pss')
        save_frames(ot, frame_comp_folder, suffix='pot')
        font = {
          'family': 'serif',
          'color':  'red',
          'weight': 'normal',
          'size': 12,
        }
        comp_losses = losses[comp_idx]
        fig = plt.figure()
        plt.plot(np.arange(0, len(comp_losses)), comp_losses)
        plt.axvline(x=np.argmin(comp_losses), c='r')
        plt.text(np.argmin(comp_losses), np.max(comp_losses), f'{np.min(comp_losses):.4f}', fontdict=font)
        plt.title('loss')
        fig.savefig(os.path.join(frame_comp_folder, 'loss_plot.png'))
        plt.close()

      # Update shape bank.
      for i, active_shape_idx in enumerate(active_shapes):
        rescale = max(element_sizes[i])
        sx = params_shape[2][i]
        sy = params_shape[3][i]
        theta = params_shape[4][i]
        kx = params_shape[5][i]
        ky = params_shape[6][i]
        tx = cxs[active_shapes_lidx[active_shape_idx]] + params_shape[0][i] * rescale / 2
        ty = cys[active_shapes_lidx[active_shape_idx]] + params_shape[1][i] * rescale / 2

        min_x, min_y, max_x, max_y = get_shape_coords(render_shape[i][:, :, 3])
        shape_params[active_shape_idx][0] = sx
        shape_params[active_shape_idx][1] = sy
        prev_cx = shape_params[active_shape_idx][2]
        prev_cy = shape_params[active_shape_idx][3]
        prev_dcx = shape_params[active_shape_idx][8]
        prev_dcy = shape_params[active_shape_idx][9]
        shape_params[active_shape_idx][2] = tx / frame_width
        shape_params[active_shape_idx][3] = ty / frame_height
        shape_params[active_shape_idx][4] = theta
        shape_params[active_shape_idx][5] = kx
        shape_params[active_shape_idx][6] = ky
        shape_params[active_shape_idx][7] = 1.0 if layer_z is None else layer_z[i]
        shape_params[active_shape_idx][8] = shape_params[active_shape_idx][2] - prev_cx  # d(cx)
        shape_params[active_shape_idx][9] = shape_params[active_shape_idx][3] - prev_cy  # d(cy)
        if len(shape_bank[active_shape_idx]) < 2:
          shape_params[active_shape_idx][10] = 0.0  # d^2(cx)
          shape_params[active_shape_idx][11] = 0.0  # d^2(cy)
        else:
          shape_params[active_shape_idx][10] = shape_params[active_shape_idx][8] - prev_dcx  # d^2(cx)
          shape_params[active_shape_idx][11] = shape_params[active_shape_idx][9] - prev_dcy  # d^2(cy)
        
        r_min_x, r_min_y, r_max_x, r_max_y = target_bounds_elements[i]
        full_shape = place_mask(
          render_shape[i], r_min_x, r_min_y, np.zeros((frame_height, frame_width, 4)))
        shape_info = {
          't': t,
          'coords': [(min_x, min_y), (max_x, max_y)],
          'centroid': (tx, ty),
          'h': shape_params[active_shape_idx].copy()
        }
        shape_bank[active_shape_idx].append(shape_info)
        mask = np.zeros_like(full_shape[:, :, 3], dtype=np.int32)
        mask[full_shape[:, :, 3]<1] = -1
        mask[full_shape[:, :, 3]>0] = active_shape_idx
        # time_bank['shapes'][t][i] = mask

    # Save reconstruction.
    recon_vis = bg_img / 255.0
    if len(active_shapes) > 0:
      active_shape_zs = [shape_params[active_shape_idx][7] for active_shape_idx in active_shapes]
      curr_active_shapes_sorted = [shape_idx for _, shape_idx in sorted(zip(active_shape_zs, active_shapes))]
      frame_shapes = []
      cxs = []
      cys = []
      sxs = []
      sys = []
      thetas = []
      kxs = []
      kys = []
      for shape_idx in curr_active_shapes_sorted:
        if shape_idx == -1:
          continue
        shape = shapes[shape_idx]
        frame_shapes.append(shape)
        cx, cy = shape_bank[shape_idx][-1]['centroid']
        cxs.append(cx)
        cys.append(cy)
        sxs.append(shape_params[shape_idx][0])
        sys.append(shape_params[shape_idx][1])
        thetas.append(shape_params[shape_idx][4])
        kxs.append(shape_params[shape_idx][5])
        kys.append(shape_params[shape_idx][6])
      frame_shapes = compositing.place_shape(
        frame_shapes, cxs, cys, sxs, sys, thetas, kxs, kys, frame_width, frame_height, 
        bg=np.transpose(recon_vis, (2, 0, 1))[None, ...] / 255.0, keep_alpha=True
      )
      frame_shapes = compositing.torch2numpy(frame_shapes)
      for i in range(frame_shapes.shape[0]):
        frame_shape_alpha = frame_shapes[i, :, :, 3:4]
        recon_vis = recon_vis * (1 - frame_shape_alpha) + frame_shapes[i, :, :, :3] * frame_shape_alpha
    cv2.imwrite(os.path.join(recon_folder, f'{frame_idx:03d}.png'), np.uint8(255 * recon_vis))
    diff_img = np.abs(recon_vis - frame / 255.0)
    
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(diff_img, aspect='auto', cmap='gray')
    fig.savefig(os.path.join(diff_folder, f'{frame_idx:03d}.png'))
    plt.close()

    prev_fg_layers = fg_layers.copy()

    frame_t1 = time.perf_counter()
    print(f'[TIME] Processing frame {frame_idx} took {frame_t1 - frame_t0:.2f}s')

    if (t + 1) % 50 == 0:
      # Save template info.
      with open(os.path.join(video_folder, 'shape_bank.pkl'), 'wb') as handle:
        pickle.dump(shape_bank, handle)
      print('[NOTE] Shape bank saved to:', os.path.join(video_folder, 'shape_bank.pkl'))

    #   # Save time bank info.
    #   with open(os.path.join(video_folder, 'time_bank_opt.pkl'), 'wb') as handle:
    #     pickle.dump(time_bank, handle)
    #   print('[NOTE] Time bank saved to:', os.path.join(video_folder, 'time_bank_opt.pkl'))

  # Save template info.
  with open(os.path.join(video_folder, 'shape_bank.pkl'), 'wb') as handle:
    pickle.dump(shape_bank, handle)
  print('[NOTE] Shape bank saved to:', os.path.join(video_folder, 'shape_bank.pkl'))

  # Save time bank info.
  # with open(os.path.join(video_folder, 'time_bank_opt.pkl'), 'wb') as handle:
  #   pickle.dump(time_bank, handle)
  # print('[NOTE] Time bank saved to:', os.path.join(video_folder, 'time_bank_opt.pkl'))

if __name__ == '__main__':
  main()
