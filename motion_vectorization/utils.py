import collections
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import expand_labels
import torch
from kornia.filters import gaussian_blur2d
from scipy.spatial.distance import cdist
import time
from scipy import ndimage
import torch.nn.functional as F
from torchvision.transforms import Resize
from pyefd import elliptic_fourier_descriptors
from pymatting import estimate_alpha_cf

from . import compositing
from . import sampling
from .linefiller.trappedball_fill import trapped_ball_fill_multi, \
  flood_fill_multi, mark_fill, build_fill_map, merge_fill, show_fill_map


def decompose(A):
  '''Decompose a 3x3 affine matrix into translation x rotation x shear x scale.
    Based off of:
    https://caff.de/posts/4X4-matrix-decomposition/decomposition.pdf
  '''
  tx, ty = A[0, 2], A[1, 2]
  C = A[:2, :2]
  C_ = C.T @ C
  det_C = np.linalg.det(C)
  d_xx = np.sqrt(C_[0, 0])
  d_xy = C_[0, 1] / d_xx
  d_yy = np.sqrt(C_[1, 1] - d_xy**2)
  if det_C <= 0:
    d_xx = -d_xx
  D = np.array([[d_xx, d_xy], [0, d_yy]])
  R = C @ np.linalg.inv(D)
  theta = np.arctan2(R[1, 0], R[0, 0])
  sx = d_xx
  sy = d_yy
  kx = d_xy / sy
  return tx, ty, sx, sy, theta, kx


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def get_numbers(dir):
  files = [os.path.splitext(f.name)[0].split('_')[0] for f in os.scandir(dir)]
  numbers = []
  for n in files:
    numbers_str = re.findall(r'\d+', n)
    numbers.extend([int(n_str) for n_str in numbers_str])
  return sorted(np.unique(numbers))


def get_shape_coords(mask, thresh=0.0):
  if len(np.where(mask>thresh)[0]) < 1:
    return 0, 0, 0, 0
  min_x = np.min(np.where(mask>thresh)[1])
  max_x = np.max(np.where(mask>thresh)[1]) + 1
  min_y = np.min(np.where(mask>thresh)[0])
  max_y = np.max(np.where(mask>thresh)[0]) + 1
  return min_x, min_y, max_x, max_y


def get_shape_mask(labels, idx, expand=False, dtype=np.uint8):
  if expand:
    return dtype(labels==idx)[..., None]
  else:
    return dtype(labels==idx)


def get_shape_centroid(mask):
  min_x, min_y, max_x, max_y = get_shape_coords(mask)
  return [(min_x + max_x - 1) / 2, (min_y + max_y - 1) / 2]


def rotmax(rad):
  return np.array([
    [np.cos(rad), -np.sin(rad)],
    [np.sin(rad), np.cos(rad)]
  ])


def params_to_mat(sx, sy, theta, kx, ky, tx=None, ty=None, k_first=False):
  K = np.array([
    [1, kx],
    [ky, 1 + kx * ky]
  ])
  R = rotmax(theta)
  S = np.array([
    [sx, 0],
    [0, sy]
  ])
  if tx is None:
    tx = 0.0
  if ty is None:
    ty = 0.0
  if k_first:
    A_ = K @ R @ S
  else:
    A_ = R @ K @ S
  A = np.zeros((3, 3))
  A[:2, :2] = A_
  A[0, 2] = tx
  A[1, 2] = ty
  A[2, 2] = 1.0
  return A


def is_one_to_one(i, prev_to_curr, curr_to_prev):
  return (
    len(prev_to_curr[i]) == 1 and 
    len(curr_to_prev[prev_to_curr[i][0]]) == 1 and 
    curr_to_prev[prev_to_curr[i][0]][0] == i
  )


def place_mask(mask, left_x, top_y, bg):
  out_bg = bg.copy()
  right_x = left_x + mask.shape[1]
  bottom_y = top_y + mask.shape[0]
  min_x = max(0, left_x)
  max_x = min(bg.shape[1], right_x)
  min_y = max(0, top_y)
  max_y = min(bg.shape[0], bottom_y)
  mask_min_x = 0 if left_x >= 0 else -left_x
  mask_max_x = mask.shape[1] if right_x < bg.shape[1] else mask.shape[1] - (right_x - bg.shape[1])
  mask_min_y = 0 if top_y >= 0 else -top_y
  mask_max_y = mask.shape[0] if bottom_y < bg.shape[0] else mask.shape[0] - (bottom_y - bg.shape[0])
  if (max_x > 0 and min_x < bg.shape[1]) and (max_y > 0 and min_y < bg.shape[0]) and (mask_max_x > 0 and mask_min_x < mask.shape[1]) and (mask_max_y > 0 and mask_min_y < mask.shape[0]):
    bg_section = bg[min_y:max_y, min_x:max_x]
    mask_section = mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x]
    out_bg[min_y:max_y, min_x:max_x] = np.where(mask_section>=0, mask_section, bg_section)
  return out_bg


def save_frames(frames, path, suffix=None):
  for i, frame in enumerate(frames):
    if suffix is not None:
      filename = f'{i + 1:03d}_{suffix}.png'
    else:
      filename = f'{i + 1:03d}.png'
    cv2.imwrite(os.path.join(path, filename), frame)


def compute_clusters_floodfill(fg_bg, edges, max_radius=3, min_cluster_size=50, min_density=0.15, min_dim=5):
  result = 255 * fg_bg * (1 - edges)
  fills = []
  fill = trapped_ball_fill_multi(result, max_radius, method='max')
  fills += fill
  result = mark_fill(result, fill)

  for rad in range(max_radius - 1, 0, -1):
    fill = trapped_ball_fill_multi(result, rad, method=None)
    fills += fill
    result = mark_fill(result, fill)

  fill = flood_fill_multi(result)
  fills += fill

  fillmap = build_fill_map(result, fills)
  fillmap = merge_fill(fillmap)

  # Remove invalid clusters.
  max_l_size = 0
  for l in np.unique(fillmap):
    if not is_valid_cluster(fillmap, l, min_cluster_size=min_cluster_size, min_density=min_density, min_dim=min_dim):
      fillmap[fillmap==l] = 0
    l_size = np.sum(np.uint8(fillmap==l))
    if l_size > max_l_size:
      max_l_size = l_size
  fillmap = expand_labels(fillmap, distance=2)
  fillmap_vis = np.uint8(show_fill_map(fillmap))
      
  # Compress ord of labels.
  idx = -1
  for l in np.unique(fillmap):
    fillmap[fillmap==l] = idx
    idx += 1

  return fillmap, fillmap_vis


def clean_labels(labels, spatial, min_cluster_size):
  labels_new = np.zeros_like(labels)
  for l in np.unique(labels):
    l_mask = np.uint8(labels==l)
    _, l_labels = cv2.connectedComponents(l_mask)
    for l_ in np.unique(l_labels):
      if l_ == 0:
        continue
      if not is_valid_cluster(spatial, l_labels, l_, min_cluster_size=min_cluster_size):
        l_mask[l_labels==l_] = 0
    labels_new[l_mask==1] = l
  return labels_new


def get_comp_label_map(comps, labels):
  label_to_comp = {}
  label_to_comp_size = {}
  for c in np.unique(comps):
    if c < 0:
      continue
    c_mask = np.uint8(comps==c)
    for l in np.unique(labels):
      if l < 0:
        continue
      l_mask = np.uint8(labels==l)
      lc_overlap = cv2.bitwise_and(c_mask, l_mask)
      lc_overlap_sum = np.sum(lc_overlap)
      if lc_overlap_sum > 0:
        # assert l not in label_to_comp
        if l not in label_to_comp:
          label_to_comp[l] = c
          label_to_comp_size[l] = lc_overlap_sum
        else:
          if lc_overlap_sum > label_to_comp_size[l]:
            label_to_comp[l] = c
            label_to_comp_size[l] = lc_overlap_sum
  comp_to_label = collections.defaultdict(list)
  for l, c in label_to_comp.items():
    comp_to_label[c].append(l)
  return comp_to_label, label_to_comp


def is_valid_cluster(labels, l, min_cluster_size=25, min_dim=5, min_density=0.15):
  cluster_size = np.sum(labels==l)
  if cluster_size < min_cluster_size:
    # print(cluster_size)
    return False
  min_x, min_y, max_x, max_y = get_shape_coords(np.uint8(labels==l))
  if max_x - min_x < min_dim:
    # print('[ERRS] X dim', max_x - min_x, '<', min_dim)
    return False
  if max_y - min_y < min_dim:
    # print('[ERRS] Y dim', max_y - min_y, '<', min_dim)
    return False
  density = cluster_size / ((max_x - min_x) * (max_y - min_y))
  if density < min_density:  # Return None if bounding box is a strip.
    # print('[ERRS] Density', f'{density:.2f}', '<', f'{min_density:.2f}')
    return False
  else:
    return True


def blank_bg(bgr, shape):
  return bgr * np.ones(shape, dtype=np.uint8)


def get_alpha(mask, img, kernel_radius=5, bg_color=None, exclude=None, expand=False):
  kernel = np.ones((kernel_radius, kernel_radius), np.uint8)
  mask_erode = cv2.erode(mask, kernel, iterations=1)
  if np.sum(mask_erode) < 1:
    mask_erode = mask.copy()
  trimap = cv2.dilate(mask, kernel, iterations=1)
  trimap[trimap!=mask_erode] = 0.5
  if exclude is not None:
    trimap[exclude>0] = 0
  alpha = estimate_alpha_cf(img / 255.0, trimap)
  if expand:
    return alpha[..., None]
  else:
    return alpha


def shape_at_border(labels, idx):
  min_x, min_y, max_x, max_y = get_shape_coords(labels==idx)
  return min_x < 2 or min_y < 2 or abs(labels.shape[1] - max_x) < 2 or abs(labels.shape[0] - max_y) < 2


def get_shape(frame, bg, labels, l, min_cluster_size=50, min_density=0.15):
  """Extracts a box around the detected shape within the frame.
  
  The returned array represents an image in which the target shape is isolated
  by a box, and the rest of the frame is transparent (black).

  Args:
    frame: A numpy array of shape (H, W), the original frame.
    labels: A numpy array of shape (H, W) mapping each pixel to its cluster index.
    l: The label to isolate, an int.

  Returns:
    shape: A numpy array with shape (H, W) or None.
    (min_x, min_y): The min box coordinates as (x, y) an tuple.
    (max_x, max_y): The max box coordinates as (x, y) an tuple.
    (centroid_x, centroid_y): The centroid of the shape as (x, y) an tuple.
  """
  frame_height, frame_width, _ = frame.shape
  if not is_valid_cluster(labels, l, min_cluster_size=min_cluster_size, min_density=min_density):
    return None, None, 0.0, (0, 0), (0, 0), (0, 0)
  else:
    mask = np.zeros((frame.shape[0], frame.shape[1], 1))
    mask[labels==l] = 1
    non_mask = np.uint8(labels>=0)
    non_mask[labels==l] = 0
    alpha = get_alpha(mask[..., 0], frame, exclude=non_mask)[..., None]
    rgb = frame * alpha + bg * (1 - alpha)
    shape = np.uint8(np.dstack([rgb, np.uint8(255 * alpha)]))
    min_x = max(np.min(np.where(alpha>0)[1]), 0)
    max_x = min(np.max(np.where(alpha>0)[1]) + 1, frame_width)
    min_y = max(np.min(np.where(alpha>0)[0]), 0)
    max_y = min(np.max(np.where(alpha>0)[0]) + 1, frame_height)
    centroid_x, centroid_y = get_shape_centroid(alpha[..., 0])
    density = np.sum(alpha) / ((max_x - min_x) * (max_y - min_y))

  return shape, mask, density, (min_x, min_y), (max_x, max_y), (centroid_x, centroid_y)


def get_active_shapes(shape_bank, latest_frame_idx):
  active_shapes = []
  for shape_idx in shape_bank:
    if shape_idx < 0:
      continue
    if shape_bank[shape_idx]['t'] == latest_frame_idx:
      active_shapes.append(shape_idx)
  return active_shapes


def compute_transforms(
  elements, frame, background, target_to_element, c_variables=None, layer_z=None, default_c_vars=None, default_z=None,
  use_k=True, use_r=True, use_s=True, use_t=True, rgb_weight=10.0,
  origin=None, use_mask=False, min_size=256, blur_kernel=3, lr=0.001, n_steps=100, loss_type='l1', 
  p_weight=0.1, bleed=0, device='cpu'):
  t0 = time.perf_counter()
  t2e_onehot = torch.stack([torch.sum(F.one_hot(
    torch.tensor(target_to_element[t], device=device).long(), num_classes=len(elements)
  ), dim=0) for t in target_to_element]).double()
  t2e_onehot = t2e_onehot / torch.sum(t2e_onehot, dim=1, keepdims=True)
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Making t2e_onehot took: {t1 - t0:.4f}s')
  
  # Compute the min size.
  new_min_size = 0
  for c in target_to_element:
    h, w, _ = frame[c].shape
    h = min(min_size, h)
    w = min(min_size, w)
    new_min_size = max(new_min_size, max(h, w))
  min_size = new_min_size

  t0 = time.perf_counter()
  resize = Resize((min_size, min_size))
  resize_ratios = [1.0] * len(elements)
  frames_square = []
  frames_square_full = []
  backgrounds_square = []
  backgrounds_square_full = []
  loss_masks = []
  for c in target_to_element:
    f = frame[c].copy()
    bg = background[c].copy()
    mask = torch.ones(1, f.shape[0], f.shape[1], dtype=torch.float64, device=device)
    if f.shape[1] > f.shape[0]:
      pad_f = ((0, f.shape[1] - f.shape[0]), (0, 0), (0, 0))
      pad_m = (0, 0, 0, f.shape[1] - f.shape[0])
      pad_b = ((0, f.shape[1] - f.shape[0]), (0, 0), (0, 0))
    else:
      pad_f = ((0, 0), (0, f.shape[0] - f.shape[1]), (0, 0))
      pad_m = (0, f.shape[0] - f.shape[1], 0, 0)
      pad_b = ((0, 0), (0, f.shape[0] - f.shape[1]), (0, 0))
    f_square_full = np.pad(f, pad_f)
    bg_square_full = np.pad(bg, pad_b)
    resize_ratio = min_size / f_square_full.shape[0]
    f_square = cv2.resize(f_square_full, (min_size, min_size))
    bg_square = cv2.resize(bg_square_full, (min_size, min_size))
    bg_square_full = np.pad(bg_square_full, ((bleed, bleed), (bleed, bleed), (0, 0)))
    mask_pad = F.pad(mask, pad_m)
    mask_pad = resize(mask_pad)
    for e in target_to_element[c]:
      resize_ratios[e] = resize_ratio
    loss_masks.append(mask_pad)
    frames_square.append(f_square)
    frames_square_full.append(f_square_full)
    backgrounds_square.append(bg_square / 255.0)
    backgrounds_square_full.append(bg_square_full / 255.0)

  frames_square = np.stack(frames_square)
  backgrounds_square = torch.tensor(np.stack(backgrounds_square), device=device).permute(0, 3, 1, 2)
  loss_masks = torch.stack(loss_masks)
  resize_ratios = np.array(resize_ratios)[..., None]
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Making target tensors took {t1 - t0:.4f}s')

  t0 = time.perf_counter()
  gt_tensor = torch.tensor(frames_square / 255.0, device=device).permute(0, 3, 1, 2)
  gt_scales = [gt_tensor.clone()]
  n_scales = min(3, int(np.log2(min(gt_tensor.shape[2], gt_tensor.shape[3]))) - 1)
  for i in range(n_scales):
    gt_tensor_d = compositing.dsample(gt_scales[-1])
    gt_tensor_d = gaussian_blur2d(gt_tensor_d, (blur_kernel, blur_kernel), (blur_kernel, blur_kernel))
    gt_scales.append(gt_tensor_d)
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Computing scales took {t1 - t0:.2f}s')
  
  # Pad elements to be square to compute sx, sy which can be applied across aspect ratios.
  t0 = time.perf_counter()
  elements_square = []
  elements_square_fullres = []
  for i in range(len(elements)):
    e = elements[i].clone()
    if e.shape[1] > e.shape[2]:
      pad = (0, e.shape[1] - e.shape[2], 0, 0)
    else:
      pad = (0, 0, 0, e.shape[2] - e.shape[1])
    e_pad = F.pad(e, pad)
    e = resize(e_pad)
    elements_square_fullres.append(e_pad)
    e_pad = resize(e_pad)
    elements_square.append(e_pad)
  elements_square = torch.stack(elements_square).to(device)
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Padding elements took {t1 - t0:.2f}s')

  # for i, bg in enumerate(backgrounds_square_full):
  #   print(f'targ {i}:', bg.shape, target_to_element[i])
  # for i, elem in enumerate(elements_square_fullres):
  #   print(f'elem {i}:', elem.shape)
  

  t0 = time.perf_counter()
  if origin is None:
    origin = torch.stack([
      torch.mean(torch.stack(torch.where(elements_square[i, 3, :, :]>0)).float(), axis=1) for i in range(elements_square.shape[0])
    ])
    origin = torch.flip(origin, dims=[1])
  origin *= torch.tensor(resize_ratios, dtype=torch.float64, device=device)
  origin = origin.to(device)
  sc = torch.tensor([elements_square.shape[3], elements_square.shape[2]], device=device)[None, ...]
  origin = origin / sc

  if c_variables is None:
    c_variables = [
      torch.tensor([0.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([1.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([1.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements_square.shape[0])], dtype=torch.float64, device=device)
    ]  # tx, ty, sx, sy, theta, kx, ky
  if default_c_vars is None:
    default_c_vars = [v.clone().to(device) for v in c_variables[2:]]
  if default_z is None:
    default_z = torch.tensor([5.0 for e in elements], dtype=torch.float64, device=device)
  for v in default_c_vars:
    v.requires_grad = False

  # Get initial loss.
  render_alls, alphas = compositing.composite_layers(
    elements_square, c_variables, origin, target_to_element, elements_square.shape, backgrounds_square,
    layer_z=layer_z, blur=False, blur_kernel=blur_kernel, debug=False, device=device
  )

  outputs = torch.cat([render_alls, alphas], dim=1) * loss_masks
  rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss = compositing.loss_fn(
    outputs, gt_scales, c_variables, 
    layer_z=layer_z, default_variables=default_c_vars, default_z=default_z,
    loss_type=loss_type, use_mask=use_mask, p_weight=p_weight, device=device
  )
  params_loss = t2e_onehot @ params_loss
  loss = rgb_weight * rgb_loss + rgb_scales_loss + alpha_loss + alpha_scales_loss + params_loss
  
  all_var_names = ['tx', 'ty', 'sx', 'sy', 'theta', 'kx', 'ky', 'z']
  opt_variables = []
  opt_var_names = []
  if not use_t:
    c_variables[0].requires_grad = False
    c_variables[1].requires_grad = False
  else:
    c_variables[0].requires_grad = True
    c_variables[1].requires_grad = True
    opt_variables.append(c_variables[0])
    opt_variables.append(c_variables[1])
    opt_var_names.append('tx')
    opt_var_names.append('ty')
  if not use_s:
    c_variables[2].requires_grad = False
    c_variables[3].requires_grad = False
  else:
    c_variables[2].requires_grad = True
    c_variables[3].requires_grad = True
    opt_variables.append(c_variables[2])
    opt_variables.append(c_variables[3])
    opt_var_names.append('sx')
    opt_var_names.append('sy')
  if not use_r:
    c_variables[4].requires_grad = False
  else:
    c_variables[4].requires_grad = True
    opt_variables.append(c_variables[4])
    opt_var_names.append('theta')
  if not use_k:
    c_variables[5].requires_grad = False
    c_variables[6].requires_grad = False
  else:
    c_variables[5].requires_grad = True
    c_variables[6].requires_grad = True
    opt_variables.append(c_variables[5])
    opt_variables.append(c_variables[6])
    opt_var_names.append('kx')
    opt_var_names.append('ky')
  if layer_z is not None:
    opt_variables.append(layer_z)
    opt_var_names.append('z')
  optimizer, scheduler = compositing.init_optimizer(opt_variables, lr=lr)
  best_c_variables = [v.clone().detach() for v in c_variables]
  best_c_variables.append(layer_z.clone().detach())
  for v in best_c_variables:
    v.requires_grad = False
  min_loss = loss
  prev_min_loss = min_loss
  best_rgb_loss = rgb_loss
  best_alpha_loss = alpha_loss
  best_params_loss = params_loss
  best_renders = render_alls
  gt_frame = None
  round = 0
  total_step = 0
  side_by_side = [[] for comp in frame]
  on_top = [[] for comp in frame]
  losses = loss.detach().cpu().numpy()[..., None]
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Initializing variables took {t1 - t0:.2f}s')

  t0 = time.perf_counter()  
  for c, (render_all, f) in enumerate(zip(render_alls * loss_masks, frames_square)):
    best_render = np.uint8(255 * compositing.torch2numpy(render_all.detach().cpu()).copy())
    gt_frame = f[:, :, :3] / 255.0
    for i in range(origin[target_to_element[c]].shape[0]):
      ox, oy = origin[target_to_element[c][i]].detach().cpu().numpy()
      best_render = cv2.circle(best_render, (int(ox * best_render.shape[1]), int(oy * best_render.shape[0])), 1, (255, 255, 255), 2)
    sbs = np.uint8(np.concatenate([best_render, 255 * gt_frame]))
    sbs = cv2.putText(sbs, 's0', (10, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    sbs = cv2.putText(sbs, f'{loss[c]:.4f}', (sbs.shape[1] - 50, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    comp = np.uint8(np.abs(best_render - 255 * gt_frame))
    comp = np.pad(comp, ((0, 200), (0, 0), (0, 0)))
    comp = np.concatenate([comp, np.zeros([comp.shape[0], 300, 3], dtype=np.uint8)], axis=1)
    comp = cv2.putText(comp, 's0', (10, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    k = 0
    for opt_var_name in opt_var_names:
      i = all_var_names.index(opt_var_name)
      for elem in target_to_element[c]:
        comp = cv2.putText(comp, f'{opt_var_name}: {best_c_variables[i][elem]:.4f}', (best_render.shape[1] + 10, 10 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        k += 1
    comp = cv2.putText(comp, f'loss: {loss[c]:.4f}', (best_render.shape[1] + 120, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i, (loss_val, loss_name) in enumerate(zip([best_rgb_loss, best_alpha_loss, best_params_loss], ['rgb', 'alpha', 'params'])):
      comp = cv2.putText(comp, f'{loss_name}: {loss_val[c]:.4f}', (best_render.shape[1] + 120, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i in range(origin.shape[0]):
      ox, oy = origin[i].detach().cpu().numpy()
      comp = cv2.putText(comp, f'centroid: ({ox:.2f}, {oy:.2f})', (best_render.shape[1] + 120, 10 * (i + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    on_top[c].append(comp)
    side_by_side[c].append(sbs)
    # cv2.imshow('ss', sbs)
    # cv2.imshow('ot', comp)
    # cv2.waitKey(1)
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Visualizing opt at one step took {t1 - t0:.4f}s')

  t0 = time.perf_counter()
  while round < 2 and total_step < 500:
    round += 1
    for step in range(n_steps):
      t2 = time.perf_counter()
      optimizer.zero_grad()
      blur = torch.max(min_loss) > 5e-2
      render_alls, alphas = compositing.composite_layers(
        elements_square, c_variables, origin, target_to_element, elements_square.shape, backgrounds_square,
        layer_z=layer_z, blur=blur, blur_kernel=blur_kernel, debug=False, device=device
      )
      outputs = torch.cat([render_alls, alphas], dim=1) * loss_masks
      t3 = time.perf_counter()
      # print(f'[LOOP_TIME] Rendering current opt results took {t3 - t2:.4f}s')
      t2 = time.perf_counter()
      rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss = compositing.loss_fn(
        outputs, gt_scales, c_variables, 
        layer_z=layer_z, default_variables=default_c_vars, default_z=default_z,
        loss_type=loss_type, use_mask=use_mask, p_weight=p_weight, device=device
      )
      t3 = time.perf_counter()
      params_loss = t2e_onehot @ params_loss
      all_loss = rgb_weight * rgb_loss + rgb_scales_loss + alpha_loss + alpha_scales_loss + params_loss
      loss = torch.mean(all_loss)
      losses = np.concatenate([losses, all_loss.detach().cpu().numpy()[..., None]], axis=-1)
      improved = torch.where(all_loss<min_loss)[0]
      for t in improved:
        for i, (v, v_name) in enumerate(zip(opt_variables, opt_var_names)):
          for e in target_to_element[int(t.item())]:
            best_c_variables[all_var_names.index(v_name)][e] = v[e].item()
        best_renders[int(t.item())] = render_alls[int(t.item())].clone()
      best_rgb_loss = torch.minimum(best_rgb_loss, rgb_loss)
      best_alpha_loss = torch.minimum(best_alpha_loss, alpha_loss)
      best_params_loss = torch.minimum(best_params_loss, params_loss)
      min_loss = torch.minimum(all_loss, min_loss)
      # print(f'[LOOP_TIME] Computing loss took {t3 - t2:.4f}s')
      t2 = time.perf_counter()
      loss.backward()
      optimizer.step()
      scheduler.step()
      t3 = time.perf_counter()
      # print(f'[LOOP_TIME] Backpropagation took {t3 - t2:.4f}s')

      if (total_step + 1) % 5 == 0:
        for c, (best_render, f) in enumerate(zip(best_renders * loss_masks, frames_square)):
          best_render = compositing.torch2numpy(best_render.detach().cpu())
          gt_frame = f[:, :, :3] / 255.0
          sbs = np.uint8(255 * np.concatenate([best_render, gt_frame]))
          sbs = cv2.putText(sbs, f's{total_step + 1}', (10, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
          sbs = cv2.putText(sbs, f'{min_loss[c]:.4f}', (sbs.shape[1] - 50, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
          comp = np.uint8(255 * np.abs(best_render - gt_frame))
          comp = np.pad(comp, ((0, 200), (0, 0), (0, 0)))
          comp = np.concatenate([comp, np.zeros([comp.shape[0], 300, 3], dtype=np.uint8)], axis=1)
          comp = cv2.putText(comp, f's{total_step + 1}', (10, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
          k = 0
          for opt_var_name in opt_var_names:
            i = all_var_names.index(opt_var_name)
            for elem in target_to_element[c]:
              comp = cv2.putText(comp, f'{opt_var_name}: {best_c_variables[i][elem]:.4f}', (best_render.shape[1] + 10, 10 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
              k += 1
          comp = cv2.putText(comp, f'loss: {min_loss[c]:.4f}', (best_render.shape[1] + 120, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
          for i, (loss_val, loss_name) in enumerate(zip([best_rgb_loss, best_alpha_loss, best_params_loss], ['rgb', 'alpha', 'params'])):
            comp = cv2.putText(comp, f'{loss_name}: {loss_val[c]:.4f}', (best_render.shape[1] + 120, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
          for i in range(origin.shape[0]):
            ox, oy = origin[i].detach().cpu().numpy()
            comp = cv2.putText(comp, f'centroid: ({ox:.2f}, {oy:.2f})', (best_render.shape[1] + 120, 10 * (i + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
          side_by_side[c].append(sbs)
          on_top[c].append(comp)
          # cv2.imshow('ot', comp)
          # cv2.imshow('ss', sbs)
          # cv2.waitKey(1)
      total_step += 1
    if torch.min(prev_min_loss - min_loss) > 1e-4:
      round = 0
    prev_min_loss = torch.minimum(min_loss, prev_min_loss)

  render_alls, alphas = compositing.composite_layers(
    elements_square, best_c_variables[:7], origin, target_to_element, elements_square.shape, backgrounds_square,
    layer_z=layer_z, blur=False, blur_kernel=blur_kernel, debug=False, device=device
  )
  for c, (best_render, f) in enumerate(zip(render_alls * loss_masks, frames_square)):
    best_render = compositing.torch2numpy(best_render.detach().cpu())
    gt_frame = f[:, :, :3] / 255.0
    sbs = np.uint8(255 * np.concatenate([best_render, gt_frame], axis=1))
    sbs = cv2.putText(sbs, f's{total_step + 1}', (10, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    sbs = cv2.putText(sbs, f'{min_loss[c]:.4f}', (sbs.shape[1] - 50, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    comp = np.uint8(255 * np.abs(best_render - gt_frame))
    comp = np.pad(comp, ((0, 200), (0, 0), (0, 0)))
    comp = np.concatenate([comp, np.zeros([comp.shape[0], 300, 3], dtype=np.uint8)], axis=1)
    comp = cv2.putText(comp, f's{total_step + 1}', (10, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    k = 0
    for opt_var_name in opt_var_names:
      i = all_var_names.index(opt_var_name)
      for elem in target_to_element[c]:
        comp = cv2.putText(comp, f'{opt_var_name}: {best_c_variables[i][elem]:.4f}', (best_render.shape[1] + 10, 10 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        k += 1
    comp = cv2.putText(comp, f'loss: {min_loss[c]:.4f}', (best_render.shape[1] + 120, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i, (loss_val, loss_name) in enumerate(zip([best_rgb_loss, best_alpha_loss, best_params_loss], ['rgb', 'alpha', 'params'])):
      comp = cv2.putText(comp, f'{loss_name}: {loss_val[c]:.4f}', (best_render.shape[1] + 120, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i in range(origin.shape[0]):
      ox, oy = origin[i].detach().cpu().numpy()
      comp = cv2.putText(comp, f'centroid: ({ox:.2f}, {oy:.2f})', (best_render.shape[1] + 120, 10 * (i + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    on_top[c].append(comp)
    side_by_side[c].append(sbs)
    # cv2.imshow('ot', comp)
    # cv2.imshow('ss', sbs)
    # cv2.waitKey(1)
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Optimization steps took {t1 - t0:.2f}s')

  t0 = time.perf_counter()
  render_list = []
  for e in range(len(elements_square_fullres)):
    target_idx = ((t2e_onehot[:, e]>0).nonzero(as_tuple=True)[0])
    element_fullres = elements_square_fullres[e][None, ...]
    render = sampling.sampling_layer(
      element_fullres, 
      best_c_variables[0][e:e + 1],  # tx
      best_c_variables[1][e:e + 1],  # ty
      best_c_variables[2][e:e + 1],  # sx
      best_c_variables[3][e:e + 1],  # sy
      best_c_variables[4][e:e + 1],  # theta
      best_c_variables[5][e:e + 1],  # kx
      best_c_variables[6][e:e + 1],  # ky
      element_fullres.shape,
      bleed=bleed,
      origin=origin[e:e + 1], blur=False, interp='bilinear', device=device)
    alpha, _ = torch.max(render[:, 3:4, :, :], dim=0)
    alpha = alpha.squeeze().detach().cpu().numpy()
    render = compositing.torch2numpy(render)[0]
    bg = backgrounds_square_full[target_idx] * np.ones((alpha.shape[0], alpha.shape[1], 3))
    render[..., :3] = render[..., 3:4] * render[..., :3] + (1 - render[..., 3:4]) * bg
    render_list.append(render)
  params = [v.detach().cpu().numpy() for v in best_c_variables[:7]]
  layer_z = best_c_variables[7].detach().cpu().numpy() if layer_z is not None else None
  outputs = torch.cat([render_alls, alphas], dim=1)
  outputs = compositing.torch2numpy(outputs)
  t1 = time.perf_counter()
  # print(f'[OPT_TIME] Generating outputs took {t1 - t0:.2f}s')
  return outputs, params, layer_z, alpha, render_list, frames_square_full, best_rgb_loss, best_alpha_loss, side_by_side, on_top, losses


def propagate_labels(frame, fg_bg_clusters, fg_bg, debug=False):
  # Coordinate grid for original frame.
  x = np.linspace(0, frame.shape[1] - 1, frame.shape[1])
  y = np.linspace(0, frame.shape[0] - 1, frame.shape[0])

  # Get unlabeled regions and mask out components which should be marked as new shapes.
  fg_bg_binary = np.uint8(fg_bg>0)
  unlabeled = fg_bg_binary.copy()
  unlabeled[fg_bg_clusters>0] = 0
  labeled = fg_bg_clusters.copy()
  labeled[labeled>0] = 1
  expanded_labels = fg_bg_clusters * fg_bg_binary
  unique_labels = np.unique(expanded_labels).tolist()
  unique_labels.remove(0)
  candidates = []
  cumulative_label_expand = np.zeros_like(fg_bg_binary, dtype=np.uint8)
  for l in unique_labels:
    label_expand_region = cv2.bitwise_or(unlabeled, np.uint8(fg_bg_clusters==l))
    cumulative_label_expand = cv2.bitwise_or(cumulative_label_expand, label_expand_region)
    _, unlabeled_comps = cv2.connectedComponents(label_expand_region)
    comp_masks = []
    for k in np.unique(unlabeled_comps):
      k_mask = np.zeros_like(unlabeled_comps)
      k_mask[unlabeled_comps==k] = 1
      comp_masks.append(k_mask)
    comp_masks = np.stack(comp_masks)
    overlap = np.uint8(fg_bg_clusters==l)[None, ...] * comp_masks
    l_candidates = np.zeros_like(fg_bg_clusters, dtype=np.uint8)
    for r in range(overlap.shape[0]):
      if np.any(overlap[r]):
        l_candidates[comp_masks[r]>0] = l
    candidates.append(l_candidates)
  if len(candidates) < 1:
    return fg_bg_clusters
  candidates = np.stack(candidates, axis=-1)
  unlabeled_indices = np.stack(np.where(unlabeled==1)).T

  if unlabeled_indices.shape[0] >= 1:
    # Compute LAB mean of each cluster.
    c_dists = []
    frame_lab = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2LAB)
    unlabeled_lab = np.stack([frame_lab[y, x] for (y, x) in unlabeled_indices])
    for c in unique_labels:
      indices = np.stack(np.where(expanded_labels==c)).T
      center = np.mean(indices, axis=0)

      # Compute smallest distance from all unlabeled indices to this component.
      xy_dists = cdist(unlabeled_indices, np.argwhere(expanded_labels==c)) / np.linalg.norm(unlabeled_indices - center[None, ...], axis=1, keepdims=True)
      xy_dist = xy_dists.min(axis=1)
      weights = np.stack([fg_bg[y, x] for (y, x) in indices])[None, ...] * np.exp(-xy_dists)
      lab_weighted = np.stack([frame_lab[y, x, :3] for (y, x) in indices])[None, ...] * weights[..., None]
      lab_mean = np.sum(lab_weighted, axis=1) / np.sum(weights, axis=1, keepdims=True)
      lab_dist = np.linalg.norm(unlabeled_lab - lab_mean, axis=1) / 255.0
      c_dists.append(lab_dist + xy_dist)
    c_dists = np.stack(c_dists, axis=-1)
    best_c = np.argsort(c_dists, axis=-1)
  
  # Propagate labels into unlabeled regions.
  if unlabeled_indices.shape[0] >= 1:
    for i, (y, x) in enumerate(unlabeled_indices):
      candidate_labels = np.unique(candidates[y, x])
      if len(candidate_labels) == 1:
        expanded_labels[y, x] = candidate_labels[0]
      else:
        for c in best_c[i]:
          if unique_labels[c] in candidate_labels:
            expanded_labels[y, x] = unique_labels[c]  # We have to offset for the background cluster.
            break
  return expanded_labels


def get_cmap(n, name='jet'):
  """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
  RGB color; the keyword argument name must be a standard mpl colormap name."""
  return plt.cm.get_cmap(name, n)


def pad_to_square(img, dim=None, color=None):
  if dim is None:
    dim = max(img.shape[0], img.shape[1])
  pad_t = (dim - img.shape[0]) // 2
  pad_b = dim - pad_t - img.shape[0]
  pad_l = (dim - img.shape[1]) // 2
  pad_r = dim - pad_l - img.shape[1]
  padding = [(pad_t, pad_b), (pad_l, pad_r), *[(0, 0) for i in range(len(img.shape) - 2)]]
  if color is None:
    img_pad = np.pad(img, padding)
  else:
    constant_values = ((color, color), (color, color), (0, 0))
    img_pad = np.pad(img, padding, constant_values=constant_values)
  return img_pad


def get_moment_features(img, radius=None, degree=8):
  gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
  shape_mask = np.uint8(img[:, :, 3]>0)
  min_x = max(np.min(np.where(shape_mask==1)[1]) - 10, 0)
  max_x = min(np.max(np.where(shape_mask==1)[1]) + 10, shape_mask.shape[1])
  min_y = max(np.min(np.where(shape_mask==1)[0]) - 10, 0)
  max_y = max(np.max(np.where(shape_mask==1)[0]) + 10, shape_mask.shape[0])
  if radius is None:
    radius = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2) // 2
  pad = int(radius * (np.sqrt(2) - 1))
  shape = shape_mask * gray
  shape = gray[min_y:max_y, min_x:max_x]
  shape = pad_to_square(shape)
  shape = np.pad(shape, ((10, 10), (10, 10)))
  contours, _ = cv2.findContours(shape, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = np.concatenate(contours, axis=0)
  coeff = elliptic_fourier_descriptors(np.squeeze(contours), order=10, normalize=True)
  coeff = coeff[3:] / np.linalg.norm(coeff[3:])
  return coeff


def get_contours_and_edges(frame, threshold1=100, threshold2=200):
    edges = cv2.Canny(frame, threshold1, threshold2)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(np.array(contour), 1) for contour in contours]
    return edges, contours


def draw_contour(img, contour, color, thickness):
  out_img = img.copy()
  for i in range(len(contour) - 1):
    out_img = cv2.line(out_img, tuple(contour[i]), tuple(contour[i + 1]), color, thickness)
  return out_img
    

def visualize_contours(frame, contours):
  contours_vis = np.zeros_like(frame)
  cmap = get_cmap(len(contours) + 1)
  for c, cnt in enumerate(contours):
    contours_vis = draw_contour(contours_vis, cnt, [255 * k for k in cmap(c)], 2)
  return contours_vis


def init_rot_scale(prev_crop, curr_crop, prev_angle, bg_crop, over_mask=None, p_weight=0.1, init_thresh=0.01, debug=False):
  # Initialize rotation.
  best_crop_rot = prev_crop.copy()
  min_rot_diff = np.inf
  best_rgb_diff = np.inf
  best_angle_diff = np.inf
  best_rot = None
  best_fallback_rot = None
  if over_mask is None:
    over_mask = np.ones_like(curr_crop[:, :, 3])
  for abs_angle in range(0, 180, 2):
    for angle in [prev_angle + abs_angle, prev_angle - abs_angle]:
      prev_crop_rot = ndimage.rotate(prev_crop, angle, order=0)
      t_min_x = np.min(np.where(prev_crop_rot[:, :, -1]>0)[1])
      t_max_x = np.max(np.where(prev_crop_rot[:, :, -1]>0)[1]) + 1
      t_min_y = np.min(np.where(prev_crop_rot[:, :, -1]>0)[0])
      t_max_y = np.max(np.where(prev_crop_rot[:, :, -1]>0)[0]) + 1
      prev_crop_rot = prev_crop_rot[t_min_y:t_max_y, t_min_x:t_max_x]
      prev_crop_rot = cv2.resize(prev_crop_rot, (curr_crop.shape[1], curr_crop.shape[0]))
      if len(prev_crop_rot.shape) < 3:
        prev_crop_rot = prev_crop_rot[..., None]
      loss_mask = np.maximum(curr_crop[:, :, -1], prev_crop_rot[:, :, -1])
      loss_mask[over_mask>0] = 0
      if prev_crop.shape[-1] >= 3:
        prev_crop_rot = prev_crop_rot[:, :, :-1] * prev_crop_rot[:, :, -1:] + bg_crop * (1 - prev_crop_rot[:, :, -1:])
      if curr_crop.shape[-1] >= 3:
        curr_crop_rgb = curr_crop[:, :, :-1] * curr_crop[:, :, -1:] + bg_crop * (1 - curr_crop[:, :, -1:])
      else:
        curr_crop_rgb = curr_crop
      rgb_diff = np.sum(loss_mask[..., None] * (prev_crop_rot - curr_crop_rgb)**2) / np.sum(loss_mask)
      angle_diff = abs(np.cos(np.deg2rad(angle)) - np.cos(prev_angle)) + abs(np.sin(angle) - np.sin(prev_angle))
      rot_diff = rgb_diff + p_weight * angle_diff
      if rgb_diff < init_thresh:
        best_rot = angle
        min_rot_diff = rot_diff
        best_rgb_diff = rgb_diff
        best_angle_diff = angle_diff
        best_crop_rot = prev_crop_rot.copy()
      if rot_diff < min_rot_diff:
        best_fallback_rot = angle
        min_rot_diff = rot_diff
        best_rgb_diff = rgb_diff
        best_angle_diff = angle_diff
        best_crop_rot = prev_crop_rot.copy()
      rot_vis = np.concatenate([curr_crop_rgb, loss_mask[..., None] * np.abs(curr_crop_rgb - best_crop_rot), loss_mask[..., None] * np.abs(curr_crop_rgb - prev_crop_rot)], axis=1)
      rot_vis = np.uint8(255 * rot_vis)
      display_best_rot = best_fallback_rot if best_rot is None else best_rot
      rot_vis = cv2.putText(
        rot_vis, f'{display_best_rot:.2f} deg: {min_rot_diff:.2f} ({best_rgb_diff:.2f} {best_angle_diff:.2f})', (10, rot_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
      rot_vis = cv2.putText(
        rot_vis, f'{angle:.2f} deg', (rot_vis.shape[1] - 50, rot_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
      if debug:
        cv2.imshow('rot_vis', rot_vis)
        cv2.waitKey(0)
    if best_rot is not None:
      break
  if best_rot is None:
    print('[NOTE] Fallback angle used!')
    best_rot = best_fallback_rot
  theta = best_rot

  # Initialize scale.
  curr_crop_alpha = curr_crop[:, :, -1]
  curr_crop_unrot = ndimage.rotate(curr_crop_alpha, -best_rot, order=0)
  t_min_x = np.min(np.where(curr_crop_unrot[:, :]>0)[1])
  t_max_x = np.max(np.where(curr_crop_unrot[:, :]>0)[1]) + 1
  t_min_y = np.min(np.where(curr_crop_unrot[:, :]>0)[0])
  t_max_y = np.max(np.where(curr_crop_unrot[:, :]>0)[0]) + 1
  sx = (t_max_x - t_min_x) / prev_crop.shape[1]
  sy = (t_max_y - t_min_y) / prev_crop.shape[0]
  return theta, sx, sy, rot_vis


def optimize(
  elements, centroids, targets, target_to_element, sx_init, sy_init, theta_init, tx_init, ty_init, z_init, shift_x, shift_y, bg_img,
  sx_prev=None, sy_prev=None, theta_prev=None, kx_prev=None, ky_prev=None, use_k=True, use_r=True, use_s=True, use_t=True, rgb_weight=1.0,
  use_z=True, bleed=0, blur_kernel=3, lr=0.01, n_steps=50, min_size=256, debug=False, p_weight=0.1, device='cpu'):
  shape_c_vars = [
    torch.tensor(tx_init, dtype=torch.float64, device=device),
    torch.tensor(ty_init, dtype=torch.float64, device=device),
    torch.tensor(sx_init, dtype=torch.float64, device=device),
    torch.tensor(sy_init, dtype=torch.float64, device=device),
    torch.tensor(theta_init, dtype=torch.float64, device=device),
    torch.tensor([0.0 for e in elements], dtype=torch.float64, device=device),
    torch.tensor([0.0 for e in elements], dtype=torch.float64, device=device)
  ]
  if sx_prev is None:
    sx_prev = sx_init
  if sy_prev is None:
    sy_prev = sy_init
  if theta_prev is None:
    theta_prev = theta_init
  if kx_prev is None:
    kx_prev = np.array([0.0 for e in elements])
  if ky_prev is None:
    ky_prev = np.array([0.0 for e in elements])
  default_c_vars = [
    torch.tensor(sx_prev, dtype=torch.float64, device=device),
    torch.tensor(sy_prev, dtype=torch.float64, device=device),
    torch.tensor(theta_prev, dtype=torch.float64, device=device),
    torch.tensor(kx_prev, dtype=torch.float64, device=device),
    torch.tensor(ky_prev, dtype=torch.float64, device=device),
  ]
  render_shape = None
  layer_z = torch.tensor([5.0 for e in elements], dtype=torch.float64, device=device)
  default_z = torch.tensor(z_init, dtype=torch.float64, device=device)

  # Compute origins for each previous shape element.
  origin = [
    torch.tensor([centroid[0], centroid[1]], dtype=torch.float64, device=device) for centroid in centroids
  ]
  origin = torch.stack(origin)

  # Set params to coarse params as starting point.
  output, params_shape, layer_z, alpha_shape, render_shape, targets, rgb_loss, alpha_loss, side_by_side, on_top, losses = compute_transforms(
    elements, targets, bg_img, target_to_element, c_variables=shape_c_vars, layer_z=layer_z, default_c_vars=default_c_vars, default_z=default_z, 
    use_k=use_k, use_r=use_r, use_s=use_s, use_t=use_t, rgb_weight=rgb_weight,
    blur_kernel=blur_kernel, origin=origin, min_size=min_size, lr=lr, n_steps=n_steps, p_weight=p_weight, bleed=bleed, loss_type='l1', device=device)
  shape_c_vars = [torch.tensor(v, dtype=torch.float64) for v in params_shape]
  
  return output, params_shape, layer_z, alpha_shape, render_shape, targets, rgb_loss.detach().cpu().numpy(), alpha_loss.detach().cpu().numpy(), side_by_side, on_top, losses


def warp_flo(x, flo, to_numpy=True):
  """
  warp an image/tensor (im2) back to im1, according to the optical flow
  x_arr: [H, W, C] (im2)
  flo: [B, 2, H, W] flow
  """
  B, C, H, W = x.size()
  # mesh grid
  xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
  yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
  xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
  yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
  grid = torch.cat((xx, yy), 1).float()

  if x.is_cuda:
    grid = grid.cuda()
  vgrid = grid + flo
  # scale grid to [-1,1]
  vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
  vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

  vgrid = vgrid.permute(0, 2, 3, 1)
  output = F.grid_sample(x, vgrid, mode='nearest')

  if to_numpy:
    output = output[0].permute(1, 2, 0)
    output = output.detach().cpu().numpy()
  return output


class InputPadder:
  """ Pads images such that dimensions are divisible by 8 """
  def __init__(self, dims, mode='sintel'):
    self.ht, self.wd = dims[-2:]
    pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
    pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
    if mode == 'sintel':
      self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    else:
      self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

  def pad(self, *inputs):
    return [F.pad(x, self._pad) for x in inputs]

  def unpad(self,x):
    ht, wd = x.shape[-2:]
    c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
    return x[..., c[0]:c[1], c[2]:c[3]]
