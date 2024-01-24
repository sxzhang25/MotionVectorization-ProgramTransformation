import kornia
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR
import cv2
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# from scipy.interpolate import griddata
from tqdm import tqdm
import time

from .sampling import sampling_layer, torch2numpy

dsample = kornia.geometry.transform.PyrDown()


def place_shape(shapes, cxs, cys, sxs, sys, thetas, kxs, kys, frame_width, frame_height, bg=None, local_centroids=None, keep_alpha=True, interp='bilinear', device='cpu'):  
  channels = 4 if keep_alpha else shapes[0].shape[-1]
  shape_bgs = []
  if local_centroids is None:
    local_centroids = [None for shape in shapes]

  txs, tys = [], []
  local_cxs, local_cys = [], []
  for (shape, cx, cy, local_centroid) in zip(shapes, cxs, cys, local_centroids):
    if local_centroid is None:
      local_centroid = [shape.shape[1] / 2, shape.shape[0] / 2]
    local_cx, local_cy = local_centroid
    local_cxs.append(local_cx)
    local_cys.append(local_cy)
    shape_bg = np.pad(shape, ((0, max(frame_height, frame_width) - shape.shape[0]), (0, max(frame_height, frame_width) - shape.shape[1]), (0, 0)))
    shape_bg = torch.tensor(shape_bg / 255.0, device=device).permute(2, 0, 1)[None, ...]
    tx = 2 * (cx - local_cx) / max(frame_height, frame_width)
    ty = 2 * (cy - local_cy) / max(frame_height, frame_width)
    txs.append(tx)
    tys.append(ty)
    shape_bgs.append(shape_bg)

  if len(shape_bgs) < 1:
    shape_bgs = torch.zeros(len(shapes), channels, max(frame_height, frame_width), max(frame_height, frame_width), dtype=torch.float64, device=device)
  else:
    shape_bgs = torch.cat(shape_bgs)
  origin = torch.stack([
    torch.tensor(local_cxs, dtype=torch.float64, device=device), 
    torch.tensor(local_cys, dtype=torch.float64, device=device)
  ], axis=1) / max(frame_height, frame_width)
  shape_bgs = sampling_layer(
    shape_bgs, 
    torch.tensor(txs, dtype=torch.float64, device=device), 
    torch.tensor(tys, dtype=torch.float64, device=device), 
    torch.tensor(sxs, dtype=torch.float64, device=device), 
    torch.tensor(sys, dtype=torch.float64, device=device), 
    torch.tensor(thetas, dtype=torch.float64, device=device),
    torch.tensor(kxs, dtype=torch.float64, device=device),
    torch.tensor(kys, dtype=torch.float64, device=device),
    shape_bgs.shape,
    origin=origin,
    interp=interp,
    device=device
  )
  shape_bgs = shape_bgs[:, :, :frame_height, :frame_width]
  return shape_bgs
  

def init_optimizer(parameters, lr):
  tuple_params = list(j for i in parameters for j in (i if isinstance(i, list) else (i,)))
  for i in tuple_params:
    i.requires_grad = True
  optimizer = torch.optim.Adam(tuple_params, lr=lr, eps=1e-06, betas=(0.9, 0.90))#, betas=(0.65, 0.70))
  scheduler = CyclicLR(optimizer, base_lr=lr, max_lr=lr*2, cycle_momentum=False, mode= 'exp_range', step_size_up=1500)
  return optimizer, scheduler


def corr_loss(input, target):
  cost = 0.0
  for i in range(input.shape[1]):
    vx = input[0, i].flatten() - torch.mean(input[0, i].flatten())
    vy = target[0, i].flatten() - torch.mean(target[0, i].flatten())
    cost = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2))
  return torch.exp(-cost)


# TODO: Switch this to BCHW.
def masked_mse_loss(input, target, mask, n_channels=4):
  loss_fn = torch.nn.MSELoss(reduction='none')
  mse_loss_out = loss_fn(input * mask, target * mask)
  return torch.sum(mse_loss_out) / (n_channels * torch.sum(mask))


def sdf_loss_fn(input_sdf, target_sdf):
  mse_loss = torch.nn.MSELoss(reduction='none')
  return torch.mean(mse_loss(input_sdf, target_sdf))


def loss_fn(render, target, c_variables, layer_z=None, default_variables=None, default_z=None, use_mask=False, single_scale=False, blur_kernel=3, device='cpu', loss_type='mse', p_weight=0.1):
  if loss_type == 'mse':
    c_loss_fn = torch.nn.MSELoss(reduction='none')
  elif loss_type == 'l1':
    c_loss_fn = torch.nn.SmoothL1Loss(reduction='none')
  elif loss_type == 'corr':
    c_loss_fn = corr_loss
  else:
    c_loss_fn = torch.nn.MSELoss(reduction='none')
  mask = None
  if use_mask:
    mask = kornia.filters.gaussian_blur2d(render[:, 3:4, :, :], (2 * blur_kernel + 1, 2 * blur_kernel + 1), (2 * blur_kernel + 1, 2 * blur_kernel + 1))
  rgb_loss = c_loss_fn(render[:, :3, :, :], target[0][:, :3, :, :])
  alpha_loss = c_loss_fn(render[:, 3:4, :, :], target[0][:, 3:4, :, :])
  if use_mask:
    rgb_loss = torch.mean(mask * rgb_loss)
    alpha_loss = torch.mean(mask * alpha_loss)
  else:
    rgb_loss = torch.mean(rgb_loss, dim=[1, 2, 3])
    alpha_loss = torch.mean(alpha_loss, dim=[1, 2, 3])
  rgb_scales_loss = torch.zeros(render.shape[0], device=device, dtype=torch.float64)
  alpha_scales_loss = torch.zeros(render.shape[0], device=device, dtype=torch.float64)
  params_loss = torch.zeros(len(c_variables[0]), device=device, dtype=torch.float64)
  if default_variables is not None:
    for (c_var, d_var) in zip(c_variables[2:7], default_variables):
      params_loss += p_weight * c_loss_fn(c_var, d_var)
  if single_scale:
    return rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss
  render_d = render.clone()
  mask_d = mask.clone() if use_mask else None
  for i in range(1, len(target)):
    render_d = dsample(render_d)
    render_d = kornia.filters.gaussian_blur2d(render_d, (blur_kernel, blur_kernel), (blur_kernel, blur_kernel))
    if use_mask:
      mask_d = dsample(mask_d)
      mask_d = kornia.filters.gaussian_blur2d(mask_d, (blur_kernel, blur_kernel), (blur_kernel, blur_kernel))
    rgb_scales_loss_ = c_loss_fn(render_d[:, :3, :, :], target[i][:, :3, :, :])
    alpha_scales_loss_ = c_loss_fn(render_d[:, 3:4, :, :], target[i][:, 3:4, :, :])
    if use_mask:
      rgb_scales_loss += torch.mean(mask_d * rgb_scales_loss_, dim=[1, 2, 3])
      alpha_scales_loss += torch.mean(mask_d * alpha_scales_loss_, dim=[1, 2, 3])
    else:
      rgb_scales_loss += torch.mean(rgb_scales_loss_, dim=[1, 2, 3])
      alpha_scales_loss += torch.mean(alpha_scales_loss_, dim=[1, 2, 3])
  return rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss


def composite_layers(
  elements, c_variables, origin, groups, size, background, layer_z=None,
  blur=True, blur_kernel=3, debug=False, device='cpu'):
  t0 = time.perf_counter()
  render = sampling_layer(
    elements, 
    c_variables[0],  # tx
    c_variables[1],  # ty
    c_variables[2],  # sx
    c_variables[3],  # sy
    c_variables[4],  # theta
    c_variables[5],  # kx
    c_variables[6],  # ky
    size, 
    origin=origin, blur=blur, blur_kernel=blur_kernel, device=device)

  if debug:
    for i in range(elements.shape[0]):
      r = elements[i]
      cv2.imshow('render', torch2numpy(r))
      cv2.waitKey(0)

  render_alls, alphas = [], []
  for comp_idx in groups:
    render_all = render[groups[comp_idx]]
    n_soft_elements = len(groups[comp_idx])
    alpha, _ = torch.max(render_all[:, 3:4, :, :], dim=0, keepdims=True)
    if layer_z is None:
      alpha_z = render_all[:, 3:4, :, :]
    else:
      layer_z_group = torch.exp(layer_z[groups[comp_idx]])
      alpha_z = F.softmax(render_all[:, 3:4, :, :] * layer_z_group[:, None, None, None], dim=0)
    render_rgb = background[comp_idx]

    for e in range(n_soft_elements):
      element_mask = render_all[e:e + 1, 3:4, :, :]
      element_alpha = element_mask * alpha_z[e:e + 1, :, :, :]
      current_rgb = render_all[e:e + 1, :3, :, :]
      render_rgb = (1 - element_alpha) * render_rgb + element_alpha * current_rgb
    render_all = render_rgb
    render_alls.append(render_all)
    alphas.append(alpha)
  
  render_alls = torch.cat(render_alls, dim=0)
  alphas = torch.cat(alphas, dim=0)
  return render_alls, alphas


def compute_sdf(alpha, res=4):
  """Compute SDF of an input tensor alpha. Currently only supports batch size 1."""
  # Assign 1 to all pixels outside region, and -1 to all pixels inside region.
  region = 1.0 - 2.0 * alpha
  region = kornia.filters.gaussian_blur2d(region, (5, 5), (5, 5))

  # Uniformly sample pixels outside of region.
  b, c, h, w = alpha.shape
  cn, rn = w // res, h // res
  xs = torch.linspace(-1, 1, cn, dtype=torch.float64)
  ys = torch.linspace(-1, 1, rn, dtype=torch.float64)
  grid_x, grid_y = torch.meshgrid(ys, xs)
  grid = torch.stack([grid_y, grid_x], axis=-1).unsqueeze(0)
  
  # Get the closest distance to region.
  sc = torch.tensor([[h, w]])
  region_s = F.grid_sample(region, grid, padding_mode='reflection')
  region_mask = 100 * (torch.sigmoid(100 * (torch.abs(region_s) - 0.99))).view(-1, 1)
  xs_ = torch.linspace(0, 1, cn, dtype=torch.float64)
  ys_ = torch.linspace(0, 1, rn, dtype=torch.float64)
  coords = torch.stack(torch.meshgrid(ys_, xs_), axis=-1).view(-1, 2)
  dists = torch.cdist(coords, coords) + region_mask.T
  min_dists, _ = torch.min(dists, dim=1)
  min_dists = min_dists.view(-1, c, rn, cn)

  # Resample into full image.
  full_xs = torch.linspace(-1, 1, w, dtype=torch.float64)
  full_ys = torch.linspace(-1, 1, h, dtype=torch.float64)
  full_grid_x, full_grid_y = torch.meshgrid(full_ys, full_xs)
  full_grid = torch.stack([full_grid_y, full_grid_x], axis=-1).unsqueeze(0)
  full_res = F.grid_sample(min_dists, full_grid, padding_mode='reflection')
  full_res = full_res * torch.sign(region)
  return full_res


def main():
  device = 'cpu'
  print('Device:', device)
  test = 'opt'

  if test == 'opt':
    img = cv2.imread('data/opt_test1.png', cv2.IMREAD_UNCHANGED) / 255.0
    img = cv2.resize(img, (128, 128))
    bg_color = np.array([250, 252, 250])
    target = cv2.imread('data/opt_test2.png', cv2.IMREAD_UNCHANGED) / 255.0
    target_bg = np.ones((target.shape[0], target.shape[1], 3)) * bg_color / 255.0
    target[:, :, :3] = target[:, :, :3] * target[:, :, 3:4] + target_bg * (1 - target[:, :, 3:4])
    target = cv2.resize(target, (128, 128))
    gt_tensor = torch.tensor(target).permute(2, 0, 1)[None, ...]
    gt_scales = [gt_tensor.clone().to(device)]
    for i in range(3):
      gt_tensor_d = dsample(gt_scales[-1]).to(device)
      gt_scales.append(gt_tensor_d)
    elements = torch.tensor(img).permute(2, 0, 1)[None, ...]

    bg_color = torch.tensor(bg_color).to(device) / 255.0
    origin = torch.stack([torch.mean(torch.stack(torch.where(elements[i, 3, :, :]>0)).float(), axis=1) for i in range(elements.shape[0])])
    origin = torch.flip(origin, dims=[1])
    origin = origin.to(device)
    sc = torch.tensor([gt_tensor.shape[3], gt_tensor.shape[2]])[None, ...]
    sc = sc.to(device)
    origin = origin / sc
    
    c_variables = [
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([1.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([1.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64, device=device)
    ]  # tx, ty, sx, sy, theta, kx, ky
    default_c_vars = [v.clone() for v in c_variables]
    for v in default_c_vars:
      v.requires_grad = False
    layer_z = None
    c_variables = [v.to(device) for v in c_variables]

    opt_variables = []
    opt_variables.extend(c_variables)
    if layer_z is not None:
      opt_variables.append(layer_z)
    optimizer, scheduler = init_optimizer(opt_variables, lr=0.001)
    best_opt_variables = opt_variables.copy()
    loss = np.inf
    rgb_loss = np.inf
    alpha_loss = np.inf
    params_loss = np.inf
    min_loss = loss
    prev_min_loss = min_loss
    best_rgb_loss = rgb_loss
    best_rgb_scales_loss = np.inf
    best_alpha_loss = alpha_loss
    best_alpha_scales_loss = np.inf
    best_params_loss = params_loss
    best_render = None
    gt_frame = None
    perturb_round = 0
    total_step = 0
    blur_kernel = 3
    p_weight = 0.0
    loss_type = 'mse'
    use_mask = False
    side_by_side = []
    on_top = []
    opt_var_names = ['tx', 'ty', 'sx', 'sy', 'theta', 'kx', 'ky']

    render_all, alpha = composite_layers(
      elements, best_opt_variables[:7], origin, elements.shape, bg_color,
      layer_z=layer_z, blur=False, blur_kernel=blur_kernel, debug=False
    )
    best_render = torch2numpy(render_all.detach().cpu())[0]
    gt_frame = target[:, :, :3]
    sbs = np.uint8(255 * np.concatenate([best_render, gt_frame]))
    sbs = cv2.putText(sbs, 's0', (10, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    sbs = cv2.putText(sbs, f'{loss:.4f}', (sbs.shape[1] - 50, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    side_by_side.append(sbs)
    comp = np.uint8(255 * np.abs(best_render - gt_frame))
    comp = np.concatenate([comp, np.zeros([comp.shape[0], 250, 3], dtype=np.uint8)], axis=1)
    comp = cv2.putText(comp, 's0', (10, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    comp = cv2.putText(comp, f'loss: {loss:.4f}', (250, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i, (opt_var, opt_var_name) in enumerate(zip(best_opt_variables, opt_var_names)):
      comp = cv2.putText(comp, f'{opt_var_name}: {best_opt_variables[i][0]:.4f}', (135, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i, (loss_val, loss_name) in enumerate(zip([best_rgb_loss, best_rgb_scales_loss, best_alpha_loss, best_alpha_scales_loss, best_params_loss], ['rgb', 'rgb_s', 'alpha', 'alpha_s', 'param'])):
      comp = cv2.putText(comp, f'{loss_name}: {loss_val:.4f}', (250, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    for i in range(origin.shape[0]):
      ox, oy = origin[i].detach().cpu().numpy()
      comp = cv2.putText(comp, f'centroid: ({ox:.2f}, {oy:.2f})', (250, 10 * (i + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
    on_top.append(comp)
    cv2.imshow('ot', comp)
    cv2.imshow('ss', sbs)
    cv2.waitKey(0)    
    n_steps = 2000
    for step in range(n_steps):
      optimizer.zero_grad()
      blur = min_loss > 5e-2
      single_scale = min_loss < 0.01
      render_all, alpha = composite_layers(
        elements, c_variables, origin, elements.shape, bg_color,
        layer_z=layer_z, blur=blur, blur_kernel=blur_kernel, debug=False, device=device
      )
      # print(render_all.max(), gt_scale[0].max())
      rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss = loss_fn(
        torch.cat([render_all, alpha], dim=1), gt_scales, opt_variables[:6], layer_z=layer_z, default_variables=default_c_vars,
        loss_type=loss_type, single_scale=single_scale, use_mask=use_mask, p_weight=p_weight, device=device
      )
      loss = rgb_loss + rgb_scales_loss + alpha_loss + alpha_scales_loss + params_loss
      if loss.item() < min_loss:
        best_opt_variables = [v.clone() for v in opt_variables]
        min_loss = loss
        best_alpha_loss = alpha_loss
        best_alpha_scales_loss = alpha_scales_loss
        best_rgb_loss = rgb_loss
        best_rgb_scales_loss = rgb_scales_loss
        best_params_loss = params_loss
        best_render = torch2numpy(render_all.detach().cpu())[0]
        gt_frame = target[:, :, :3]
      loss.backward()
      optimizer.step()
      scheduler.step()

      if (total_step + 1) % 10 == 0:
        sbs = np.uint8(255 * np.concatenate([best_render, gt_frame]))
        sbs = cv2.putText(sbs, f's{total_step + 1}', (10, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        sbs = cv2.putText(sbs, f'{min_loss:.4f}', (sbs.shape[1] - 50, sbs.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        side_by_side.append(sbs)
        comp = np.uint8(255 * np.abs(best_render - gt_frame))
        comp = np.concatenate([comp, np.zeros([comp.shape[0], 250, 3], dtype=np.uint8)], axis=1)
        comp = cv2.putText(comp, f's{total_step + 1}', (10, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        comp = cv2.putText(comp, f'loss: {loss:.4f}', (250, comp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        for i, (opt_var, opt_var_name) in enumerate(zip(best_opt_variables, opt_var_names)):
          comp = cv2.putText(comp, f'{opt_var_name}: {best_opt_variables[i][0]:.4f}', (135, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        for i, (loss_val, loss_name) in enumerate(zip([best_rgb_loss, best_rgb_scales_loss, best_alpha_loss, best_alpha_scales_loss, best_params_loss], ['rgb', 'rgb_s', 'alpha', 'alpha_s', 'param'])):
          comp = cv2.putText(comp, f'{loss_name}: {loss_val:.4f}', (250, 10 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        for i in range(origin.shape[0]):
          ox, oy = origin[i].detach().cpu().numpy()
          comp = cv2.putText(comp, f'centroid: ({ox:.2f}, {oy:.2f})', (250, 10 * (i + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
        on_top.append(comp)
        cv2.imshow('ot', comp)
        cv2.imshow('ss', sbs)
        cv2.waitKey(1)
      total_step += 1

  if test == 'sdf':
    img = cv2.imread('data/A.png', cv2.IMREAD_UNCHANGED) / 255.0
    img = np.pad(img, ((25, 25), (25, 25), (0, 0)))
    img = cv2.resize(img, (200, 256))
    img_tensor = torch.tensor(img, dtype=torch.float64).permute(2, 0, 1).unsqueeze(0)
    gt_scales = [img_tensor.clone().to(device)]
    for i in range(3):
      gt_tensor_d = dsample(gt_scales[-1]).to(device)
      gt_scales.append(gt_tensor_d)

    elements = sampling_layer(
      img_tensor, 
      torch.tensor([0.2], dtype=torch.float64), 
      torch.tensor([0.05], dtype=torch.float64), 
      torch.tensor([0.7], dtype=torch.float64), 
      torch.tensor([0.9], dtype=torch.float64), 
      torch.tensor([0.2], dtype=torch.float64), 
      img_tensor.shape
    )
    bg_color = torch.tensor([0, 0, 0]).to(device)
    origin = torch.stack([torch.mean(torch.stack(torch.where(elements[i, 3, :, :]>0)).float(), axis=1) for i in range(elements.shape[0])])
    origin = torch.flip(origin, dims=[1])
    origin = origin.to(device)
    sc = torch.tensor([img_tensor.shape[3], img_tensor.shape[2]])[None, ...]
    sc = sc.to(device)
    origin = origin / sc

    shape_sdf = compute_sdf(img_tensor[:, 3:4, :, :])
    elements_sdf = compute_sdf(elements[:, 3:4, :, :])

    def show_sdf(shape, sdf):
      fig, ax = plt.subplots(1, 2)
      ax[0].imshow(shape)
      ax[0].set_title('Shape')
      ax[0].axis('off')
      sdf_plot = ax[1].imshow(sdf)
      plt.colorbar(sdf_plot, ax=ax[1], fraction=0.046, pad=0.04)
      ax[1].set_title('SDF')
      ax[1].axis('off')
      plt.show()

    show_sdf(img[:, :, 3], shape_sdf.detach().cpu().numpy()[0, 0])
    show_sdf(torch2numpy(elements)[0, :, :, 3], elements_sdf.detach().cpu().numpy()[0, 0])

    c_variables_rgb = [
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([1.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([1.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64)
    ]  # tx, ty, sx, sy, theta, z, alpha
    c_variables_sdf = [
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([1.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([1.0 for i in range(elements.shape[0])], dtype=torch.float64),
      torch.tensor([0.0 for i in range(elements.shape[0])], dtype=torch.float64)
    ]  # tx, ty, sx, sy, theta, z, alpha
    c_variables_rgb = [v.to(device) for v in c_variables_rgb]
    c_variables_sdf = [v.to(device) for v in c_variables_sdf]
    sdf_optimizer, sdf_scheduler = init_optimizer(c_variables_sdf, lr=0.001)
    rgb_optimizer, rgb_scheduler = init_optimizer(c_variables_rgb, lr=0.001)
    n_steps = 2000
    min_rgb_loss = 1000.0
    min_sdf_loss = 1000.0
    rgb_losses = []
    sdf_losses = []
    for step in tqdm(range(n_steps)):
      rgb_optimizer.zero_grad()
      sdf_optimizer.zero_grad()
      render_sdf, alpha_sdf = composite_layers(elements, c_variables_sdf, origin, elements.shape, bg_color, blur=False, device=device, debug=False)
      render_rgb, alpha_rgb = composite_layers(elements, c_variables_rgb, origin, elements.shape, bg_color, blur=False, device=device, debug=False)
      elements_sdf = compute_sdf(alpha_sdf)
      sdf_loss = sdf_loss_fn(elements_sdf, shape_sdf)
      rgb_loss, rgb_scales_loss, alpha_loss, alpha_scales_loss, params_loss = loss_fn(
        torch.cat([render_rgb, alpha_rgb], dim=1), gt_scales, c_variables_rgb)
      rgb_loss = rgb_loss + rgb_scales_loss
      if rgb_loss < min_rgb_loss:
        rgb_best_sdf = elements_sdf.squeeze().detach().cpu().numpy()
        rgb_best_render = render_rgb.clone()
        min_rgb_loss = rgb_loss
        rgb_best_c_variables = c_variables_rgb.copy()
      if sdf_loss < min_sdf_loss:
        sdf_best_sdf = elements_sdf.squeeze().detach().cpu().numpy()
        sdf_best_render = render_sdf.clone()
        min_sdf_loss = sdf_loss
        sdf_best_c_variables = c_variables_sdf.copy()
      sdf_loss.backward()
      rgb_loss.backward()
      sdf_optimizer.step()
      rgb_optimizer.step()
      sdf_scheduler.step()
      rgb_scheduler.step()

      if (step + 1) % 10 == 0:
        sdf_params = [v.detach().cpu().numpy() for v in sdf_best_c_variables]
        rgb_params = [v.detach().cpu().numpy() for v in rgb_best_c_variables]
        sdf_losses.append(sdf_loss)
        rgb_losses.append(rgb_loss)
        sdf_best_render_ = torch2numpy(sdf_best_render)[0]
        rgb_best_render_ = torch2numpy(rgb_best_render)[0]

    # Plot losses.
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(np.arange(0, len(sdf_losses)), sdf_losses, label='sdf')
    ax[0].set_title('SDF Loss')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Loss')
    ax[1].plot(np.arange(0, len(rgb_losses)), rgb_losses, label='rgb')
    ax[1].set_title('RGB Loss')
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Loss')
    fig.tight_layout()
    plt.show()

    # Plot final errors.
    sdf_best_render_ = torch2numpy(sdf_best_render)[0]
    rgb_best_render_ = torch2numpy(rgb_best_render)[0]
    pred_rgb = torch2numpy(elements)[0]
    gt_rgb = img[:, :, :3] * img[:, :, 3:4]
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Target image')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(pred_rgb)
    ax[0, 1].set_title('Input image')
    ax[0, 1].axis('off')
    rgb_err = np.linalg.norm(rgb_best_render_ - gt_rgb, axis=-1) ** 2.0
    rgb_plot = ax[1, 0].imshow(rgb_err)
    ax[1, 0].set_title('RGB L2 error')
    ax[1, 0].axis('off')
    plt.colorbar(rgb_plot, ax=ax[1, 0])
    sdf_err = np.linalg.norm(sdf_best_render_ - gt_rgb, axis=-1) ** 2.0
    sdf_plot = ax[1, 1].imshow(sdf_err)
    ax[1, 1].set_title('SDF L2 error')
    ax[1, 1].axis('off')
    plt.colorbar(sdf_plot, ax=ax[1, 1])
    plt.show()
  if test == 'place':
    shape = cv2.imread('data/beachball.png', cv2.IMREAD_UNCHANGED)
    width = 1920
    height = 1080
    x = 0.25
    y = 0.5
    for s in range(10):
      frame = place_shape(shape, int(x * width), int(y * height), 1 + 0.05 * s, 1 - 0.05 * s, 2 * np.pi / 10 * s, width, height, keep_alpha=True)
      cv2.imshow('frame', frame)
      cv2.waitKey(0)
  
if __name__ == '__main__':
  main()
