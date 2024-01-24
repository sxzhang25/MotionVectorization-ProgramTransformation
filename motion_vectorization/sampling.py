import kornia
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time


def downsample(img, scale):
  return F.avg_pool2d(img, scale, count_include_pad=False, stride=[1, 1])


def get_grid(w, h, bleed, device):
  grid_size = [w + 2 * bleed, h + 2 * bleed]
  xs, ys = np.meshgrid(np.linspace(-(w + 2 * bleed) / w, (w + 2 * bleed) / w, grid_size[0]), np.linspace(-(h + 2 * bleed) / h, (h + 2 * bleed) / h, grid_size[1]))
  grid = np.stack([xs, ys]).reshape((2, -1)).T
  return torch.tensor(grid, dtype=torch.float64, device=device)


def torch2numpy(tensor):
  if len(tensor.shape) == 3:
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
  else:
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
  return tensor


def sampling_layer(
  img, x, y, scale_x, scale_y, theta, shear_x, shear_y, size, 
  origin=None, blur=False, blur_kernel=3, bleed=0, interp='bilinear', device='cpu'):
  """
  Renders a set to 2D canvas
  :param img: NxCxIHxIW input patch
  :param x: N center location x coordinate system on canvas is -1 to 1
  :param y: N center location y
  :param scale_x: N scale of the element wrt to canvas x ratio of length w.r.t to total canvas length
  :param scale_y: N scale of the element wrt to canvas y
  """
  t0 = time.perf_counter()
  b, c, h, w = img.shape
  repeat = x.shape[0]
  if origin is None:
    origin = torch.zeros((repeat, 2)).to(device)
  else:
    origin = 2 * origin - 1

  # NOTE: https://discuss.pytorch.org/t/rotating-non-square-images-using-affine-grid/21592/2.
  shear_matrix_x = torch.stack(
    [torch.ones_like(shear_x), -shear_x, torch.zeros_like(x),
     torch.zeros_like(shear_x), torch.ones_like(shear_x), torch.zeros_like(y)], dim=1).view(-1, 2, 3)
  shear_matrix_y = torch.stack(
    [torch.ones_like(shear_y), torch.zeros_like(shear_y), torch.zeros_like(x),
     -shear_y, torch.ones_like(shear_y), torch.zeros_like(y)], dim=1).view(-1, 2, 3)
  aff_matrix = torch.stack(
    [torch.cos(-theta), -torch.sin(-theta), -x,
     torch.sin(-theta), torch.cos(-theta), -y], dim=1).view(-1, 2, 3)
  aff_matrix = aff_matrix.to(device)
  A_batch = aff_matrix[:, :, :2]
  b_batch = aff_matrix[:, :, 2].unsqueeze(1)
  Kx_batch = shear_matrix_x[:, :, :2].to(device)
  Ky_batch = shear_matrix_y[:, :, :2].to(device)

  coords = get_grid(size[3], size[2], bleed, device).unsqueeze(0).repeat(repeat, 1, 1)
  sc = torch.stack([scale_x, scale_y], dim=-1).to(device)
  coords = coords - origin[:, None, :]
  coords = coords + b_batch
  coords = coords.bmm(A_batch.transpose(1, 2))
  coords = coords.bmm(Ky_batch.transpose(1, 2))
  coords = coords.bmm(Kx_batch.transpose(1, 2))
  coords = coords / sc[:, None, :]
  coords = coords + origin[:, None, :]

  grid = coords.view(-1, size[3] + 2 * bleed, size[2] + 2 * bleed, 2)
  if blur:
    img = kornia.filters.gaussian_blur2d(img, (blur_kernel, blur_kernel), (blur_kernel, blur_kernel))
  if b == 1:
    img = img.repeat(repeat, 1, 1, 1)
  render =  F.grid_sample(img, grid, interp, 'zeros')
  t1 = time.perf_counter()
  return render


def main():
  # Coordinate grid has origin at top-left corner, increases left->right (x) and top->bottom (y).
  n_examples = 360
  # img = np.zeros((256, 448, 3))
  img = cv2.imread('data/bunny.png')
  if img.shape[0] > img.shape[1]:
    pad = ((0, 0), (0, img.shape[0] - img.shape[1]), (0, 0))
  else:
    pad = ((0, img.shape[1] - img.shape[0]), (0, 0), (0, 0))
  img = np.pad(img, pad)
  origin = (54, 128)
  bleed = 50
  rows, cols = [8, 8]
  for i in range(rows):
    img = cv2.line(img, (0, img.shape[0] * i // rows), (img.shape[1], img.shape[0] * i // rows), (0, 0, 255))
  for i in range(cols):
    img = cv2.line(img, (img.shape[1] * i // cols, 0), (img.shape[1] * i // cols, img.shape[0]), (0, 255, 0))
  img = cv2.circle(img, origin, 1, (255, 0, 0), 2)
  cv2.imshow('img', img)
  cv2.waitKey(0)
  img = torch.tensor(img / 255.0).repeat(n_examples, 1, 1, 1).permute(0, 3, 1, 2)
  size = (1, 3, img.shape[2], img.shape[3])
  # All inputs specified relative to origin, [-1.0, 1.0], where 1.0 is the right side of the canvas
  # and -1.0 is the left side (same for top and bottom).
  c_variables = [
    torch.tensor([0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([1.0 - i / n_examples for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([1.0 + i / n_examples for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([i * 2 * np.pi / n_examples for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([-0.0 * i for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 * i for i in range(n_examples)], dtype=torch.float64)
  ]  # tx, ty, sx, sy, theta, shear_x, shear_y
  # Origin coordinates are [-1.0, 1.0].
  sc = torch.tensor([size[3], size[2]])[None, ...]
  origin = torch.stack([
    torch.tensor([origin[0], origin[1]]) for i in range(n_examples)
  ])
  origin = origin / sc
  render = sampling_layer(
    img, 
    c_variables[0], 
    c_variables[1],
    c_variables[2], 
    c_variables[3], 
    c_variables[4], 
    c_variables[5], 
    c_variables[6],
    size, 
    origin=origin,
    bleed=bleed)
  render_pre = sampling_layer(
    img, 
    c_variables[0], 
    c_variables[1],
    c_variables[2], 
    c_variables[3], 
    c_variables[4], 
    c_variables[5], 
    c_variables[6],
    size, 
    origin=origin,
    bleed=bleed)
  render = torch2numpy(render)
  render_pre = torch2numpy(render_pre)
  img_numpy = torch2numpy(img)
  for i in range(render.shape[0]):
    cv2.imshow('original, render', np.concatenate([np.pad(img_numpy[i], ((bleed, bleed), (bleed, bleed), (0, 0))), np.ones((img_numpy.shape[1] + 2 * bleed, 10, 3)), render[i, :, :, :3], np.ones((img_numpy.shape[1] + 2 * bleed, 10, 3)), render_pre[i, :, :, :3]], axis=1))
    cv2.waitKey(0)

  img_crop = img[:, :, :128, :]
  c_variables = [
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([1.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([1.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64)
  ]
  c_variables_crop = [
    torch.tensor([0.25 * i for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([1.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([1.0 - 0.1 * i for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64),
    torch.tensor([0.0 for i in range(n_examples)], dtype=torch.float64)
  ]

  render = sampling_layer(img, c_variables[0], c_variables[1], c_variables[2], c_variables[3], c_variables[4], c_variables[5], c_variables[6], size, origin=None)
  render = torch2numpy(render)
  render_crop = sampling_layer(img_crop, c_variables_crop[0], c_variables_crop[1], c_variables_crop[2], c_variables_crop[3], c_variables_crop[4], c_variables_crop[5], c_variables[6], (1, 3, 128, 256), origin=None)
  render_crop = torch2numpy(render_crop)
  img_numpy = torch2numpy(img)
  img_crop = torch2numpy(img_crop)
  for i in range(render.shape[0]):
    cv2.imshow('original, render', np.concatenate([img_numpy[i], np.ones((img_numpy.shape[1], 10, 3)), render[i, :, :, :3]], axis=1))
    cv2.imshow('original, render_crop', np.concatenate([img_crop[i], np.ones((img_crop.shape[1], 10, 3)), render_crop[i, :, :, :3]], axis=1))
    cv2.waitKey(0)


if __name__ == '__main__':
  main()
