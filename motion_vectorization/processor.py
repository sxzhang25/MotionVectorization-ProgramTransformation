import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
import io
import collections
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
# from pyefd import elliptic_fourier_descriptors
# from elliptic_fourier_descriptors import *
import networkx as nx
import matplotlib.pyplot as plt
import time

from .utils import warp_flo
from .shape_context import ShapeContext


def _get_moment_features3(sc, img, color=None):
  points, _ = sc.get_points_from_img(img[:, :, 3])
  desc = sc.compute(np.array(points))
  return desc


class Processor():
  def __init__(self): 
    self.sc = ShapeContext()
    pass

  @staticmethod
  def warp_labels(curr_labels, prev_labels, forw_flo, back_flo):
    '''
    labels: NxHxW
    flow: Nx2xHxW
    '''
    all_warped_labels = []
    for (labels, fg, flow) in zip([curr_labels, prev_labels], [prev_labels, curr_labels], [forw_flo, back_flo]):
      labels_tensor = torch.tensor(labels[..., None] + 1).permute(2, 0, 1)[None, ...].float()
      pad_h = labels.shape[0] - flow.shape[0]
      pad_w = labels.shape[1] - flow.shape[1]
      if pad_h < 0:
        flow = flow[-pad_h // 2:pad_h + (-pad_h) // 2]
      if pad_w < 0:
        flow = flow[:, -pad_w // 2:pad_w + (-pad_w) // 2]
      pad_h = max(0, pad_h)
      pad_w = max(0, pad_w)
      pad = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
      flow_pad = np.pad(flow, pad)
      flow_tensor = torch.tensor(flow_pad, dtype=torch.float32).permute(2, 0, 1)[None, ...]
      warped_labels = warp_flo(labels_tensor, flow_tensor) * np.uint8(fg>=0)[..., None]
      warped_labels = np.int32(warped_labels[:, :, 0] - 1)
      all_warped_labels.append(warped_labels)
    curr_warped_labels, prev_warped_labels = all_warped_labels
    curr_warped_labels[prev_labels<0] = -1
    prev_warped_labels[curr_labels<0] = -1
    return curr_warped_labels, prev_warped_labels

  @staticmethod
  def compute_match_graphs(curr_labels, prev_labels, curr_fg_labels, prev_fg_labels, curr_warped_labels, prev_warped_labels):
    curr_labels_stack = np.tile(curr_fg_labels, [len(curr_labels), 1, 1])
    curr_labels_stack = np.reshape(curr_labels_stack, (len(curr_labels), -1))
    prev_labels_stack = np.tile(prev_fg_labels, [len(prev_labels), 1, 1])
    prev_labels_stack = np.reshape(prev_labels_stack, (len(prev_labels), -1))
    curr_labels_mask = np.uint8(curr_labels_stack==curr_labels[..., None]) / 1.0
    prev_labels_mask = np.uint8(prev_labels_stack==prev_labels[..., None]) / 1.0
    curr_warped_labels_stack = np.tile(curr_warped_labels, [len(curr_labels), 1, 1])
    curr_warped_labels_stack = np.reshape(curr_warped_labels_stack, (len(curr_labels), -1))
    prev_warped_labels_stack = np.tile(prev_warped_labels, [len(prev_labels), 1, 1])
    prev_warped_labels_stack = np.reshape(prev_warped_labels_stack, (len(prev_labels), -1))
    curr_warped_labels_mask = np.uint8(curr_warped_labels_stack==curr_labels[..., None]) / 1.0
    prev_warped_labels_mask = np.uint8(prev_warped_labels_stack==prev_labels[..., None]) / 1.0
    prev_total = np.sum(prev_warped_labels_mask, axis=1)
    curr_total = np.sum(curr_warped_labels_mask, axis=1)
    prev_in_curr = prev_warped_labels_mask @ curr_labels_mask.T / prev_total[..., None]
    prev_in_curr[np.isnan(prev_in_curr)] = 0.0
    curr_in_prev = curr_warped_labels_mask @ prev_labels_mask.T / curr_total[..., None]
    curr_in_prev[np.isnan(curr_in_prev)] = 0.0
    return prev_in_curr, curr_in_prev

  @staticmethod
  def hungarian_matching(scores):
    row_ind, col_ind = linear_sum_assignment(scores)
    unmatched_prev = set([i for i in range(scores.shape[0])])
    unmatched_curr = set([j for j in range(scores.shape[1])])
    matching = []
    for r, c in zip(row_ind, col_ind):
      unmatched_prev.remove(r)
      matching.append([[r], [c]])
    return matching, unmatched_prev, unmatched_curr

  @staticmethod
  def main_matching(scores):
    unmatched_prev = set([i for i in range(scores.shape[0])])
    unmatched_curr = set([j for j in range(scores.shape[1])])

    # Make matching graph.
    B = nx.Graph()
    B.add_nodes_from([f'p{i}' for i in range(scores.shape[0])], bipartite=0)
    B.add_nodes_from([f'c{j}' for j in range(scores.shape[1])], bipartite=1)
    for i in range(scores.shape[0]):
      for j in range(scores.shape[1]):
        if scores[i, j] > 0:
          B.add_edge(f'p{i}', f'c{j}')
          if i in unmatched_prev:
            unmatched_prev.remove(i)
          if j in unmatched_curr:
            unmatched_curr.remove(j)

    matching = []
    cc = nx.connected_components(B)
    for C in cc:
      prev, curr = [], []
      for v in C:
        if v[0] == 'p':
          prev.append(int(v[1:]))
        if v[0] == 'c':
          curr.append(int(v[1:]))
      if len(prev) > 0 and len(curr) > 0:
        matching.append([prev, curr])
    return matching, unmatched_prev, unmatched_curr

  @staticmethod
  def get_appearance_graphs(prev_shapes, curr_shapes, prev_centroids, curr_centroids):
    # Compute shape features.
    sc = ShapeContext()
    prev_shape_features = []
    for prev_shape in prev_shapes:
      feature = _get_moment_features3(sc, prev_shape)
      prev_shape_features.append(feature)
    curr_shape_features = []
    for curr_shape in curr_shapes:
      feature = _get_moment_features3(sc, curr_shape)
      curr_shape_features.append(feature)
    shape_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
    t0 = time.perf_counter()
    for i in range(len(prev_shapes)):
      for j in range(len(curr_shapes)):
        shape_diffs[i, j] = np.exp(-sc.diff(prev_shape_features[i], curr_shape_features[j], idxs=False))
    t1 = time.perf_counter()

    # Compute shape RGB histograms.
    rgb_diffs = np.zeros((len(prev_shapes), len(curr_shapes)))
    for i, prev_shape in enumerate(prev_shapes):
      prev_shape_lab = cv2.cvtColor(prev_shape[:, :, :3], cv2.COLOR_BGR2LAB)
      prev_hist = cv2.calcHist(
        [prev_shape_lab[:, :, :3]], [0, 1, 2], prev_shape[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
      prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
      for j, curr_shape in enumerate(curr_shapes):
        curr_shape_lab = cv2.cvtColor(curr_shape[:, :, :3], cv2.COLOR_BGR2LAB)
        curr_hist = cv2.calcHist(
          [curr_shape_lab[:, :, :3]], [0, 1, 2], curr_shape[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        rgb_diff = 1.0 - cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        rgb_diffs[i, j] = rgb_diff

    return shape_diffs, rgb_diffs

  @staticmethod
  def fallback_matching(prev_shapes, curr_shapes, prev_centroids, curr_centroids, frame_width, frame_height, thresh=0.6):
    shape_diffs, rgb_diffs = Processor().get_appearance_graphs(
      prev_shapes, curr_shapes, prev_centroids, curr_centroids)

    # Compute centroid distances.
    dists = cdist(np.array(prev_centroids), np.array(curr_centroids))
    costs = (-np.log(shape_diffs) + (1.0 - rgb_diffs) + dists) / 3
    row_ind, col_ind = linear_sum_assignment(costs)
    matching = {}
    for i, j in zip(row_ind, col_ind):
      if j >= 0 and costs[i, j] < thresh:
        matching[i] = j
    return matching, costs

  @staticmethod
  def compute_matching_comp_groups(matching, prev_labels, curr_labels, prev_fg_comp_to_label, curr_fg_comp_to_label):
    ceg = nx.Graph()
    ceg.add_nodes_from([f'p{i}' for i in prev_labels])
    ceg.add_nodes_from([f'c{j}' for j in curr_labels])
    ceg.add_nodes_from([f'M{c}' for c in prev_fg_comp_to_label])
    ceg.add_nodes_from([f'N{c}' for c in curr_fg_comp_to_label])
    for prev, curr in matching:
      for i in prev:
        for j in curr:
          ceg.add_edges_from([(f'p{prev_labels[i]}', f'c{curr_labels[j]}')])
    for c in prev_fg_comp_to_label:
      ceg.add_edges_from([(f'M{c}', f'p{i}') for i in prev_fg_comp_to_label[c]])
    for c in curr_fg_comp_to_label:
      ceg.add_edges_from([(f'N{c}', f'c{j}') for j in curr_fg_comp_to_label[c]])

    matching_comp_groups = []
    for C in nx.connected_components(ceg):
      prev_comps = []
      curr_comps = []
      for v in C:
        if v[0] == 'M':
          prev_comps.append(int(v[1:]))
        if v[0] == 'N':
          curr_comps.append(int(v[1:]))
      matching_comp_groups.append([prev_comps, curr_comps])
    return matching_comp_groups, ceg

  @staticmethod
  def compare_shapes(shape_a, shape_b, rgb_method='hist'):
    shape_a_features = _get_moment_features2(shape_a)
    shape_b_features = _get_moment_features2(shape_b)
    # shape_diff = 1.0 - shape_a_features @ shape_b_features.T
    shape_diff = np.exp(-np.abs(shape_a_features - shape_b_features))

    if rgb_method == 'hist':
      shape_a_hist = cv2.calcHist(
        [shape_a[:, :, :3]], [0, 1, 2], shape_a[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
      shape_a_hist = cv2.normalize(shape_a_hist, shape_a_hist).flatten()
      shape_b_hist = cv2.calcHist(
        [shape_b[:, :, :3]], [0, 1, 2], shape_b[:, :, 3], [64, 64, 64], [0, 256, 0, 256, 0, 256])
      shape_b_hist = cv2.normalize(shape_b_hist, shape_b_hist).flatten()
      rgb_diff = 1.0 - cv2.compareHist(shape_a_hist, shape_b_hist, cv2.HISTCMP_BHATTACHARYYA)
    else:
      shape_a_lab = cv2.cvtColor(shape_a[:, :, :3], cv2.COLOR_BGR2HSV)
      shape_b_lab = cv2.cvtColor(shape_b[:, :, :3], cv2.COLOR_BGR2HSV)
      shape_a_lab = np.mean(np.reshape(shape_a_lab, (-1, 3)), axis=0)
      shape_b_lab = np.mean(np.reshape(shape_b_lab, (-1, 3)), axis=0)
      rgb_diff = np.linalg.norm(shape_a_lab - shape_b_lab)

    return shape_diff, rgb_diff