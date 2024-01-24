import numpy as np
import cv2
import networkx as nx

from .utils import pad_to_square, get_shape_coords
from .compositing import torch2numpy


class Visualizer():
  def __init__(self):
    pass

  def set_info(self, prev_labels, curr_labels, prev_frame, curr_frame, prev_fg_labels, curr_fg_labels, prev_fg_comps, curr_fg_comps, prev_fg_comp_to_label, curr_fg_comp_to_label, bg_img):
    self.prev_labels = prev_labels
    self.curr_labels = curr_labels
    self.prev_frame = prev_frame
    self.curr_frame = curr_frame
    self.prev_fg_labels = prev_fg_labels
    self.curr_fg_labels = curr_fg_labels
    self.prev_fg_comps = prev_fg_comps
    self.curr_fg_comps = curr_fg_comps
    self.prev_fg_comp_to_label = prev_fg_comp_to_label
    self.curr_fg_comp_to_label = curr_fg_comp_to_label
    self.bg_img = bg_img

  @staticmethod
  def _get_shape(labels, idx, frame, bg, size=25):
    shape_alpha = np.uint8(labels==idx)[..., None]
    min_x, min_y, max_x, max_y = get_shape_coords(shape_alpha)
    shape = frame * shape_alpha + bg * (1 - shape_alpha)
    shape = shape[min_y:max_y, min_x:max_x]
    shape = pad_to_square(shape)
    shape = cv2.resize(shape, (size, size))
    return shape

  @staticmethod
  def show_labels(labels):
    rng = np.random.default_rng(seed=1)
    colors = rng.integers(255, size=(np.max(labels) + 1, 3))
    colors[0] = [0, 0, 0]
    labels_vis = np.uint8(colors[labels])
    return labels_vis

  def main_matching(self, matching,):
    matching_img = 255 * np.ones((25 * max(len(self.prev_labels), len(self.curr_labels)), 100, 3), dtype=np.uint8)
    undrawn_prev = set(self.prev_labels)
    undrawn_curr = set(self.curr_labels)
    idx_i = 0
    idx_j = 0
    for prev, curr in matching:
      start_i = idx_i
      start_j = idx_j
      for i in prev:
        prev_shape_idx = self.prev_labels[i]
        shape = Visualizer._get_shape(self.prev_fg_labels, prev_shape_idx, self.prev_frame, self.bg_img)
        matching_img[idx_i * 25:(idx_i + 1) * 25, :25] = shape
        idx_i += 1
        if self.prev_labels[i] in undrawn_prev:
          undrawn_prev.remove(self.prev_labels[i])
      for j in curr:
        curr_shape_idx = self.curr_labels[j]
        shape = Visualizer._get_shape(self.curr_fg_labels, curr_shape_idx, self.curr_frame, self.bg_img)
        matching_img[idx_j * 25:(idx_j + 1) * 25, 75:] = shape
        idx_j += 1
        if self.curr_labels[j] in undrawn_curr:
          undrawn_curr.remove(self.curr_labels[j])
      for arrow_i, i in enumerate(prev):
        for arrow_j, j in enumerate(curr):
          matching_img = cv2.arrowedLine(
            matching_img, 
            (25, int(25 * (start_i + arrow_i + 0.5))), 
            (75, int(25 * (start_j + arrow_j + 0.5))), 
            (0, 0, 0), 1
          )
          matching_img = cv2.arrowedLine(
            matching_img, 
            (75, int(25 * (start_j + arrow_j + 0.5))), 
            (25, int(25 * (start_i + arrow_i + 0.5))), 
            (0, 0, 0), 1
          )
    
    for prev_shape_idx in undrawn_prev:
      shape = Visualizer._get_shape(self.prev_fg_labels, prev_shape_idx, self.prev_frame, self.bg_img)
      matching_img[idx_i * 25:(idx_i + 1) * 25, :25] = shape
      idx_i += 1
    for curr_shape_idx in undrawn_curr:
      shape = Visualizer._get_shape(self.curr_fg_labels, curr_shape_idx, self.curr_frame, self.bg_img)
      matching_img[idx_j * 25:(idx_j + 1) * 25, 75:] = shape
      idx_j += 1
    return matching_img

  @staticmethod
  def fallback_matching(matching, prev_shapes, curr_shapes):
    matching_img = 255 * np.ones((25 * max(len(prev_shapes), len(curr_shapes)), 100, 3), dtype=np.uint8)
    for i, shape in enumerate(prev_shapes):
      shape = pad_to_square(shape)
      shape = cv2.resize(shape, (25, 25))
      matching_img[i * 25:(i + 1) * 25, :25] = shape[:, :, :3]
    for j, shape in enumerate(curr_shapes):
      shape = pad_to_square(shape)
      shape = cv2.resize(shape, (25, 25))
      matching_img[j * 25:(j + 1) * 25, 75:] = shape[:, :, :3]

    for i, j in matching.items():
      matching_img = cv2.arrowedLine(
        matching_img, 
        (25, int(25 * (i + 0.5))), 
        (75, int(25 * (j + 0.5))), 
        (0, 0, 0), 1
      )
    return matching_img

  @staticmethod
  def clusters(frame, optim_bank, shape_layers, color, highest_res_update):
    track_vis = frame.copy()
    for label in np.unique(shape_layers)[1:]:
      track_vis[np.uint8(shape_layers==label)>0, :] = color[label]
    for label in np.unique(shape_layers)[1:]:
      cx, cy = optim_bank[label]['centroid']
      track_vis = cv2.circle(track_vis, (int(cx), int(cy)), 1, (255, 255, 255), 2)
      label_str = f'{str(label)}*' if highest_res_update[label] else str(label)
      track_vis = cv2.putText(
        track_vis, label_str, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return track_vis

  @staticmethod
  def target_to_element(target_to_element, elements, targets):
    max_elements = np.max(np.array([len(elems) for elems in target_to_element.values()]))
    t2e_img = 255 * np.ones((25 * len(targets), 25 * (max_elements + 1), 3), dtype=np.uint8)
    for i, target_idx in enumerate(target_to_element):
      target = targets[target_idx]
      target = pad_to_square(target)
      target = cv2.resize(target, (25, 25))
      t2e_img[i * 25:(i + 1) * 25, :25] = target[:, :, :3]
      for j, element_idx in enumerate(target_to_element[target_idx]):
        element = np.uint8(255 * torch2numpy(elements[element_idx]))
        element = pad_to_square(element)
        element = cv2.resize(element, (25, 25))
        t2e_img[i * 25:(i + 1) * 25, (j + 1) * 25:(j + 2) * 25] = element[:, :, :3]
    return t2e_img

  def matching_comp_groups(self, ceg):
    num_rows = 0
    for cc in nx.connected_components(ceg):
      counts = [0, 0, 0, 0]
      for v in cc:
        if v[0] == 'M':
          counts[0] += 1
        if v[0] == 'N':
          counts[1] += 1
        if v[0] == 'p':
          counts[2] += 1
        if v[0] == 'c':
          counts[3] += 1
      num_rows += np.max(counts)
    vis = 255 * np.ones((25 * num_rows, 7 * 25, 3), dtype=np.uint8)

    start_m, start_n, start_p, start_c = 0, 0, 0, 0
    for cc in nx.connected_components(ceg):
      M, N, P, C = [], [], [], []
      m_idx, n_idx, p_idx, c_idx = 0, 0, 0, 0
      for v in cc:
        if v[0] == 'M':
          comp_idx = int(v[1:])
          comp_rgb = Visualizer._get_shape(self.prev_fg_comps, comp_idx, self.prev_frame, self.bg_img)
          comp_rgb = cv2.resize(comp_rgb, (25, 25))
          vis[(start_m + m_idx) * 25:(start_m + m_idx + 1) * 25, :25] = comp_rgb
          M.append([int(v[1:]), m_idx])
          m_idx += 1
        if v[0] == 'N':
          comp_idx = int(v[1:])
          comp_rgb = Visualizer._get_shape(self.curr_fg_comps, comp_idx, self.curr_frame, self.bg_img)
          vis[(start_n + n_idx) * 25:(start_n + n_idx + 1) * 25, 150:] = comp_rgb
          N.append([int(v[1:]), n_idx])
          n_idx += 1
        if v[0] == 'p':
          shape_idx = int(v[1:])
          shape_rgb = Visualizer._get_shape(self.prev_fg_labels, shape_idx, self.prev_frame, self.bg_img)
          vis[(start_p + p_idx) * 25:(start_p + p_idx + 1) * 25, 50:75] = shape_rgb
          P.append([int(v[1:]), p_idx])
          p_idx += 1
        if v[0] == 'c':
          shape_idx = int(v[1:])
          shape_rgb = Visualizer._get_shape(self.curr_fg_labels, shape_idx, self.curr_frame, self.bg_img)
          vis[(start_c + c_idx) * 25:(start_c + c_idx + 1) * 25, 100:125] = shape_rgb
          C.append([int(v[1:]), c_idx])
          c_idx += 1

      # Draw connections.
      for m_arrow_idx, (comp_idx, m) in enumerate(M):
        for p_arrow_idx, (i, p) in enumerate(P):
          if i in self.prev_fg_comp_to_label[comp_idx]:
            vis = cv2.arrowedLine(
              vis, 
              (25, int(25 * (start_m + m + 0.5))), 
              (50, int(25 * (start_p + p + 0.5))), 
              (0, 0, 0), 1
            )
      for p_arrow_idx, (i, p) in enumerate(P):
        for c_arrow_idx, (j, c) in enumerate(C):
          if ceg.has_edge(f'p{i}', f'c{j}'):
            vis = cv2.arrowedLine(
              vis, 
              (75, int(25 * (start_p + p + 0.5))), 
              (100, int(25 * (start_c + c + 0.5))), 
              (0, 0, 0), 1
            )
      for n_arrow_idx, (comp_idx, n) in enumerate(N):
        for c_arrow_idx, (j, c) in enumerate(C):
          if j in self.curr_fg_comp_to_label[comp_idx]:
            vis = cv2.arrowedLine(
              vis, 
              (125, int(25 * (start_c + c + 0.5))), 
              (150, int(25 * (start_n + n + 0.5))), 
              (0, 0, 0), 1
            )
      new_start_idx = max(start_p + p_idx, start_c + c_idx)
      start_m = new_start_idx
      start_n = new_start_idx
      start_p = new_start_idx
      start_c = new_start_idx
    return vis

  def vis_graph(self, graph, mark_pos=False):
    vis = 255 * np.ones((25 * (len(self.prev_labels) + 1), 25 * (len(self.curr_labels) + 1), 3), dtype=np.uint8)
    for i in range(len(self.prev_labels)):
      shape = Visualizer._get_shape(self.prev_fg_labels, self.prev_labels[i], self.prev_frame, self.bg_img)
      vis[(i + 1) * 25:(i + 2) * 25, :25] = shape
    for j in range(len(self.curr_labels)):
      shape = Visualizer._get_shape(self.curr_fg_labels, self.curr_labels[j], self.curr_frame, self.bg_img)
      vis[:25, (j + 1) * 25:(j + 2) * 25] = shape
    for i in range(len(self.prev_labels)):
      for j in range(len(self.curr_labels)):
        color = (graph[i, j] * 255, graph[i, j] * 255, graph[i, j] * 255)
        vis[25 * (i + 1):25 * (i + 2), 25 * (j + 1):25 * (j + 2)] = color
        if mark_pos:
          if graph[i, j] > 0:
            vis[25 * (i + 1) + 10:25 * (i + 2) - 10, 25 * (j + 1) + 10:25 * (j + 2) - 10] = (0, 0, 255)
    return vis

  def vis_mapping(self, from_nodes, to_nodes, score, diff=None):
    if from_nodes[0][0] == 'p':
      prev = from_nodes
      curr = to_nodes
    else:
      prev = to_nodes
      curr = from_nodes
    vis = 255 * np.ones((25 * max(len(prev), len(curr)), 125, 3), dtype=np.uint8)
    for i, u in enumerate(prev):
      shape = Visualizer._get_shape(self.prev_fg_labels, int(u[1:]), self.prev_frame, self.bg_img)
      vis[i * 25:(i + 1) * 25, :25] = shape
    for j, v in enumerate(curr):
      shape = Visualizer._get_shape(self.curr_fg_labels, int(v[1:]), self.curr_frame, self.bg_img)
      vis[j * 25:(j + 1) * 25, 75:100] = shape
    vis = cv2.putText(
      vis, f'{score:.4f}', (30, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    if diff is not None:
      vis[:25, 100:] = cv2.resize(np.uint8(255 * diff), (25, 25))

    colors = np.random.randint(0, 255, (len(from_nodes), len(to_nodes), 3))
    for i, u in enumerate(from_nodes):
      for j, v in enumerate(to_nodes):
        if u[0] == 'p':
          vis = cv2.arrowedLine(
            vis, 
            (25, int(25 * (i + 0.5))), 
            (75, int(25 * (j + 0.5))), 
            colors[i, j].tolist(), 1
          )
        if u[0] == 'c':
          vis = cv2.arrowedLine(
            vis, 
            (75, int(25 * (i + 0.5))), 
            (25, int(25 * (j + 0.5))), 
            colors[i, j].tolist(), 1
          )
    return vis
      
  def nx_graph(self, graph):
    prev, curr = [], []
    for v in graph.nodes:
      if v[0] == 'p':
        prev.append(v)
      if v[0] == 'c':
        curr.append(v)
    # vis = 255 * np.ones((25 * (len(prev) + 1), 25 * (len(curr) + 1), 3), dtype=np.uint8)
    vis = 255 * np.ones((25 * max(len(prev), len(curr)), 100, 3), dtype=np.uint8)
    for i, u in enumerate(prev):
      shape = Visualizer._get_shape(self.prev_fg_labels, int(u[1:]), self.prev_frame, self.bg_img)
      vis[i * 25:(i + 1) * 25, :25] = shape
    for j, v in enumerate(curr):
      shape = Visualizer._get_shape(self.curr_fg_labels, int(v[1:]), self.curr_frame, self.bg_img)
      vis[j * 25:(j + 1) * 25, 75:] = shape

    colors = np.random.randint(0, 255, (len(prev), len(curr), 3))
    for i, u in enumerate(prev):
      for j, v in enumerate(curr):
        if graph.has_edge(u, v):
          vis = cv2.arrowedLine(
            vis, 
            (25, int(25 * (i + 0.5))), 
            (75, int(25 * (j + 0.5))), 
            colors[i, j].tolist(), 1
          )
        if graph.has_edge(v, u):
          vis = cv2.arrowedLine(
            vis, 
            (75, int(25 * (j + 0.5))), 
            (25, int(25 * (i + 0.5))), 
            colors[i, j].tolist(), 1
          )
    return vis

  def matching_digraph(self, prev_in_curr, curr_in_prev):
    vis = 255 * np.ones((25 * max(len(self.prev_labels), len(self.curr_labels)), 100, 3), dtype=np.uint8)

    colors = np.random.randint(0, 255, (len(self.prev_labels), len(self.curr_labels), 3))
    for i in range(prev_in_curr.shape[0]):
      prev_shape_idx = self.prev_labels[i]
      shape = Visualizer._get_shape(self.prev_fg_labels, prev_shape_idx, self.prev_frame, self.bg_img)
      vis[i * 25:(i + 1) * 25, :25] = shape
    for j in range(curr_in_prev.shape[0]):
      curr_shape_idx = self.curr_labels[j]
      shape = Visualizer._get_shape(self.curr_fg_labels, curr_shape_idx, self.curr_frame, self.bg_img)
      vis[j * 25:(j + 1) * 25, 75:] = shape

    for i in range(prev_in_curr.shape[0]):
      for j in range(prev_in_curr.shape[1]):
        if prev_in_curr[i, j] > 0:
          vis = cv2.arrowedLine(
            vis, 
            (25, int(25 * (i + 0.5))), 
            (75, int(25 * (j + 0.5))), 
            colors[i, j].tolist(), 1
          )
    for j in range(curr_in_prev.shape[0]):
      for i in range(curr_in_prev.shape[1]):
        if curr_in_prev[j, i] > 0:
          vis = cv2.arrowedLine(
            vis, 
            (75, int(25 * (j + 0.5))), 
            (25, int(25 * (i + 0.5))), 
            colors[i, j].tolist(), 1
          )

    return vis

  @staticmethod
  def concat_vis(vis_list, labels_list, pad=25):
    divider = 255 * np.ones((vis_list[0].shape[0], pad, 3), dtype=np.uint8)
    full_vis_list = [vis_list[0]]
    for i in range(1, len(vis_list)):
      full_vis_list.append(divider)
      full_vis_list.append(vis_list[i])
    full_vis = np.concatenate(full_vis_list, axis=1)
    border = 255 * np.ones((pad, full_vis.shape[1], 3), dtype=np.uint8)
    full_vis = np.concatenate([full_vis, border])
    for i, label in enumerate(labels_list):
      full_vis = cv2.putText(
        full_vis, label, (10 + i * (pad + vis_list.shape[1]), full_vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return full_vis
