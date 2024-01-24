import os
import numpy as np
import cv2
import torch
from scipy import interpolate

from .utils import get_numbers

class DataLoader():
  def __init__(self, video_dir, max_frames=-1):
    self.dir = video_dir
    # self.dtype = dtype
    # self.device = device

    # Directories.
    self.frame_folder = os.path.join(video_dir, 'rgb')
    self.labels_folder = os.path.join(video_dir, 'labels')
    self.fgbg_folder = os.path.join(video_dir, 'fgbg')
    self.comps_folder = os.path.join(video_dir, 'comps')
    self.forw_flow_folder = os.path.join(video_dir, 'flow', 'forward')
    self.back_flow_folder = os.path.join(video_dir, 'flow', 'backward')
    
    # Get total number of frames.
    self.frame_idxs = get_numbers(self.frame_folder)
    self.pos = 0
    if max_frames >= 0:
      self.frame_idxs = self.frame_idxs[:max_frames]

  def load_data(self, i):
    '''loads data for processing frame i'''
    frame = cv2.imread(os.path.join(self.frame_folder, f'{self.frame_idxs[i]:03d}.png'))
    labels = np.load(os.path.join(self.labels_folder, f'{self.frame_idxs[i]:03d}.npy'))
    comps = np.load(os.path.join(self.comps_folder, f'{self.frame_idxs[i]:03d}.npy'))
    fg_bg = np.load(os.path.join(self.fgbg_folder, f'{self.frame_idxs[i]:03d}.npy'))
    if i == 0:
      forw_flow = np.zeros((frame.shape[0], frame.shape[1], 2))
      back_flow = np.zeros((frame.shape[0], frame.shape[1], 2))
    else:
      forw_flow = np.load(os.path.join(self.forw_flow_folder, f'{self.frame_idxs[i - 1]:03d}.npy'))
      back_flow = np.load(os.path.join(self.back_flow_folder, f'{self.frame_idxs[i - 1]:03d}.npy'))

    return frame, labels, fg_bg, comps, forw_flow, back_flow


    