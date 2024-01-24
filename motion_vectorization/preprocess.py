# Extract frames.
import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# Video and directory information.
parser.add_argument(
  '--video_file', type=str, required=True, 
  help='Name of the video to process.')
parser.add_argument(
  '--video_dir', default='videos', 
  help='Directory containing videos.')
parser.add_argument(
  '--thresh', default=1e-4, type=float, 
  help='RGB difference threshold.')
parser.add_argument(
  '--min_dim', default=1024, type=int, 
  help='Minimum frame dimension.')
parser.add_argument(
  '--max_frames', default=-1, type=int, 
  help='Maximum number of frames to process.')
arg = parser.parse_args()

np.random.seed(0)

def main():
  video_name = os.path.splitext(arg.video_file.split('/')[-1])[0]
  video_folder = os.path.join(arg.video_dir, video_name)
  rgb_folder = os.path.join(video_folder, 'rgb')
  if not os.path.exists(rgb_folder):
    os.makedirs(rgb_folder)
  cap = cv2.VideoCapture(os.path.join(arg.video_dir, arg.video_file))
  prev_frame = None

  frame_idx = 0
  while True:
    if arg.max_frames >= 0:
      if frame_idx >= arg.max_frames:
        break
    
    _, frame = cap.read()
    if frame is None:
      break
    frame_height, frame_width, _ = frame.shape
    if arg.min_dim < 0:
      arg.min_dim = max(frame_height, frame_width)
    if frame_height > arg.min_dim or frame_width > arg.min_dim:
      resize_ratio = min(arg.min_dim / frame_height, arg.min_dim / frame_width)
      frame = cv2.resize(frame, (int(resize_ratio * frame_width), int(resize_ratio * frame_height)))
    save = True
    if frame_idx > 0:
      if np.mean(np.abs(frame / 255.0 - prev_frame / 255.0)) < arg.thresh:
        save = False
    if save:
      cv2.imwrite(os.path.join(rgb_folder, f'{frame_idx + 1:03d}.png'), frame)
    frame_idx += 1
    prev_frame = frame.copy()


if __name__ == '__main__':
  main()
