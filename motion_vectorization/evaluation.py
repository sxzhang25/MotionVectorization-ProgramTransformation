import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
mpl.use('Agg')
from tqdm import tqdm


def get_numbers(dir):
  files = [os.path.splitext(f.name)[0].split('_')[0] for f in os.scandir(dir)]
  numbers = []
  for n in files:
    numbers_str = re.findall(r'\d+', n)
    numbers.extend([int(n_str) for n_str in numbers_str])
  return sorted(np.unique(numbers))

# video_names = []
# for filename in ['videos/eval.txt', 'videos/demo.txt']:
#   f = open(filename, 'r')
#   for video_name in f.readlines():
#     video_name = video_name.strip()
#     video_names.append(video_name.split('.')[0])
video_names = ['logo3.mp4']

eval = 1

if eval == 1:
  total_error = 0
  num_frames = 0
  for full_video_name in video_names:
    video_name = os.path.splitext(full_video_name)[0]
    if video_name[0] == '#':
      continue
    print('\nVIDEO:', video_name)
    video_out_dir = f'outputs/{video_name}_None'
    recon_folder = os.path.join(video_out_dir, 'outputs', 'recon')
    orig_folder = os.path.join('videos', video_name, 'rgb')
    diff_folder = os.path.join('videos', video_name, 'diff')
    if not os.path.exists(diff_folder):
      os.makedirs(diff_folder)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', ["blue", "cyan", "orange", "yellow"])
    norm = plt.Normalize(0, 1.5)
    frames = get_numbers(orig_folder)
    total_video_error = 0
    prev_frame = 0
    for frame in tqdm(frames):
      recon = cv2.imread(os.path.join(recon_folder, f'{frame:03d}.png'))
      orig = cv2.imread(os.path.join(orig_folder, f'{frame:03d}.png'))
      diff = np.linalg.norm(recon / 255.0 - orig / 255.0, axis=-1)
      min_val = np.min(diff)
      max_val = np.max(diff)
      avg_pixel_err = np.mean(diff)
      total_video_error += avg_pixel_err
      total_error += (frame - prev_frame) * avg_pixel_err
      fig = plt.figure()
      im = plt.imshow(diff, cmap=cmap, norm=norm)
      # plt.title(f'Diff (Average pixel error = {avg_pixel_err:.4f})')
      plt.axis('off')
      cbar = plt.colorbar(im, fraction=0.146, pad=0.02, shrink=0.5)
      # Get the default ticks and tick labels
      ticklabels = cbar.ax.get_ymajorticklabels()
      ticks = list(cbar.get_ticks())
      # Append the ticks (and their labels) for minimum and the maximum value
      cbar.set_ticks([0.0, 1.5])
      # cbar.set_ticklabels([min_val, max_val])
      fig.savefig(os.path.join(diff_folder, f'{frame:03d}.png'), format='png', bbox_inches='tight')
      plt.close()
      prev_frame = frame

    avg_video_error = total_video_error / frame
    num_frames += frame
    print(f'Avg error: {avg_video_error:.4f}')
  all_video_error = total_error / num_frames
  print(f'Avg pixel error over {num_frames} frames = {all_video_error:.4f}')

if eval == 2:
  print('VIDEO NAME    VIDEO (BYTES)    PROG (BYTES)    REDUCE (%)')
  total_reduce_amt = 0
  total_vid_size = 0
  total_prog_size = 0
  for full_video_name in video_names:
    video_name = os.path.splitext(full_video_name)[0]
    video_file_size = os.path.getsize(f'videos/{full_video_name}')
    total_vid_size += video_file_size / 100000
    program_file_size = os.path.getsize(f'outputs/{video_name}_None/motion_file.svg')
    total_prog_size += program_file_size / 100000
    reduce_amt = 100 * (1 - program_file_size / video_file_size)
    total_reduce_amt += reduce_amt
    print(f'{full_video_name:12}{video_file_size:14d}{program_file_size:17d}{reduce_amt:14.2f}')
  avg_vid_size = int(np.round(100000 * total_vid_size / len(video_names)))
  avg_prog_size = int(np.round(100000 * total_prog_size / len(video_names)))
  avg_reduce_amt = total_reduce_amt / len(video_names)
  avg = 'Average'
  print(f'{avg:12}{avg_vid_size:14d}{avg_prog_size:17d}{avg_reduce_amt:14.2f}')
  print(100 - avg_reduce_amt)
