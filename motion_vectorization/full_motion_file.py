import json
import sys

video_name = sys.argv[1]
suffix = 'None'

motion_file_path = f'motion_vectorization/outputs/{video_name}_{suffix}/motion_file.json'
new_motion_file_path = f'motion_vectorization/outputs/{video_name}_{suffix}/motion_file_full.json'
motion_file = json.load(open(motion_file_path, 'rb'))


for shape_idx in motion_file:
  time = motion_file[shape_idx]['time']
  if shape_idx == '-1':
    params = ['bg_color']
  else:
    params = ['cx', 'cy', 'sx', 'sy', 'theta', 'kx', 'ky', 'z']

  lo_t = 0
  new_motion_file_params = {
    p: [] for p in params
  }
  for t in range(time[-1] + 1):
    if t > time[min(len(time) - 1, lo_t + 1)]:
      lo_t += 1
    hi_t = lo_t + 1
    for param in params:
      if t >= motion_file[shape_idx]['time'][0] and t <= motion_file[shape_idx]['time'][-1]:
        copy_param = motion_file[shape_idx][param][lo_t]
        new_motion_file_params[param].append(copy_param)
  for param in params:
    motion_file[shape_idx][param] = new_motion_file_params[param]
  motion_file[shape_idx]['time'] = [i for i in range(motion_file[shape_idx]['time'][0], motion_file[shape_idx]['time'][-1] + 1)]

with open(new_motion_file_path, 'w', encoding='utf-8') as handle:
  json.dump(motion_file, handle, ensure_ascii=False, indent=4)