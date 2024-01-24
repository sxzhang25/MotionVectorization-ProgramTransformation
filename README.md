# Editing Motion Graphics Video via Motion Vectorization and Program Transformation

## Setup
This project was tested using Python 3.8 and CUDA v12.0 on Ubuntu 20.04.6. To set up your conda environment, please install the following packages:

```
- cv2
- numpy
- torch (v2.0.1)
- torchvision (v0.15.2)
- kornia (v0.6.12)
- matplotlib
- tqdm
- time
- PIL
- tkinter
- json
- scipy
- skimage
- datetime
- networkx
- pymatting
- pyefd
- drawsvg (v1.x)
- easing-functions (https://spicyyoghurt.com/tools/easing-functions)
```

#### Directory structure
The `motion_vectorization/` directory contains most of the files relevant to motion vectorization. Within this folder, you can modify the JSON files in the `config/` subdirectory to adjust the motion vectorization parameters.

The `scripts/` directory contains all scripts needed to run motion vectorization. The main scripts to run are `scripts/script.sh` and `convert_to_svg.sh`.

The `videos/` folder contains all video files (e.g., `demo.mp4`, `giftbox.mp4`, `test1.mp4`). All videos should be saved to this file. The file `videos/videos.txt` can be modified to adjust what videos to process.

## Motion Vectorization

## Program Transformation
*Coming soon!*
