# Editing Motion Graphics Video via Motion Vectorization and Program Transformation

## Setup
This project was tested using Python 3.8 and CUDA v12.0 on Ubuntu 20.04.6. To set up your conda environment, please install the following packages:

```
- cv2
- numpy
- torch (v2.0.1)
- torchvision (v0.15.2)
- kornia (v0.6.12)
- networkx
- matplotlib
- scipy
- skimage
- pymatting
- pyefd
- PIL
- tqdm
- time
- tkinter
- json
- datetime
- drawsvg (v1.x)
- easing-functions (https://spicyyoghurt.com/tools/easing-functions)
```

#### Directory structure
The `motion_vectorization/` directory contains most of the files relevant to motion vectorization. Within this folder, you can modify the JSON files in the `config/` subdirectory to adjust the motion vectorization parameters.

The `scripts/` directory contains all scripts needed to run motion vectorization. The main scripts to run are `scripts/script.sh` and `convert_to_svg.sh`.

The `videos/` folder contains all video files (e.g., `demo.mp4`, `giftbox.mp4`, `test1.mp4`). All videos should be saved to this file. The file `videos/videos.txt` can be modified to adjust what videos to process.

## Motion Vectorization

#### Setup testing
If your setup is correct, you should be able to run

```
./scripts/script.sh videos/test.txt
```

This should produce a series of outputs in `motion_vectorization/outputs/test1_None/`, including the files `time_bank.pkl`, `shape_bank.pkl`, and `motion_file.json`. To see logs that might be useful for error tracking, you can check the `motion_vectorization/logs/` directory. **Note: This step may take several hours, especially for longer and more complicated videos.**

After this script is done running, you can generate an SVG from the motion file JSON file by running

```
./scripts/convert_to_svg.sh test1 30
```

The SVG should be saved as `motion_file.svg` under the same directory as above, `motion_vectorization/outputs/test1_None/`.

#### Processing your own videos
To process a new video, follow these steps:

1. Save the video as `${VIDEO_NAME}.mp4` under `videos/` (the video can also be a MOV file).
2. In the config directory `motion_vectorization/config/`, duplicate the file `default.json` to a file named `${VIDEO_NAME}.json` in the same directory. You can update any of the configs in the JSON file to guide the motion vectorization process. More information on the different configs is included below.
3. Add the line `${VIDEO_NAME}.mp4` to `videos/videos.txt` or any other `.txt` file in the `videos/` folder. Make sure to only include the names of videos you want to process in these folders, each on a separate line. If you do not wish to process a video in the file, you can comment it out with `#` at the beginning of the line.
4. Run `./scripts/script.sh videos/videos.txt`.
5. Run `./scripts/convert_to_svg.sh ${VIDEO_NAME} ${FRAME_RATE}`, where `${FRAME_RATE}` is your desired SVG animation frame rate.

#### Configs
The configs can be set to improve motion vectorization results, particularly in the tracking stage. The most relevant parameters are:

- `suffix`: This is a string that is appended to the output video folder. You can change it from `null` if you want to test out different configs and automatically save the resulting outputs in separate folders.
- `max_frames`: This is the index (use zero-indexing) of the last frame to process.
- `start_frame`: This is the index (use zero-indexing) of the first frame to process.
- `base_frame`: This is the frame to start processing frome. For example, if `start_frame` is 10, `max_frames` is 100, and `base_frame` is 50, we will first track from frame 50 to frame 100, and then track backwards from frame 50 to frame 10.
- `drop_thresh`: The threshold $\epsilon$ that defines what is an acceptable mapping error. Increasing this value means we are more tolerant to low-quality mappings.
- `use_k`, `use_r`, `use_s`, `use_t`: Set any of these to `false` if you know the objects in a video do not undergo skew, rotation, scale or translation, respectively.

## Program Transformation
*Coming soon!*
