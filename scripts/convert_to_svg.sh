#!/bin/bash

VIDEO_NAME=$1
FRAME_RATE=$2

python3 -m motion_vectorization.create_motion_file --video_name $VIDEO_NAME --config "motion_vectorization/config/${VIDEO_NAME}.json"
python3 -m motion_vectorization.full_motion_file $VIDEO_NAME
python3 -m svg_utils.create_svg_dense --video_dir "motion_vectorization/outputs/${VIDEO_NAME}_None" --frame_rate $2