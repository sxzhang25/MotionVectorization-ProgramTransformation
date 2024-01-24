#!/bin/bash

LINE=$1

if [[ $LINE =~ ^#.*  ]]; then
	echo "skip ${LINE}"
else
	EXT="${LINE##*.}"
	VID_NAME="${LINE%.*}"
  	python3 -m motion_vectorization.create_motion_file --video_name ${VID_NAME} --output_dir motion_vectorization/outputs --config "motion_vectorization/config/${VID_NAME}.json"
fi