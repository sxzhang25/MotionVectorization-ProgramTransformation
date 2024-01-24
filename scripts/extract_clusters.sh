#!/bin/bash

LINE=$1

if [[ $LINE =~ ^#.*  ]]; then
	echo "skip ${LINE}"
else
	EXT="${LINE##*.}"
	VID_NAME="${LINE%.*}"
	echo $VID_NAME
	python3 -m motion_vectorization.extract_clusters --video_file $LINE --config "motion_vectorization/config/${VID_NAME}.json"
fi