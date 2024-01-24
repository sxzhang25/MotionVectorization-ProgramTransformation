#!/bin/bash

BATCH_FILE=$1
THRESH=$2
MAX_FRAME=$3

while read -r line
do
	if [[ $line =~ ^#.*  ]]; then
		continue
	else
		EXT="${line##*.}"
		VID_NAME="${line%.*}"
		echo $VID_NAME
		rm -rf "../videos/${VID_NAME}/rgb"
		rm -rf "../videos/${VID_NAME}/flow"
		python3 -m motion_vectorization.preprocess --video_file $line --thresh $THRESH --max_frames $MAX_FRAME
	fi
done < "$BATCH_FILE"
