#!/bin/bash

BATCH_FILE=$1
MAX_FRAMES=$2

while read -r line
do
	if [[ $line =~ ^#.*  ]]; then
		continue
	else
		EXT="${line##*.}"
		VID_NAME="${line%.*}"
		echo $VID_NAME
		python3 -m RAFT.extract_flow --path "videos/${VID_NAME}" \
			--model RAFT/models/raft-sintel.pth --max_frames $MAX_FRAMES --add_back
	fi
done < "$BATCH_FILE"
