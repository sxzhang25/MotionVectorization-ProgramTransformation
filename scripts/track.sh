#!/bin/bash

LINE=$1

if [[ $LINE =~ ^#.*  ]]; then
	echo "skip ${LINE}"
else
	EXT="${LINE##*.}"
	VID_NAME="${LINE%.*}"
	LOGFILE="motion_vectorization/logs/${VID_NAME}_track_${RANDOM}.out"
	echo "Logging to ${LOGFILE}"
	python3 -m motion_vectorization.extract_shapes --video_file $LINE --config "motion_vectorization/config/${VID_NAME}.json" --verbose | tee $LOGFILE
fi
