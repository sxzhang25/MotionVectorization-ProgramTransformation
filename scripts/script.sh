#!/bin/bash

MAX_FRAME=500

for LIST in  "$@"
do
	echo $LIST

	echo "PREPROCESS"
	./scripts/preprocess.sh $LIST 0.0001 $MAX_FRAME

	echo "RAFT"
	./scripts/extract_flow.sh "${LIST}" $MAX_FRAME
done

for LIST in "$@"
do
	echo $LIST
	
	echo "CLUSTER"
	parallel -a $LIST ./scripts/extract_clusters.sh
	
	echo "TRACK"
	parallel -a $LIST ./scripts/track.sh
	
	echo "OPTIM"
	parallel -a $LIST ./scripts/optim.sh

	echo "PROGRAM"
	parallel -a $LIST ./scripts/motion_file.sh
done
