#!/bin/bash

# Usage: ./local-test.sh -i <input data directory> -o <output directory> -m <model directory>

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
APP_DIR=$SCRIPT_DIR/../prostate_mri_lesion_seg_app

# Read in command line arguments for i, o, m flags
while getopts i:o:m: flag
do

    if [ "$flag" == "i" ]; then
        DATA_DIR=$OPTARG
    fi
    if [ "$flag" == "o" ]; then
        OUTPUT_DIR=$OPTARG
    else
        OUTPUT_DIR=$SCRIPT_DIR/../output
    fi
    if [ "$flag" == "m" ]; then
        MODEL_DIR=$OPTARG
    else
        MODEL_DIR=$APP_DIR/models/
    fi

    # Print error if no data dir provided
    if [ -z "$DATA_DIR" ]; then
        echo "Error: No input data directory specified"
        echo "Usage: local-test.sh -i <input data directory> -o <output directory> -m <model directory>"
        exit 1
    fi

done

# Check if data dir exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Input data directory does not exist"
    echo "Usage: local-test.sh -i <input data directory> -o <output directory> -m <model directory>"
    exit 1
fi

# Set environment variable
export MONAI_MODELPATH=$MODEL_DIR

echo "Processing data in $DATA_DIR"
python $APP_DIR -i $DATA_DIR -o $OUTPUT_DIR -m $MODEL_DIR