#!/bin/bash

# Usage: ./build-and-run-MAP.sh -i <input data directory> -o <output directory> -m <model directory>

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
        echo "Usage: build-and-run-MAP.sh -i <input data directory> -o <output directory> -m <model directory>"
        exit 1
    fi

done

# Check if data dir exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Input data directory does not exist"
    echo "Usage: build-and-run-MAP.sh -i <input data directory> -o <output directory> -m <model directory>"
    exit 1
fi

echo "Packaging MAP..."
monai-deploy package -l DEBUG -b nvcr.io/nvidia/pytorch:22.08-py3 $APP_DIR --tag lesion_seg_workflow_app:v1.0 -m $MODEL_DIR

echo "Running MAP..."
monai-deploy run lesion_seg_workflow_app:v1.0 $DATA_DIR $OUTPUT_DIR