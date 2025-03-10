#!/bin/bash

error() {
    echo "Usage: local-test.sh -i <input data directory> -o <output directory> -m <model directory>"
    echo "-b flag can be included to build MAP"
    exit 1
}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
APP_DIR=$SCRIPT_DIR/../prostate_mri_lesion_seg_app

# Remove any existing output directory
rm -rf $SCRIPT_DIR/../output/*

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Set defaults for optional arguments
GPU_FLAG="--gpus all"
OUTPUT_DIR=$SCRIPT_DIR/../output
MODEL_DIR=$APP_DIR/models/

# Function to handle options and arguments
handle_options() {
    while [ $# -gt 0 ]; do
    case $1 in
        -i | --input)
            if ! has_argument $@; then
                echo "Error: No input data directory specified" && error
            fi
            DATA_DIR=$(extract_argument $@)
            shift
            ;;
        -o | --output)
            if ! has_argument $@; then
                echo "Error: No output directory specified" && error
            fi
            OUTPUT_DIR=$(extract_argument $@)
            shift
            ;;
        -m | --model)
            if ! has_argument $@; then
                echo "Error: No model directory specified" && error
            fi
            MODEL_DIR=$(extract_argument $@)
            shift
            ;;
        -b | --build) BUILD_ARG=1 ;;
      *) echo "Invalid option: $1" >&2 && error ;;
    esac
    shift
  done
}
handle_options "$@"

# Check if data dir exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Input data directory does not exist" && error
fi

# Build MAP if -b flag is included
if [ $BUILD_ARG ]; then
    echo "Building MAP since -b flag is included..."
    monai-deploy package $APP_DIR -l DEBUG -t lesion_seg_workflow_app:1.0 -m $MODEL_DIR -c $APP_DIR/app.yaml --platform x64-workstation
fi

echo "Running MAP..."
time monai-deploy run lesion_seg_workflow_app-x64-workstation-dgpu-linux-amd64:1.0 -i $DATA_DIR -o $OUTPUT_DIR
