#!/bin/bash

error() {
    echo "Usage: local-test.sh -i <input data directory> -o <output directory> -m <model directory>"
    echo "-c flag can be included to run on CPU"
    exit 1
}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
APP_DIR=$SCRIPT_DIR/../prostate_mri_lesion_seg_app

# Remove any existing output directory
rm -rf $SCRIPT_DIR/../output/*

# Set defaults for optional arguments
OUTPUT_DIR=$SCRIPT_DIR/../output
MODEL_DIR=$APP_DIR/models/

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

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
        -c | --cpu) CPU_ARG=1 ;;
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

# Check if CPU flag is included
if [ $CPU_ARG ]; then
    echo "Running local test on CPU..."
    time CUDA_VISIBLE_DEVICES='' python $APP_DIR -i $DATA_DIR -o $OUTPUT_DIR -m $MODEL_DIR
else
    echo "Running local test on GPU..."
    time python $APP_DIR -i $DATA_DIR -o $OUTPUT_DIR -m $MODEL_DIR
fi
