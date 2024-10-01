#!/bin/bash

error() {
    echo "Usage: local-test.sh -i <input data directory> -o <output directory> -m <model directory>"
    echo "-b flag can be included to build MAP"
    echo "-c flag can be included to run on CPU"
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
        -o | --output) OUTPUT_DIR=$OPTARG ;;
        -m | --model) MODEL_DIR=$OPTARG ;;
        -b | --build) BUILD_ARG=1 ;;
        -c | --cpu) GPU_FLAG='' ;;
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
    monai-deploy package $APP_DIR -l DEBUG -t lesion_seg_workflow_app:v1.0 -m $MODEL_DIR -c $APP_DIR/app.yaml --platform x64-workstation
fi

echo "Running MAP..."
time docker run --cap-add CAP_SYS_PTRACE $GPU_FLAG \
        --env NVIDIA_DRIVER_CAPABILITIES=all \
        --env HOLOSCAN_HOSTING_SERVICE=HOLOSCAN_RUN \
        --env UCX_CM_USE_ALL_DEVICES=n \
        --env NVIDIA_VISIBLE_DEVICES=0 \
        --env HOLOSCAN_APPLICATION=/opt/holoscan/app \
        --env HOLOSCAN_INPUT_PATH=/var/holoscan/input \
        --env HOLOSCAN_OUTPUT_PATH=/var/holoscan/output \
        --env HOLOSCAN_WORKDIR=/var/holoscan \
        --env HOLOSCAN_MODEL_PATH=/opt/holoscan/models \
        --env HOLOSCAN_CONFIG_PATH=/var/holoscan/app.yaml \
        --env HOLOSCAN_APP_MANIFEST_PATH=/etc/holoscan/app.json \
        --env HOLOSCAN_PKG_MANIFEST_PATH=/etc/holoscan/pkg.json \
        --env HOLOSCAN_DOCS_PATH=/opt/holoscan/docs \
        --env HOLOSCAN_LOGS_PATH=/var/holoscan/logs \
        --group-add 44 \
        --ipc host \
        --network host \
        --rm \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --user 1000:1000 \
        --volume $DATA_DIR:/var/holoscan/input \
        --volume $OUTPUT_DIR:/var/holoscan/output \
        --workdir /var/holoscan \
        lesion_seg_workflow_app-x64-workstation-dgpu-linux-amd64:v1.0
