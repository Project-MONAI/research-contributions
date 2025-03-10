#!/bin/bash

# Computes the dice scores for the organ mask and lesion mask outputs
#
# Usage:
# ./compare_output.sh <reference_dir> <output_dir>

# Check if the correct number of arguments were passed
if [ "$#" -ne 2 ]; then
    echo "Usage: ./compare_output.sh <reference_dir> <output_dir>"
    exit 1
fi

# Check if the reference directory exists
if [ ! -d "$1" ]; then
    echo "Reference directory does not exist"
    exit 1
fi

# Check if the output directory exists
if [ ! -d "$2" ]; then
    echo "Output directory does not exist"
    exit 1
fi

# Evaluate the dice scores for the organ mask and lesion mask outputs
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Evaluating Organ Mask..."
python $SCRIPT_DIR/eval_dice.py $1/organ/organ.nii.gz $2/organ/organ.nii.gz

echo "Evaluating Lesion Mask..."
python $SCRIPT_DIR/eval_dice.py $1/lesion/lesion_mask.nii.gz $2/lesion/lesion_mask.nii.gz
