#!/bin/bash
clear

pip install pandas
python -m pip install -U scikit-image
pip install nibabel

pip uninstall --yes monai
pip install git+https://github.com/Project-MONAI/MONAI#egg=monai

pip uninstall --yes tqdm
pip install tqdm
