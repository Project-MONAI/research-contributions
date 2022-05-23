#!/usr/bin/env bash

python main.py --json_list='/home/ali/Desktop/data_local/Synapse_Orig/dataset_0.json' --data_dir='/home/ali/Desktop/data_local/Synapse_Orig' \
--val_every=50 --use_normal_dataset --roi_x=96 --roi_y=96 --roi_z=96  --in_channels=1 --spatial_dims=3 --use_checkpoint --feature_size=48 --use_ssl_pretrained