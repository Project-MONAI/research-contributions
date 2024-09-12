import argparse
import glob
import numpy as np
import torch
import copy
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib
import os
import skimage.io as io

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from func.model_arch import SegAirwayModel
from func.model_run import get_image_and_label, get_crop_of_image_and_label_within_the_range_of_airway_foreground, \
semantic_segment_crop_and_cat, dice_accuracy
from func.post_process import post_process, add_broken_parts_to_the_result, find_end_point_of_the_airway_centerline, \
get_super_vox, Cluster_super_vox, delete_fragments, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, fill_inner_hole
from func.detect_tree import tree_detection
from func.ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case,load_one_CT_img, \
get_df_of_centerline, get_df_of_line_of_centerline

parser = argparse.ArgumentParser(description='Data Preprocessing')
parser.add_argument('--job_id', type=int, default=0)
parser.add_argument('--num_job', type=int, default=8)
parser.add_argument('--data_path', type=str, default='/jet/home/lisun/work/r3/results/moved_img_nii/')
parser.add_argument('--output_path', type=str, default='/jet/home/lisun/work/r3/results/moved_img_nii_seg_airway_256_label_v3/')
parser.add_argument('--seg_real', type=bool, default=True) # real image contains 'Series' in name

def load_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model=SegAirwayModel(in_channels=1, out_channels=2)
    model.to(device)
    load_path = "model_para/checkpoint.pkl"
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model_semi_supervise_learning=SegAirwayModel(in_channels=1, out_channels=2)
    model_semi_supervise_learning.to(device)
    load_path = "model_para/checkpoint_semi_supervise_learning.pkl"
    checkpoint = torch.load(load_path)
    model_semi_supervise_learning.load_state_dict(checkpoint['model_state_dict'])
    
    return device, model, model_semi_supervise_learning

def segment_airway(img_path, device, model, model_semi_supervise_learning, args):
    
    file_name = img_path.split('/')[-1]
    #print(11, args.seg_real)
    if args.seg_real:
        my_list = file_name.split('_')
        sid_idx = my_list.index('Series') - 1
        sid = my_list[sid_idx]
        if os.path.exists(args.output_path+"/"+sid+"_Reg.npy"):
            print('skipped:', args.output_path+"/"+sid+"_Reg.npy")
            return
    else:
        sid = file_name.split('.')[0]
        if os.path.exists(args.output_path+"/"+sid+".npy"):
            print('skipped:', args.output_path+"/"+sid+".npy")
            return

    raw_img = load_one_CT_img(img_path)
    
    threshold = 0.5
    
    seg_result_semi_supervise_learning = semantic_segment_crop_and_cat(raw_img, model_semi_supervise_learning, device,
                                                                       crop_cube_size=[32, 128, 128], stride=[16, 64, 64],
                                                                       windowMin=-1000, windowMax=600)
    seg_onehot_semi_supervise_learning = np.array(seg_result_semi_supervise_learning>threshold, dtype=int)
    
    seg_result = semantic_segment_crop_and_cat(raw_img, model, device,
                                           crop_cube_size=[32, 128, 128], stride=[16, 64, 64],
                                           windowMin=-1000, windowMax=600)
    seg_onehot = np.array(seg_result>threshold, dtype=int)
    
    seg_onehot_comb = np.array((seg_onehot+seg_onehot_semi_supervise_learning)>0, dtype=int)
    seg_result_comb = (seg_result+seg_result_semi_supervise_learning)/2
    
    seg_processed,_ = post_process(seg_onehot_comb, threshold=threshold)
    
    if args.seg_real:
        np.save(args.output_path+"/"+sid+"_Reg.npy", seg_processed.astype(np.uint8))
    else:
        np.save(args.output_path+"/"+sid+".npy", seg_processed.astype(np.uint8))

def main():
    args = parser.parse_args()
    print(00, args.seg_real)
    img_list = list(glob.glob(args.data_path+'/*.nii.gz'))
    #img_list.reverse()

    device, model, model_semi_supervise_learning = load_model(args)
    
    for idx, img_path in enumerate(img_list):
        if idx % args.num_job != args.job_id:
            continue
            
        segment_airway(img_path, device, model, model_semi_supervise_learning, args)
        #exit()
        
    

if __name__ == '__main__':
    main()
