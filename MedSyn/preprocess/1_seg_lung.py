import os
import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default="./nii_raw/")
    #parser.add_argument('--result_folder', type=str, default="./lung_mask_raw/")
    args = parser.parse_args()
    #assert args.batch_index != None
    #print("Batch index:", args.batch_index)
    
    img_folder = args.img_folder
    result_folder = "./lung_mask_raw/"
    
    moving_img_list = list(glob.glob(img_folder+"*.nii.gz"))

    for idx, line in enumerate(moving_img_list):
        #if idx % 4 != args.batch_index: # batch
        #    continue
        img_path = line
        if os.path.exists(result_folder+img_path.split("/")[-1]):
            continue
        # Main executive
        run_result = os.system("lungmask "+img_path+" "+result_folder+img_path.split("/")[-1])
