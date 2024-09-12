# run vessel and lobe segmentation

import argparse
import os
import glob

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu_id', default=0, type=int)

def main():
    # read configurations
    args = parser.parse_args()
    gpu_id = args.gpu_id

    file_list = list(glob.glob("./moved_img_nii/*.nii.gz"))#[::-1]
    for idx, item in enumerate(file_list):
        sid = item.split('/')[-1][:-7] #.split('.')[0]
        #if idx % 4 != gpu_id:
        #    continue
        if os.path.exists('./moved_img_nii_seg/'+sid+'/lung_trachea_bronchia.nii.gz'):
            continue
        print(sid)    
        result = os.system('export CUDA_VISIBLE_DEVICES='+str(gpu_id)+'; TotalSegmentator -i moved_img_nii/'+sid+'.nii.gz -o moved_img_nii_seg/'+sid)
        if result == 0:
            os.system('export CUDA_VISIBLE_DEVICES='+str(gpu_id)+'; TotalSegmentator -i moved_img_nii/'+sid+'.nii.gz -o moved_img_nii_seg/'+sid+' -ta lung_vessels')

if __name__ == '__main__':
    main()
