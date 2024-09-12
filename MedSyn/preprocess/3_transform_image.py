import os, glob
import SimpleITK as sitk

import numpy as np
from skimage.transform import resize
from joblib import Parallel, delayed

LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600
NUM_JOB = 2
RAW_IMAGE_FOLDER = "./nii_raw/"
OUTPUT_FOLDER = "./moved_img_256/"

def sub_job(batch_index):
    # Input raw images
    moving_img_list = list(glob.glob(RAW_IMAGE_FOLDER+"/*.nii.gz"))

    fixed_img = "../RAD_ChestCT/clamped_mask/Patient_0111262324_Study_CT_CHEST_WITHOUT_CONTRAST_42526394_Series_2_DR_30_0.625_Reg.nii.gz"

    for idx, moving_img in enumerate(moving_img_list):
        if idx % NUM_JOB != batch_index:
            continue
        subject_id = moving_img.split('/')[-1].split('.')[0]
        transform = "./transform_mask/"+subject_id+"_Reg_Atlas_Affine_0GenericAffine.mat"
        
        if not (os.path.exists(transform)) or (not os.path.exists(moving_img)):
            print(transform)
            print(moving_img)
            continue
            
        warped_img = OUTPUT_FOLDER + "/"+subject_id+"_Reg.nii.gz"

        run_result = os.system("antsApplyTransforms -d 3 -i "+moving_img+" -r "+fixed_img+\
                     " -o "+warped_img+" -n Linear -t "+transform+" -f -1024")

        if run_result == 0:
            result_img = sitk.ReadImage(warped_img)
            result_img = sitk.GetArrayFromImage(result_img)
            if idx == 0:
                print("img size:", result_img.shape) # (292, 316, 316)

            result_img = resize(result_img, (256, 256, 256), mode='constant', cval=LOW_THRESHOLD, preserve_range=True)
            result_img[result_img>HIGH_THRESHOLD] = HIGH_THRESHOLD
            result_img[result_img<LOW_THRESHOLD] = LOW_THRESHOLD
            
            result_img = (result_img - LOW_THRESHOLD) / (HIGH_THRESHOLD-LOW_THRESHOLD) # [-1024, 600] -> [0,1]
            result_img = 2*result_img-1 # [0,1] -> [-1,1]


            np.save(warped_img[:-7]+".npy", result_img)
            os.unlink(warped_img)
            
if __name__ == '__main__':
    Parallel(n_jobs=NUM_JOB)(delayed(sub_job)(item) for item in range(NUM_JOB))
