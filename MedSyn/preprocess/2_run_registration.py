import os
import SimpleITK as sitk
import argparse
import glob

def clamp_img(img_path):
    img=sitk.ReadImage(img_path)
    img_arr=sitk.GetArrayFromImage(img)
    img_arr[img_arr==1]=50
    img_arr[img_arr==2]=100
    new_img=sitk.GetImageFromArray(img_arr)
    new_img.CopyInformation(img)
    new_img=sitk.Cast(new_img, sitk.sitkInt16)
    basename = img_path.split('/')[-1]
    sitk.WriteImage(new_img, "./clamped_mask/"+basename+".nii.gz")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--batch_index', type=int)
    args = parser.parse_args()
    #assert args.batch_index != None
    #print("Batch index:", args.batch_index)
    
    # create output folders
    os.makedirs("./image_mask/", exist_ok = True)
    os.makedirs("./transform_mask/", exist_ok = True)
    os.makedirs("./clamped_mask/", exist_ok = True)

    moving_img_list = list(glob.glob("./lung_mask_raw/*.nii.gz"))

    # lung mask of the Atlas image
    fixed_img = "./Patient_0111262324_Study_CT_CHEST_WITHOUT_CONTRAST_42526394_Series_2_DR_30_0.625_Reg_mask.nii.gz"
    assert os.path.exists(fixed_img)

    for idx, line in enumerate(moving_img_list):
        #if idx % 8 != args.batch_index: # batch
        #    continue
        basename = line.split('/')[-1]
        
        moving_img = "./clamped_mask/" + basename
        warped_img = basename[:-7]
        if os.path.exists("./transform_mask/"+warped_img+"_Reg_Atlas_Affine_0GenericAffine.mat"): # finished
            continue

        if not os.path.exists(moving_img):
            clamp_img(line) # Clamp intensity        

        # Main executive
        run_result = os.system("antsRegistration -d 3 -o [./transform_mask/"+warped_img+"_Reg_Atlas_Affine_,./image_mask/"+warped_img+"_Reg_Atlas_Affine.nii.gz] -r ["+fixed_img+", "+moving_img+",1] -t Affine[0.01] -m MI["+fixed_img+", "+moving_img+",1,32,Regular,0.5] -c [500x250x100] -s 2x1x0 -f 4x2x1")

        if run_result == 0:
            try:
                os.unlink("./image_mask/"+warped_img+"_Reg_Atlas_Affine.nii.gz")
            except Exception as e:
                continue
