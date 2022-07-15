import os
import glob
import torch
import argparse
import imageio
import cv2
import numpy as np
from skimage import color, img_as_ubyte
from monai import transforms, data
from swinunetr import SwinUnetrModelForInference, SwinUnetrConfig

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device for model - cpu/gpu')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--last_n_frames', default=64, type=int, help='Limit the frames inference. -1 for all frames.')
args = parser.parse_args()


model = SwinUnetrModelForInference.from_pretrained('darragh/swinunetr-btcv-tiny')
model.eval()
model.to(args.device)

test_files = glob.glob('dataset/imagesSampleTs/*.nii.gz')
test_files = [{'image': f} for f in test_files]

test_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.AddChanneld(keys=["image"]),
        transforms.Spacingd(keys="image",
                            pixdim=(args.space_x, args.space_y, args.space_z),
                            mode="bilinear"),
        transforms.ScaleIntensityRanged(keys=["image"],
                                        a_min=args.a_min,
                                        a_max=args.a_max,
                                        b_min=args.b_min,
                                        b_max=args.b_max,
                                        clip=True),
        #transforms.Resized(keys=["image"], spatial_size = (256,256,-1)), 
        transforms.ToTensord(keys=["image"]),
    ])

test_ds = test_transform(test_files)
test_loader = data.DataLoader(test_ds,
                             batch_size=1,
                             shuffle=False)

for i, batch in enumerate(test_loader):
    
    tst_inputs = batch["image"]
    if args.last_n_frames>0:
        tst_inputs = tst_inputs[:,:,:,:,-args.last_n_frames:]
    
    with torch.no_grad():
        outputs = model(tst_inputs,
                            (args.roi_x,
                             args.roi_y,
                             args.roi_z),
                            8,
                            overlap=args.infer_overlap,
                            mode="gaussian")
        
    tst_outputs = torch.softmax(outputs.logits, 1)
    tst_outputs = torch.argmax(tst_outputs, axis=1)
    
    fnames = batch['image_meta_dict']['filename_or_obj']
    
    # Write frames to video
    for fname, inp, outp in zip(fnames, tst_inputs, tst_outputs):
        
        dicom_name = fname.split('/')[-1]
        video_name = f'videos/{dicom_name}.mp4'
        
        writer = imageio.get_writer(video_name,
                                    fps = 4,
                                    codec='mjpeg', 
                                    quality=10, 
                                    pixelformat='yuvj444p')
    
        for idx in range(inp.shape[-1]):
            # Segmentation
            seg = outp[:,:,idx].numpy().astype(np.uint8)
            # Input dicom frame
            img = (inp[0,:,:,idx]*255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            frame = color.label2rgb(seg,img, bg_label = 0)
            frame = img_as_ubyte(frame)
            frame = np.concatenate((img, frame), 1)
            writer.append_data(frame)
        writer.close()