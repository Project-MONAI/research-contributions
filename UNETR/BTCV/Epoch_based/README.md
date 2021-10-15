# Model Overview
This repository contains the code for UNETR: Transformers for 3D Medical Image Segmentation [1]. UNETR is the first 3D segmentation network that uses a pure vision transformer as its encoder without relying on CNNs for feature extraction.
The code presents a volumetric (3D) multi-organ segmentation application using the BTCV challenge dataset.
![image](https://lh3.googleusercontent.com/pw/AM-JKLU2eTW17rYtCmiZP3WWC-U1HCPOHwLe6pxOfJXwv2W-00aHfsNy7jeGV1dwUq0PXFOtkqasQ2Vyhcu6xkKsPzy3wx7O6yGOTJ7ZzA01S6LSh8szbjNLfpbuGgMe6ClpiS61KGvqu71xXFnNcyvJNFjN=w1448-h496-no?authuser=0)

### Installing Dependencies
Dependencies can be installed using:
``` bash
./requirements.sh
```

### Training command
The following command initiates training using PyTorch native AMP package:
``` bash
python main.py
--feature_size=32 
--batch_size=1
--logdir=unetr_test
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5 
--save_checkpoint
```

To initiate distributed multi-gpu training, ```--distributed``` needs to be added to the training command.

To disable AMP, ```--noamp``` needs to be added to the training command. 

## Dataset
![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)

The training data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

Under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study. The 50 scans were captured during portal venous contrast phase with variable volume sizes (512 x 512 x 85 - 512 x 512 x 198) and field of views (approx. 280 x 280 x 280 mm3 - 500 x 500 x 650 mm3). The in-plane resolution varies from 0.54 x 0.54 mm2 to 0.98 x 0.98 mm2, while the slice thickness ranges from 2.5 mm to 5.0 mm. 

- Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4.Gallbladder 5.Esophagus 6. Liver 7. Stomach 8.Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12.Right adrenal gland 13.Left adrenal gland.
- Task: Segmentation
- Modality: CT  
- Size: 30 3D volumes (24 Training + 6 Testing)  
- Size: BTCV MICCAI Challenge

## Inference Details
Inference is performed in a sliding window manner with a window overlap specified in ```infer_overlap```.

# References
[1] Hatamizadeh, Ali, et al. "UNETR: Transformers for 3D Medical Image Segmentation.", 2021. https://arxiv.org/abs/2103.10504.
