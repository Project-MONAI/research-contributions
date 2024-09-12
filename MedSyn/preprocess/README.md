# Pre-process Data for training MedSyn


The data preprocessing pipeline is composed of five steps, which are as follows:


### Step 1: Lung Segmentation from CT Scans for Registration Purposes

```bash
python 1_seg_lung.py --img_folder path_to/
```
The ``Patient_0111262324_Study_CT_CHEST_WITHOUT_CONTRAST_42526394_Series_2_DR_30_0.625_Reg_mask.nii.gz`` file is the lung mask of the Atlas image.
We use [lungmask](https://github.com/JoHof/lungmask) to segment lung, please install it.

### Step 2: Run registration

```bash
python 2_run_registration.py
```

We use registration on the lung mask for faster convergence and more robust performance. This is the most time-consuming step, it takes about 7 min per sample.

We use [ANTs](https://stnava.github.io/ANTs/) for image registration, please install it.

### Step 3: Apply registration transform to images

```bash
python 3_transform_image.py
```

### Step 4: Run vessel and lobe segmentation

```bash
python 4_run_vessel_seg.py
```

We use [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for vessel and lobe segmentation, please install it.


### Step 5:  Run airway segmentation

```bash
sh 5_run_airway_segment.sh
```
We use [NaviAirway](https://github.com/AntonotnaWang/NaviAirway) for airway segmentation, please install it.