# Model Overview

This repository contains the code for Swin UNETR [1,2]. Swin UNETR is the state-of-the-art on Medical Segmentation
Decathlon (MSD) and Beyond the Cranial Vault (BTCV) Segmentation Challenge dataset. In [1], a novel methodology is devised for pre-training Swin UNETR backbone in a self-supervised
manner. We provide the option for training Swin UNETR by fine-tuning from pre-trained self-supervised weights or from scratch.

The source repository for the training of these models can be found [here](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV).

# Installing Dependencies
Dependencies for training and inference can be installed using the model requirements :
``` bash
pip install -r requirements.txt
```

# Intended uses & limitations

You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task.

Note that this model is primarily aimed at being fine-tuned on tasks which segment CAT scans or MRIs on images in dicom format.  

# How to use

To be filled....

# Limitations and bias

The training data used for this model is specific to CAT scans from certain health facilities and machines. Data from other facilities may difffer in image distributions, and may require finetuning of the models for best performance.  

# Evaluation results

We provide several pre-trained models on BTCV dataset in the following.

<table>
  <tr>
    <th>Name</th>
    <th>Dice (overlap=0.7)</th>
    <th>Dice (overlap=0.5)</th>
    <th>Feature Size</th>
    <th># params (M)</th>
    <th>Self-Supervised Pre-trained </th>
  </tr>
<tr>
    <td>Swin UNETR/Base</td>
    <td>82.25</td>
    <td>81.86</td>
    <td>48</td>
    <td>62.1</td>
    <td>Yes</td>
</tr>

<tr>
    <td>Swin UNETR/Small</td>
    <td>79.79</td>
    <td>79.34</td>
    <td>24</td>
    <td>15.7</td>
    <td>No</td>
</tr>

<tr>
    <td>Swin UNETR/Tiny</td>
    <td>72.05</td>
    <td>70.35</td>
    <td>12</td>
    <td>4.0</td>
    <td>No</td>
</tr>

</table>

# Data Preparation
![image](https://lh3.googleusercontent.com/pw/AM-JKLX0svvlMdcrchGAgiWWNkg40lgXYjSHsAAuRc5Frakmz2pWzSzf87JQCRgYpqFR0qAjJWPzMQLc_mmvzNjfF9QWl_1OHZ8j4c9qrbR6zQaDJWaCLArRFh0uPvk97qAa11HtYbD6HpJ-wwTCUsaPcYvM=w1724-h522-no?authuser=0)

The training data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752).

- Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4.Gallbladder 5.Esophagus 6. Liver 7. Stomach 8.Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12.Right adrenal gland 13.Left adrenal gland.
- Task: Segmentation
- Modality: CT
- Size: 30 3D volumes (24 Training + 6 Testing)

# Training

See the source repository [here](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV) for information on training. 

# BibTeX entry and citation info
If you find this repository useful, please consider citing the following papers:

```
@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20730--20740},
  year={2022}
}

@article{hatamizadeh2022swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2201.01266},
  year={2022}
}
```

# References
[1]: Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[2]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.
