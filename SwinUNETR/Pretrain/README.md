# Model Overview
![image](./assets/ssl_swin.png)

This repository contains the code for self-supervised pre-training of Swin UNETR model[1] for medical image segmentation. Swin UNETR is the state-of-the-art on Medical Segmentation
Decathlon (MSD) and Beyond the Cranial Vault (BTCV) Segmentation Challenge dataset.

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Pre-trained Models

We provide the self-supervised pre-trained weights for Swin UNETR backbone (CVPR paper [1]) in this <a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"> link</a>.
In the following, we describe steps for pre-training the model from scratch. 

## Datasets

The following datasets were used for pre-training (~5050 3D CT images). Please download the corresponding the json files of each dataset for more details and place them in ```jsons``` folder:

- Head & Neck Squamous Cell Carcinoma (HNSCC) ([Link](https://wiki.cancerimagingarchive.net/display/Public/HNSCC)) ([Download json](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/dataset_HNSCC_0.json))
- Lung Nodule Analysis 2016 (LUNA 16) ([Link](https://luna16.grand-challenge.org/Data/)) ([Download json](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/dataset_LUNA16_0.json))
- TCIA CT Colonography Trial ([Link](https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY/)) ([Download json](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/dataset_TCIAcolon_v2_0.json))
- TCIA Covid 19 ([Link](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19/)) ([Download json](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/dataset_TCIAcovid19_0.json))
- TCIA LIDC ([Link](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI/)) ([Download json](https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/dataset_LIDC_0.json))


### Training

#### Multi-GPU Training

To train a `Swin UNETR` encoder using multi-gpus:

```bash
python -m torch.distributed.launch --nproc_per_node=<Num-GPUs> --master_port=1234 main.py 
--batch_size=<Batch-Size> --num_steps=<Num-Steps> --lrdecay --eval_num=<Eval-Num> --logdir=<Exp-Num> --lr=<Lr>
```

#### Training from self-supervised weights on multiple GPUs (base model without gradient check-pointing)

To train a `Swin UNETR` encoder using a single gpu with gradient-checkpointing and a specified patch size:

```bash
python main.py --use_checkpoint --batch_size=<Batch-Size> --num_steps=<Num-Steps> --lrdecay 
--eval_num=<Eval-Num> --logdir=<Exp-Num> --lr=<Lr> --roi_x=<Roi_x> --roi_y=<Roi_y> --roi_z=<Roi_z>
```


## Citation
If you find this repository useful, please consider citing UNETR paper:

```
@article{tang2021self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  journal={arXiv preprint arXiv:2111.14791},
  year={2021}
}

@article{hatamizadeh2022swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2201.01266},
  year={2022}
}
```

## References
[1] Tang, Yucheng, et al. "Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis
", 2022. https://arxiv.org/abs/2111.14791.

[2] Hatamizadeh, Ali, et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", 2022. https://arxiv.org/abs/2201.01266.
