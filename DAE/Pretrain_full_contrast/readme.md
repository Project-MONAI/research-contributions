
## Model Overview

This repository contains the code for Disruptive Auto Encoders (DAE), the DAE model is focused towards learning better representations as it focuses on low-level features of the data via self-supervised learning.

The below figure provides an overview that the 3D medical image is disrupted with a combination of low-level perturbations - noise and downsampling, followed by tokenization and local masking. These disrupted tokens are then passed through a transformer encoder and convolutional decoder to learn to reconstruct the original image. Our method also includes cross modal contrastive learning to bring in modality-awareness to the pre-training framework. This can act as an effective pre-training strategy to extract meaningful low-level representations for 3D medical image analysis.

<img src=figs/dae_overview.png width="400" height="530" align="center">

The reconstruction quality is an indicator that low-level pretraining can provide better representations, we can clearly observe that finer features of the image are neglected during reconstruction when using masking based techniques


<img src=figs/dae_recon.png width="400" height="530" align="center">


## Installing Dependencies

Dependencies can be installed using:

```bash
pip install -r requirements.txt
```

## Using the code:

The pretraining Code is tested with Swin UNETR as backbone. It trains on CT+MRI data with the new Cross Modal Contrastive loss. The current code base has 3 pretraining options:

1) Denoising
2) Super-Resolution
3) Local Disturbance

Change the argument --choice to switch between different pretraining setups

Use --onlycovid tag to train only on the COVID Data.
Use --encoder_off tag to switch off encoder during training (not generally used)
Use --decoder_off tag to switch off decoder during training (not generally used)

1) Denoising

Sample Code:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_runner.py --batch_size=2 --sw_batch_size=1 --mask_ratio=0.6 --epoch=300 --mask_patch_size=16 --img_size=96 --min_lr=1e-5 --warmpup_epoch=4 --loss_type=all_img --base_lr=1e-4 --warmup_lr=1e-6 --weight_decay=0.05 --cache_dataset --cache_rate=1 --model_type=swin_skip --save_freq=5 --print_freq=1 --log_dir="./logdir/swin_denoise_mm" --output="./output/swin_denoise_mm" --thread_loader --out_channels=1 --choice "denoise" --variance 0.1 --mm_con 0.03 --temperature 0.5 --amp_opt_level="O0"
```

--variance can be use to change the variance of gaussian noise. Default=0.1

--mm_con is used to change ratio of CFM loss. Default=0.03

--temperature is used to change temperature in CFM loss. Default=0.5

We don't use AMP in these experiments as it causes CFM loss to go NaN

2) Super-Resolution

Sample Code:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_runner.py --batch_size=2 --sw_batch_size=1 --mask_ratio=0.6 --epoch=300 --mask_patch_size=16 --img_size=96 --min_lr=1e-5 --warmpup_epoch=4 --loss_type=all_img --base_lr=1e-4 --warmup_lr=1e-6 --weight_decay=0.05 --cache_dataset --cache_rate=1 --model_type=swin_skip --save_freq=5 --print_freq=1 --log_dir="./logdir/swin_superres_mm" --output="./output/swin_superres_mm" --thread_loader --out_channels=1 --choice "superres" --interpolate 4 --mm_con 0.03 --temperature 0.5 --amp_opt_level="O0"
```

--interpolate can be use to change the downsampling. Default=4

--mm_con is used to change ratio of CFM loss. Default=0.03

--temperature is used to change temperature in CFM loss. Default=0.5

We don't use AMP in these experiments as it causes CFM loss to go NaN

3) Local Disturbance


Sample Code:

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_runner.py --batch_size=2 --sw_batch_size=1 --mask_ratio=0.6 --epoch=300 --mask_patch_size=16 --img_size=96 --min_lr=1e-5 --warmpup_epoch=4 --loss_type=all_img --base_lr=1e-4 --warmup_lr=1e-6 --weight_decay=0.05 --cache_dataset --cache_rate=1 --model_type=swin_skip --save_freq=5 --print_freq=1 --log_dir="./logdir/swin_LD_mm" --output="./output/swin_LD_mm" --thread_loader --out_channels=1 --choice "LD" --mm_con 0.03 --temperature 0.5 --amp_opt_level="O0"
```

--mm_con is used to change ratio of CFM loss. Default=0.03

--temperature is used to change temperature in CFM loss. Default=0.5

We don't use AMP in these experiments as it causes CFM loss to go NaN

4) ALL DISTURBANCE: COMBINE DENOISE+SUPERRES+LocalDIsturbance

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_runner.py --batch_size=2 --sw_batch_size=1 --mask_ratio=0.6 --epoch=1000 --mask_patch_size=16 --img_size=96 --min_lr=1e-5 --warmpup_epoch=4 --loss_type=all_img --base_lr=1e-4 --warmup_lr=1e-6 --weight_decay=0.05 --cache_dataset --cache_rate=1 --model_type=swin_skip --save_freq=5 --print_freq=1 --log_dir="./logdir/swin_COMB_mm" --output="./output/swin_COMB_mm" --thread_loader --out_channels=1 --choice "all" --variance 0.1 --mm_con 0.03 --temperature 0.5 --amp_opt_level="O0"
```

--mm_con is used to change ratio of CFM loss. Default=0.03

--temperature is used to change temperature in CFM loss. Default=0.5

## Datasets

Below is a list of datasets that have been used. We have included a list of json files (jsons directory) that contain a list of the training data from the below mentioned datasets:

- Head & Neck Squamous Cell Carcinoma (HNSCC) ([Link](https://wiki.cancerimagingarchive.net/display/Public/HNSCC))
- Lung Nodule Analysis 2016 (LUNA 16) ([Link](https://luna16.grand-challenge.org/Data/))
- TCIA CT Colonography Trial ([Link](https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY/))
- TCIA Covid 19 ([Link](https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19/))
- TCIA LIDC ([Link](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI/))
- Brats 2021 ([Link](http://braintumorsegmentation.org/))

## References

1.) Valanarasu, J.M.J., Tang, Y., Yang, D., Xu, Z., Zhao, C., Li, W., Patel, V.M., Landman, B., Xu, D., He, Y. and Nath, V., 2023. Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training. arXiv preprint arXiv:2307.16896.

## Model Weights

All json files and pre-trained model weights can be downloaded from the below
([DAE_SSL_Weights](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/dae_weights_midl_2024/DAE_SSL_WEIGHTS.zip))
([Feta Json](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/dae_weights_midl_2024/data_folds_feta_json.zip))
([BTCV Json](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/dae_weights_midl_2024/json_data_folds_btcv.zip))
([Pretraining Json](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/dae_weights_midl_2024/pretrain_jsons.zip))

Bibtex:
```commandline
@article{valanarasu2023disruptive,
  title={Disruptive Autoencoders: Leveraging Low-level features for 3D Medical Image Pre-training},
  author={Valanarasu, Jeya Maria Jose and Tang, Yucheng and Yang, Dong and Xu, Ziyue and Zhao, Can and Li, Wenqi and Patel, Vishal M and Landman, Bennett and Xu, Daguang and He, Yufan and others},
  journal={arXiv preprint arXiv:2307.16896},
  year={2023}
}

```
