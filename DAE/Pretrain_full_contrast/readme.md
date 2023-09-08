### Using the code:

The pretraining Code is tested with Swin UNETR as backbone. It trains on CT+MRI data with the new Cross Modal Contrastive loss. The current code base has 3 pretraining options:

1) Denoising
2) Super-Resolution
3) Local Disturbance

Change the argument --choice to switch between different pretraining setups

- Install Dependencies:

```bash
pip install -r requirements.txt
```

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