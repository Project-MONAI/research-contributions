# Note for SwinMM Finetuning

## Training

### FIXME(outdated)

```bash
python main.py
--feature_size=48
--batch_size=1
--logdir="swin_mm_test/"
--roi_x=64
--roi_y=64
--roi_z=64
--optim_lr=1e-4
--lrschedule="warmup_cosine"
--infer_overlap=0.5
--save_checkpoint
--data_dir="/dataset/dataset0/"
--distributed
--use_ssl_pretrained
--pretrained_dir="./pretrained_models/"
--pretrained_model_name="model_bestValRMSE.pt"
```

## Testing

### FIXME(outdated)

```bash
python test.py
--feature_size=48
--batch_size=1
--exp_name="swin_mm_test/"
--roi_x=64
--roi_y=64
--roi_z=64
--infer_overlap=0.5
--data_dir="/dataset/dataset0/"
--pretrained_dir="./runs/multiview_101021/"
--pretrained_model_name="model.pt"
```
