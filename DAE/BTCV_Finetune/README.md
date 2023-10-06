# Finetune on BTCV

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Main Finetuning Exps

The main fineuning exps are based on Swin UNETR architecture.

All json files and pre-trained model weights can be downloaded from this ([Link](https://drive.google.com/drive/folders/1_EFLbOWn8E6fVUnFVfT4BNNFMlz7RCpM?usp=sharing))

Sample Code:

To train models from scratch without any pre-trained weights

``` bash

python main_for_ngc.py --json_list=/json_files/can_be/found_in/data_folds/xxxx.json --data_dir=/data_root --feature_size=48 --pos_embed='perceptron' --roi_x=96 --roi_y=96 --roi_z=96 --use_checkpoint --batch_size=4 --max_epochs=1000 --save_checkpoint --model_name swin --logdir ./provide_a_path/for_tensorboard_logs --optim_lr 8e-4 --val_every 5 --set_determ True --seed 120

```

Train models with pre-trained weights

```bash

python main_for_ngc.py --json_list=/json_files/can_be/found_in/data_folds/xxxx.json --data_dir=/data_root --feature_size=48 --pos_embed='perceptron' --roi_x=96 --roi_y=96 --roi_z=96 --use_checkpoint --batch_size=4 --max_epochs=1000 --save_checkpoint --model_name swin --logdir ./provide_a_path/for_tensorboard_logs --optim_lr 8e-4 --val_every 5 --use_ssl_pretrained --finetune_choice both --load_dir /path/to/ssl_pretrained_checkpoint --set_determ True --seed 120

```

--load_dir to specify which folder to load pretrained models from

--finetune_choice to specify which part of network the pretrained model needs to be loaded for. Default is both. If specified "encoder", only encoder weights will be copied from the pretrained model. If specified "decoder", only decoder weights will be copied from the pretrained model.
