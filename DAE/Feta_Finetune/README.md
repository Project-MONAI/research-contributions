# Finetune on Feta Dataset

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Main Finetuning Exps

The main fineuning exps are based on Swin UNETR architecture. 

Sample Code:

``` bash
python main_runner.py --json_list=./data_folds/d10/data_0.json --data_dir=/path/to/data --feature_size=48 --pos_embed='perceptron' --roi_x=96 --roi_y=96 --roi_z=96 --use_checkpoint --batch_size=4 --max_epochs=600 --save_checkpoint --model_name swin --logdir ./runs/swin_finetune --optim_lr 8e-4 --use_ssl_pretrained --load_dir "../../Pretrain//output/FOLDER_NAME/" --finetune_choice "both"
```

--load_dir to specify which folder to load pretrained models from

--finetune_choice to specify which part of network the pretrained model needs to be loaded for. Default is both. If specified "encoder", only encoder weights will be copied from the pretrained model. If specified "decoder", only decoder weights will be copied from the pretrained model.
