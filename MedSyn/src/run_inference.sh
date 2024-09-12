#!/bin/bash

set -x

# extract features from text prompt
python extract_text_feature.py --prompt 'There is no airspace opacity, effusion or pneumothorax. There is no evidence of suspicious pulmonary nodule or mass.' \
                               --text_model_path './model/pretrained_lm' \
                               --save_path './result/text_feature/normal.npy'

# Inference for low-res
python eval_low_res_given_prompt.py --text_feature_folder './result/text_feature/' \
                                    --pretrain_model_path './model/results_text_low_res_improved_unet_seg/' \
                                    --save_path './result/image_low_res/'

# Inference for high-res
python eval_super_res.py            --low_res_folder './result/image_low_res/' \
                                    --pretrain_model_path './model/results_SR_gaussian_aug_full_volume_seg/' \
                                    --save_path './result/image_high_res/'