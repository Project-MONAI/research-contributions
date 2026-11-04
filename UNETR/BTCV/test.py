# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from monai.transforms import Compose, Invertd, SaveImaged, AsDiscreted
from monai.inferers import sliding_window_inference
from monai.data.meta_tensor import MetaTensor

import numpy as np
import torch

from networks.unetr import UNETR
from trainer import dice
from dataset.customDataset import getDatasetLoader, getPredictLoader

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")

parser.add_argument(
    "--mode",choices=['predict', 'validation'], default="validation", type=str, help="mode for predict or validation"
)

parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--pretrained_model_name", default="model_final.pt", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=64, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="learnable", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

def inference(inputs,model, args):
    out = sliding_window_inference(inputs, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size, model, overlap=args.infer_overlap)
    prob = torch.softmax(out, 1).cpu().numpy()
    predict_label = np.argmax(prob, axis=1).astype(np.uint8)
    return predict_label

def main():
    args = parser.parse_args()
    args.test_mode = True
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        model_dict = torch.load(pretrained_pth, weights_only=False)
        model.load_state_dict(model_dict['state_dict'])
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        if args.mode == 'validation':
            loader = getDatasetLoader(args)[1]
            for batch, label in loader:
                val_inputs, val_labels = (batch.cuda(), label.cuda())
                val_outputs = inference(val_inputs, model, args)
                val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
                dice_list_sub = []
                for i in range(1, args.out_channels):
                    every_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                    dice_list_sub.append(every_Dice)
                mean_dice = np.mean(dice_list_sub)
                print("Mean Dice: {}".format(mean_dice))
                dice_list_case.append(mean_dice)
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        elif args.mode == 'predict': # ref: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py
            loader, preTransform = getPredictLoader(args)
            postTransforms = Compose([
                Invertd(
                    keys="pred",
                    transform=preTransform,
                    orig_keys="image",        # invert from history of image tag
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", threshold=0.5),
                SaveImaged(
                    keys="pred",
                    output_dir="./output",
                    output_postfix="seg",
                    resample=False,
                    output_dtype=np.uint8,
                    print_log=True,
                )
            ])

            for d in loader:

                # shape: (b, c, h, w, d)
                input_data = (d["image"] if torch.is_tensor(d["image"]) else torch.as_tensor(d["image"])).to(device)

                predict_raw = inference(input_data, model, args) # shape: (B, H, W, D)
                predict_tensor = torch.from_numpy(predict_raw.astype(np.float32)) # shape: (B, H, W, D)

                meta = getattr(d["image"], "meta", None)
                d["pred"] = MetaTensor(predict_tensor, meta=meta) if meta is not None else predict_tensor

                postTransforms(d)

if __name__ == "__main__":
    main()
