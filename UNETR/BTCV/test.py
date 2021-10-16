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

import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Compose,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandSpatialCropSamplesd,
    ToTensord
)
from networks.unetr import UNETR
from trainer import dice
from monai.data import (
    DataLoader,
    load_decathlon_datalist,
    Dataset,
)
import argparse

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str)
parser.add_argument('--data_dir', default='/dataset/dataset0/', type=str)
parser.add_argument('--pretrained_model_name', default='UNETR_torchscript.pt', type=str)
parser.add_argument('--saved_checkpoint', default='torchscript', type=str, help='Supports torchscript or statedict')
parser.add_argument('--mlp_dim', default=3072, type=int)
parser.add_argument('--hidden_size', default=768, type=int)
parser.add_argument('--feature_size', default=16, type=int)
parser.add_argument('--infer_overlap', default=0.5, type=float)
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--out_channels', default=14, type=int)
parser.add_argument('--num_heads', default=12, type=int)
parser.add_argument('--res_block', action='store_true')
parser.add_argument('--conv_block', action='store_true')
parser.add_argument('--roi_x', default=96, type=int)
parser.add_argument('--roi_y', default=96, type=int)
parser.add_argument('--roi_z', default=96, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--pos_embedd', default='perceptron', type=str)
parser.add_argument('--norm_name', default='instance', type=str)

def main():
    args = parser.parse_args()
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(os.path.join(pretrained_dir, model_name)).to(device)
    elif args.saved_checkpoint == 'statedict':
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
            res_block=True,
            dropout_rate=args.dropout_rate).to(device)
    model.eval()
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"],
                     pixdim=(1.5, 1.5, 2.0),
                     mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"],
                         axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=-175,
                                 a_max=250,
                                 b_min=0.0,
                                 b_max=1.0,
                                 clip=True),
            CropForegroundd(keys=["image", "label"],
                            source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    data_dir = args.data_dir
    split_JSON = 'dataset_0.json'
    datasets = data_dir + split_JSON
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]
            dice_list_sub = []
            for i in range(1, 14):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':
    main()