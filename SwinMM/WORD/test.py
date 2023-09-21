# Copyright 2020 - 2022 MONAI Consortium
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

import nibabel as nib
import numpy as np
import torch
from inferers import double_sliding_window_inference
from models import SwinUNETR
from utils.data_utils import get_loader
from utils.misc import resample_3d

from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance, compute_meandice
from monai.networks.utils import one_hot
from monai.transforms import Spacing

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/multiview_101616/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/dataset12_WORD/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="multiview_101616/", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset12_WORD.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.7, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=17, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
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
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

spacing = Spacing(pixdim=(1, 1, 1), mode="nearest")
hd_per = 95
view = ["Cor1", "Sag2", "Sag1", "Axi2", "Axi1", "Cor2", "Fuse"]


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        fusion_depths=(1, 1, 1, 1, 1, 1),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    dice_out = np.zeros((len(val_loader), len(view), args.out_channels - 1))
    hd_out = np.zeros((len(val_loader), len(view), args.out_channels - 1))
    asd_out = np.zeros((len(val_loader), len(view), args.out_channels - 1))

    with torch.no_grad():
        for id, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            val_labels = val_labels.cpu().numpy()[0, :, :, :, :]
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            torch.cuda.empty_cache()
            output_list = []
            val_fuse = 0

            val_labels = spacing(val_labels, original_affine)[0]
            val_labels = np.expand_dims(val_labels, axis=0)
            val_labels = one_hot(torch.from_numpy(val_labels), num_classes=args.out_channels, dim=1)

            for i in range(3):
                val_outputs_1, val_outputs_2 = double_sliding_window_inference(
                    val_inputs,
                    i,
                    (args.roi_x, args.roi_y, args.roi_z),
                    16,
                    model,
                    overlap=args.infer_overlap,
                    mode="gaussian",
                )

                val_outputs_1 = torch.softmax(val_outputs_1, 1).cpu().numpy()[0]
                val_outputs_2 = torch.softmax(val_outputs_2, 1).cpu().numpy()[0]
                val_fuse = val_fuse + val_outputs_1 + val_outputs_2
                output_list.append(val_outputs_1)
                output_list.append(val_outputs_2)
            output_list.append(val_fuse)

            for i, output in enumerate(output_list):
                output = np.argmax(output, axis=0, keepdims=False)
                output = resample_3d(output, target_shape)
                target_ornt = nib.orientations.axcodes2ornt(tuple(nib.aff2axcodes(original_affine)))
                out_ornt = [[0, 1], [1, 1], [2, 1]]
                ornt_transf = nib.orientations.ornt_transform(out_ornt, target_ornt)
                output = nib.orientations.apply_orientation(output, ornt_transf)
                nib.save(
                    nib.Nifti1Image(output[::-1, ::-1, :].astype(np.uint8), affine=original_affine),
                    os.path.join(output_directory, view[i] + "_" + img_name),
                )
                output = np.expand_dims(spacing(np.expand_dims(output, axis=(0)), original_affine)[0], axis=0)
                output = one_hot(torch.from_numpy(output), num_classes=args.out_channels, dim=1)
                print(output.shape, val_labels.shape)
                dice_ = compute_meandice(output, val_labels, include_background=False).numpy()[0]
                hd_ = compute_hausdorff_distance(output, val_labels, percentile=hd_per).numpy()[0]
                asd_ = compute_average_surface_distance(output, val_labels).numpy()[0]
                print("{} View Mean Dice: {}".format(view[i], np.mean(dice_)))
                print("{} View Mean HD: {}".format(view[i], np.mean(hd_)))
                print("{} View Mean ASD: {}".format(view[i], np.mean(asd_)))
                dice_out[id, i, :] = dice_
                hd_out[id, i, :] = hd_
                asd_out[id, i, :] = asd_

        for i in range(len(view)):
            print("Overall {} View Mean Dice: {}".format(view[i], np.mean(dice_out[:, i, :], axis=0)))
            print("Overall {} View Mean HD: {}".format(view[i], np.mean(hd_out[:, i, :], axis=0)))
            print("Overall {} View Mean ASD: {}".format(view[i], np.mean(asd_out[:, i, :], axis=0)))


if __name__ == "__main__":
    main()
