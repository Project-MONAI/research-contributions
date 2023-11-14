import argparse
import logging
import os

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from inferers import double_sliding_window_inference
from models import SwinUNETR
from timm.utils import setup_default_logging
from utils.data_utils import get_loader
from utils.misc import dice, distributed_all_gather, resample_3d

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
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument(
    "--nouse_multi_epochs_loader",
    action="store_true",
    help="not use the multi-epochs-loader to save time at the beginning of every epoch",
)
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--cross_attention_in_origin_view", action="store_true", help="Whether compute cross attention in original view"
)

spacing = Spacing(pixdim=(1, 1, 1), mode="nearest")
hd_per = 95
view = ["Cor1", "Sag2", "Sag1", "Axi2", "Axi1", "Cor2", "Fuse"]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    args = parser.parse_args()
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    output_directory = "./outputs/" + args.exp_name
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    # np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = True
    val_loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        setup_default_logging()
        # logging.info(f"Batch size is: {args.batch_size}, epochs: {args.max_epochs}")

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
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
        cross_attention_in_origin_view=args.cross_attention_in_origin_view,
    )
    model.load_state_dict(torch.load(pretrained_pth, map_location="cpu")["state_dict"])
    model.cuda(args.gpu)
    model.eval()
    model_without_ddp = model
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model_without_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, broadcast_buffers=False, find_unused_parameters=True
        )

    dice_all = []  # np.zeros((len(val_loader), len(view), args.out_channels - 1))
    hd_all = []  # np.zeros((len(val_loader), len(view), args.out_channels - 1))
    asd_all = []  # np.zeros((len(val_loader), len(view), args.out_channels - 1))

    with torch.no_grad():
        for id, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(args.gpu), batch["label"].cpu())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            val_labels = val_labels.numpy()[0, :, :, :, :]
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            torch.cuda.empty_cache()
            output_list = []
            val_fuse = 0

            val_labels = spacing(val_labels, original_affine)[0]
            val_labels = np.expand_dims(val_labels, axis=0)
            val_labels = one_hot(torch.from_numpy(val_labels), num_classes=args.out_channels, dim=1)

            for i in range(3):
                # j = i + 3
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
            print("Inference finished on case {}".format(img_name))

            dice = np.zeros((len(view), args.out_channels - 1))
            hd = np.zeros((len(view), args.out_channels - 1))
            asd = np.zeros((len(view), args.out_channels - 1))
            for i, output in enumerate(output_list):
                output = np.argmax(output, axis=0, keepdims=False)
                output = resample_3d(output, target_shape)
                target_ornt = nib.orientations.axcodes2ornt(tuple(nib.aff2axcodes(original_affine)))
                out_ornt = [[0, 1], [1, 1], [2, 1]]
                ornt_transf = nib.orientations.ornt_transform(out_ornt, target_ornt)
                output = nib.orientations.apply_orientation(output, ornt_transf)
                nib.save(
                    nib.Nifti1Image(output.astype(np.uint8), affine=original_affine),
                    os.path.join(output_directory, view[i] + "_" + img_name),
                )
                output = np.expand_dims(spacing(np.expand_dims(output, axis=(0)), original_affine)[0], axis=0)
                output = one_hot(torch.from_numpy(output), num_classes=args.out_channels, dim=1)
                # print(output.shape, val_labels.shape)
                dice_ = compute_meandice(output, val_labels, include_background=False).numpy()[0]
                hd_ = compute_hausdorff_distance(output, val_labels, percentile=hd_per).numpy()[0]
                asd_ = compute_average_surface_distance(output, val_labels).numpy()[0]
                print("{} {} View Mean Dice: {}".format(img_name, view[i], np.mean(dice_)))
                print("{} {} View Mean HD: {}".format(img_name, view[i], np.mean(hd_)))
                print("{} {} View Mean ASD: {}".format(img_name, view[i], np.mean(asd_)))
                dice[i, :] = dice_
                hd[i, :] = hd_
                asd[i, :] = asd_
            dice_all.append(dice)
            hd_all.append(hd)
            asd_all.append(asd)

    dice_all = torch.tensor(np.stack(dice_all, axis=0)).cuda(args.gpu)
    hd_all = torch.tensor(np.stack(hd_all, axis=0)).cuda(args.gpu)
    asd_all = torch.tensor(np.stack(asd_all, axis=0)).cuda(args.gpu)
    dice_list = distributed_all_gather([dice_all], out_numpy=False, is_valid=True)
    hd_list = distributed_all_gather([hd_all], out_numpy=False, is_valid=True)
    asd_list = distributed_all_gather([asd_all], out_numpy=False, is_valid=True)
    dice_list = torch.flatten(torch.stack(dice_list[0], axis=0), start_dim=0, end_dim=1).cpu().numpy()
    hd_list = torch.flatten(torch.stack(hd_list[0], axis=0), start_dim=0, end_dim=1).cpu().numpy()
    asd_list = torch.flatten(torch.stack(asd_list[0], axis=0), start_dim=0, end_dim=1).cpu().numpy()

    if args.rank == 0:
        for i in range(len(view)):
            print(dice_list.shape)
            print("Overall {} View Mean Dice: {}".format(view[i], np.mean(dice_list[:, i, :], axis=0)))
            print("Overall {} View Mean HD: {}".format(view[i], np.mean(hd_list[:, i, :], axis=0)))
            print("Overall {} View Mean ASD: {}".format(view[i], np.mean(asd_list[:, i, :], axis=0)))
            np.savetxt(
                os.path.join(output_directory, view[i] + "Dice.txt"),
                np.mean(dice_list[:, i, :], axis=0),
                delimiter="\t",
            )
            np.savetxt(
                os.path.join(output_directory, view[i] + "HD.txt"), np.mean(hd_list[:, i, :], axis=0), delimiter="\t"
            )
            np.savetxt(
                os.path.join(output_directory, view[i] + "ASD.txt"), np.mean(asd_list[:, i, :], axis=0), delimiter="\t"
            )


if __name__ == "__main__":
    main()
