import argparse
import datetime
import os
import pdb
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler

# from config import get_config
from models import build_model
from optimizer import build_optimizer
from timm.utils import AverageMeter
from utils import TensorboardLogger, auto_resume_helper, get_grad_norm, load_checkpoint, reduce_tensor, save_checkpoint

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser("SimMIM pre-training script", add_help=False)

    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--num_classes", default=0, type=int, help="number of input channels")
    parser.add_argument("--window_size", default=(7, 7, 7), type=tuple, help="window size")
    parser.add_argument("--patch_size", default=(2, 2, 2), type=tuple, help="window size")
    parser.add_argument("--mask_patch_size", default=16, type=int, help="window size")
    parser.add_argument("--img_size", default=96, type=int, help="image size")
    parser.add_argument("--num_heads", default=[3, 6, 12, 24], type=list, help="number of heads")
    parser.add_argument("--depths", default=[2, 2, 2, 2], type=list, help="number of depths")
    parser.add_argument("--embed_dim", default=48, type=int, help="embedding dimention")
    parser.add_argument("--mlp_ratio", default=4.0, type=float, help="MLP ratio")
    parser.add_argument("--drop_rate", default=0.0, type=float, help="drop rate")
    parser.add_argument("--attn_drop_rate", default=0.0, type=float, help="attention drop rate")
    parser.add_argument("--drop_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--layer_decay", default=1.0, type=float, help="layer decay")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--mask_ratio", default=0.6, type=float, help="drop path rate")

    parser.add_argument("--optimizer_name", type=str, default="adamw", help="optimizer name")
    parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
    parser.add_argument("--base_lr", default=5e-4, type=float, help="base learning rate")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="weight decay")
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple, help="optimizer betas")
    parser.add_argument("--eps", default=1e-8, type=float, help="eps")
    parser.add_argument("--decoder", type=str, default="upsample", help="decoder type")
    parser.add_argument("--loss_type", type=str, default="mask_only", help="decoder type")

    parser.add_argument("--amp_opt_level", type=str, default="O1", help="amp opt level")
    parser.add_argument("--epoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="number of epochs")
    parser.add_argument("--warmpup_epoch", default=20, type=int, help="warmup epoch")
    parser.add_argument("--decay_epoch", default=30, type=int, help="warmup epoch")
    parser.add_argument("--save_freq", default=1, type=int, help="saving frequency")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--accumulate_step", default=0, type=int, help="accumulation step")
    parser.add_argument("--clip_grad", default=1, type=int, help="saving frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument("--lr_scheduler_name", type=str, default="cosine", help="learning rate scheduler name")
    parser.add_argument("--min_lr", default=5e-6, type=float, help="min learning rate")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warmup lr")
    parser.add_argument("--lr_decay_rate", default=0.1, type=float, help="lr decay rate")
    parser.add_argument("--lr_gamma", default=0.1, type=float, help="lr gamma")
    parser.add_argument("--auto_resume", default=True, type=bool)
    parser.add_argument("--iso_spacing", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--model_type", type=str, default="swin", help="model type")
    parser.add_argument("--cache_dataset", default=True, action="store_true")
    parser.add_argument("--thread_loader", default=True, action="store_true")
    parser.add_argument("--onlycovid", default=False, action="store_true")

    parser.add_argument("--cache_rate", default=0.5, type=float, help="drop path rate")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for single GPU")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")

    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--deterministic", help="set seed for deterministic training", action="store_true")
    parser.add_argument(
        "--use_grad_checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
    )

    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--decoder_off", action="store_true")
    parser.add_argument("--encoder_off", action="store_true")
    parser.add_argument(
        "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
    )

    # parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help="local rank for DistributedDataParallel")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    # parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    # parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--choice", default="mae", type=str, help="choice")
    parser.add_argument("--inf", default="notsim", type=str, help="choice")

    parser.add_argument("--variance", default=0.1, type=float, help="")
    parser.add_argument("--interpolate", default=4, type=float, help="")
    parser.add_argument("--temperature", default=0.07, type=float, help="drop path rate")
    parser.add_argument("--mm_con", default=0.02, type=float, help="drop path rate")

    args = parser.parse_args()

    return args


def main(args):
    data_loader_train, data_loader_val = build_loader(args, is_pretrain=True)
    model = build_model(args, is_pretrain=True)
    model.cuda()
    logger.info(str(model))
    pretrained_dir = args.pretrained_dir
    optimizer = build_optimizer(args, model, logger, is_pretrain=True)
    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True
    )
    # model = torch.nn.parallel.DataParallel(model)

    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    logger.info("Start training")
    if args.resume:
        model_dict = torch.load(os.path.join(args.pretrained_dir, "ckpt_best.pth"))["model"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")
    start_time = time.time()
    val_loss_best = 1e9

    if args.encoder_off:
        for param in model.module.encoder.parameters():
            param.requires_grad = False
        for param in model.module.encoder2.parameters():
            param.requires_grad = False
        for param in model.module.encoder3.parameters():
            param.requires_grad = False
        for param in model.module.encoder4.parameters():
            param.requires_grad = False
        for param in model.module.encoder10.parameters():
            param.requires_grad = False
    if args.decoder_off:
        for param in model.module.decoder1.parameters():
            param.requires_grad = False
        for param in model.module.decoder2.parameters():
            param.requires_grad = False
        for param in model.module.decoder3.parameters():
            param.requires_grad = False
        for param in model.module.decoder4.parameters():
            param.requires_grad = False
        for param in model.module.decoder5.parameters():
            param.requires_grad = False
    # pdb.set_trace()

    # print("LENGTH ", len(data_loader_train))
    for epoch in range(args.start_epoch, args.epoch):
        if not args.thread_loader:
            data_loader_train.sampler.set_epoch(epoch)
        train_loss_avg, train_L1_avg, train_MM_avg = train_one_epoch(
            args, model, data_loader_train, optimizer, epoch, lr_scheduler
        )
        val_loss_avg, val_L1_avg, val_MM_avg, x_orig, x_recon, x_masked = validate(data_loader_val, model)
        x_orig = torch.cat(x_orig, dim=0)[18:24]
        x_recon = torch.cat(x_recon, dim=0)[18:24]
        x_masked = torch.cat(x_masked, dim=0)[18:24]
        if dist.get_rank() == 0:
            x_orig = x_orig[:, :, 48, :, :]
            x_recon = x_recon[:, :, 48, :, :]
            x_masked_in = x_masked[:, :, 48, :, :]

            grid_imgs = torchvision.utils.make_grid(x_orig)
            grid_recons = torchvision.utils.make_grid(x_recon)
            if args.choice == "mae":
                x_masked = x_orig * x_masked_in.float()
                grid_masks = torchvision.utils.make_grid(x_masked)
            if args.choice == "denoise":
                grid_masks = torchvision.utils.make_grid(x_masked_in)
            if args.choice == "superres":
                grid_masks = torchvision.utils.make_grid(x_masked_in)
            if args.choice == "sld":
                grid_masks = torchvision.utils.make_grid(x_masked_in)
            if args.choice == "sld_noise":
                grid_masks = torchvision.utils.make_grid(x_masked_in)
            log_writer.update_img(orig_imgs=grid_imgs, head="images", step=epoch)
            log_writer.update_img(masked_imgs=grid_masks, head="images", step=epoch)
            log_writer.update_img(recon_imgs=grid_recons, head="images", step=epoch)
            log_writer.update(loss_val_avg=val_loss_avg, head="perf", step=epoch)
            log_writer.update(loss_val_L1=val_L1_avg, head="perf", step=epoch)
            log_writer.update(loss_val_MM=val_MM_avg, head="perf", step=epoch)
            log_writer.update(loss_train_avg=train_loss_avg, head="perf", step=epoch)
            log_writer.update(loss_train_L1=train_L1_avg, head="perf", step=epoch)
            log_writer.update(loss_train_MM=train_MM_avg, head="perf", step=epoch)
            if val_loss_avg <= val_loss_best:
                save_checkpoint(args, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger, best_model=True)

        if dist.get_rank() == 0 and (epoch == (args.epoch - 1)):
            save_checkpoint(args, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger, best_model=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(args, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()
    loss_meter = AverageMeter()

    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (img, mask) in enumerate(data_loader):
        # img = img[0]['image'].cuda(non_blocking=True)
        # print(img['class'])
        cl_type = img["class"]
        img = img["image"].cuda(non_blocking=True)
        B, C, H, W, Z = img.shape
        noise = (args.variance**0.5) * torch.randn(B, C, H, W, Z).cuda()
        img_noisy = img + noise
        mask = mask.cuda(non_blocking=True)
        if args.choice == "mae":
            loss, _, _ = model(img, mask, img, cl_type)
        elif args.choice == "denoise":
            loss1, loss2, _, _ = model(img_noisy, mask, img, cl_type)
        elif args.choice == "superres":
            img_lowres = F.interpolate(
                img, size=(int(96 / args.interpolate), int(96 / args.interpolate), int(96 / args.interpolate))
            )
            img_resam = F.interpolate(img_lowres, size=(96, 96, 96))
            loss1, loss2, _, _ = model(img_resam, mask, img, cl_type)
        elif args.choice == "sld":
            loss1, loss2, _, _ = model(img, mask, img, cl_type)
        elif args.choice == "sld_noise":
            loss1, loss2, _, _ = model(img, mask, img, cl_type)
        elif args.choice == "all":
            img_lowres = F.interpolate(
                img_noisy, size=(int(96 / args.interpolate), int(96 / args.interpolate), int(96 / args.interpolate))
            )
            img_resam = F.interpolate(img_lowres, size=(96, 96, 96))
            loss1, loss2, _, _ = model(img_resam, mask, img, cl_type)
        mask = mask.cuda(non_blocking=True)

        loss2 = args.mm_con * loss2
        loss = loss1 + loss2
        # loss, _, _ = model(img, mask)
        if args.accumulate_step > 1:
            loss = loss / args.accumulate_step
            if args.amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if args.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % args.accumulate_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.zero_grad()
            if args.amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_grad)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if args.clip_grad:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_l1_meter.update(loss1.item(), img.size(0))
        loss_cont_meter.update(loss2.item(), img.size(0))

        loss_meter.update(loss.item(), img.size(0))

        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{args.epoch}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t"
                f"loss_MM {loss_cont_meter.val:.4f} ({loss_cont_meter.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg, loss_l1_meter.avg, loss_cont_meter.avg


@torch.no_grad()
def validate(data_loader, model):
    model.eval()
    loss_meter = AverageMeter()
    loss_l1_meter = AverageMeter()
    loss_cont_meter = AverageMeter()

    img_list = []
    img_recon_list = []
    mask_out_list = []

    for idx, (img, mask) in enumerate(data_loader):
        cl_type = img["class"]
        img = img["image"].cuda(non_blocking=True)
        B, C, H, W, Z = img.shape
        noise = (args.variance**0.5) * torch.randn(B, C, H, W, Z).cuda()
        img_noisy = img + noise
        mask = mask.cuda(non_blocking=True)
        if args.choice == "mae":
            loss, img_recon, mask_out = model(img, mask, img, cl_type)
            mask_out_list.append(mask_out)
        elif args.choice == "denoise":
            loss1, loss2, img_recon, mask_out = model(img_noisy, mask, img, cl_type)
            mask_out_list.append(img_noisy)
        elif args.choice == "superres":
            img_lowres = F.interpolate(
                img, size=(int(96 / args.interpolate), int(96 / args.interpolate), int(96 / args.interpolate))
            )
            img_resam = F.interpolate(img_lowres, size=(96, 96, 96))
            loss1, loss2, img_recon, mask_out = model(img_resam, mask, img, cl_type)
            mask_out_list.append(img_resam)
        elif args.choice == "sld":
            loss1, loss2, img_recon, mask_out = model(img, mask, img, cl_type)
            mask_out_list.append(img)
        elif args.choice == "sld_noise":
            loss1, loss2, img_recon, mask_out = model(img, mask, img, cl_type)
            mask_out_list.append(img)
        elif args.choice == "all":
            img_lowres = F.interpolate(
                img_noisy, size=(int(96 / args.interpolate), int(96 / args.interpolate), int(96 / args.interpolate))
            )
            img_resam = F.interpolate(img_lowres, size=(96, 96, 96))
            loss1, loss2, img_recon, mask_out = model(img_resam, mask, img, cl_type)
            mask_out_list.append(img)
        mask = mask.cuda(non_blocking=True)
        # loss, img_recon, mask_out = model(img, mask)
        loss2 = args.mm_con * loss2

        loss = loss1 + loss2
        loss = reduce_tensor(loss)
        loss_l1_meter.update(loss1.item(), img.size(0))
        loss_cont_meter.update(loss2.item(), img.size(0))
        loss_meter.update(loss.item(), img.size(0))

        logger.info(
            f"Test: [{idx}/{len(data_loader)}]\t"
            f"Loss_L1 {loss_l1_meter.val:.4f} ({loss_l1_meter.avg:.4f})\t"
            f"Loss_MM {loss_cont_meter.val:.4f} ({loss_cont_meter.avg:.4f})\t"
            f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        )
        img_list.append(img)
        # mask_out_list.append(mask_out)
        img_recon_list.append(img_recon)
    logger.info(f" * Val Loss {loss_meter.avg:.3f}")

    return loss_meter.avg, loss_l1_meter.avg, loss_cont_meter.avg, img_list, img_recon_list, mask_out_list


if __name__ == "__main__":
    args = parse_option()

    if args.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = args.seed + dist.get_rank()

    if args.deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)

    cudnn.benchmark = True
    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = args.base_lr * args.batch_size * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = args.warmup_lr * args.batch_size * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = args.min_lr * args.batch_size * dist.get_world_size() / 512.0
    # # gradient accumulation also need to scale the learning rate
    # if args.accumulate_step > 1:
    #     linear_scaled_lr = linear_scaled_lr * args.accumulate_step
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * args.accumulate_step
    #     linear_scaled_min_lr = linear_scaled_min_lr * args.accumulate_step

    if dist.get_rank() == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)

    if dist.get_rank() == 0:
        log_writer = TensorboardLogger(log_dir=args.log_dir)

    logger = create_logger(output_dir=args.output, name=f"{args.model_type}")

    main(args)
