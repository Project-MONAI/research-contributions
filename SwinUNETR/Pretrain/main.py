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

import os
import argparse
import numpy as np
from time import time

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils.data_utils import get_loader
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.nn.parallel import DistributedDataParallel
from models.ssl_head import SSLHead
from losses.loss import Loss
from utils.ops import rot_rand, aug_rand


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args,
              global_step,
              train_loader,
              val_best,
              scaler
              ):

        model.train()
        loss_train = []
        loss_train_recon = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = (batch["image"].cuda())
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment
            with autocast(enabled=args.amp):
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                loss, losses_tasks = loss_function(rot_p,
                                                   rots,
                                                   contrastive1_p,
                                                   contrastive2_p,
                                                   imgs_recon,
                                                   imgs)
            loss_train.append(loss.item())
            loss_train_recon.append(losses_tasks[2].item())
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.distributed:
                if dist.get_rank() == 0:
                    print('Step:{}/{}, Loss:{:.4f}, Time:{:.4f}'.format(global_step, args.num_steps, loss, time()-t1))
            else:
                print('Step:{}/{}, Loss:{:.4f}, Time:{:.4f}'.format(global_step, args.num_steps, loss, time()-t1))

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_loss, val_loss_recon, img_list = validation(args, test_loader)
                writer.add_scalar("Validation/loss_recon",
                                  scalar_value=val_loss_recon,
                                  global_step=global_step)
                writer.add_scalar("train/loss_total",
                                  scalar_value=np.mean(loss_train),
                                  global_step=global_step)
                writer.add_scalar("train/loss_recon",
                                  scalar_value=np.mean(loss_train_recon),
                                  global_step=global_step)

                writer.add_image("Validation/x1_gt", img_list[0], global_step, dataformats='HW')
                writer.add_image("Validation/x1_aug", img_list[1], global_step, dataformats='HW')
                writer.add_image("Validation/x1_recon", img_list[2], global_step, dataformats='HW')

                if val_loss_recon < val_best:
                    val_best = val_loss_recon
                    checkpoint = {'global_step': global_step,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, logdir + '/model_bestValRMSE.pt')
                    print('Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}'.format(val_best,
                                                                                                   val_loss_recon))
                else:
                    print('Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}'.format(val_best,
                                                                                                      val_loss_recon))
        return global_step, loss, val_best

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_recon = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = (batch["image"].cuda())
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(enabled=args.amp):
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                    rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)
                    loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
                loss_recon = losses_tasks[2]
                loss_val.append(loss.item())
                loss_val_recon.append(loss_recon.item())
                x_gt = x1.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt[0][0][:, :, 48] * 255.0
                xgt = xgt.astype(np.uint8)
                x1_augment = x1_augment.detach().cpu().numpy()
                x1_augment = (x1_augment - np.min(x1_augment)) / (np.max(x1_augment) - np.min(x1_augment))
                x_aug = x1_augment[0][0][:, :, 48] * 255.0
                x_aug = x_aug.astype(np.uint8)
                rec_x1 = rec_x1.detach().cpu().numpy()
                rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
                recon = rec_x1[0][0][:, :, 48] * 255.0
                recon = recon.astype(np.uint8)
                img_list = [xgt, x_aug, recon]
                print('Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}'.format(step, loss, loss_recon))

        return np.mean(loss_val), np.mean(loss_val_recon), img_list

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--num_steps', default=100000, type=int, help='number of training iterations')
    parser.add_argument('--eval_num', default=100, type=int, help='evaluation frequency')
    parser.add_argument('--warmup_steps', default=500, type=int, help='warmup steps')
    parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--feature_size', default=48, type=int, help='embedding size')
    parser.add_argument('--dropout_path_rate', default=0.0, type=float, help='drop path rate')
    parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
    parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
    parser.add_argument('--a_min', default=-1000, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=1000, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--batch_size', default=2, type=int, help='number of batch size')
    parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.1, type=float, help='decay rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lrdecay', action='store_true', help='enable learning rate decay')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='maximum gradient norm')
    parser.add_argument('--loss_type', default='SSL', type=str)
    parser.add_argument('--opt', default='adamw', type=str, help='optimization algorithm')
    parser.add_argument('--lr_schedule', default='warmup_cosine', type=str)
    parser.add_argument('--resume', default=None, type=str, help='resume training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--grad_clip', action='store_true', help='gradient clip')
    parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--smartcache_dataset', action='store_true',
                        help='use monai smartcache Dataset')
    parser.add_argument('--cache_dataset', action='store_true',
                        help='use monai cache Dataset')

    args = parser.parse_args()
    logdir = './runs/' + args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    model = SSLHead(args)
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(),
                               lr=args.lr,
                               weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.decay)

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict['state_dict'])
        model.epoch = model_dict['epoch']
        model.optimizer = model_dict['optimizer']

    if args.lrdecay:
        if args.lr_schedule == 'warmup_cosine':
            scheduler = WarmupCosineSchedule(optimizer,
                                             warmup_steps=args.warmup_steps,
                                             t_total=args.num_steps)

        elif args.lr_schedule == 'poly':
            lambdas = lambda epoch: (1 - float(epoch) / float(args.epochs)) ** 0.9
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=lambdas)

    loss_function = Loss(args.batch_size * args.sw_batch_size,
                         args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    train_loader, test_loader = get_loader(args)

    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args,
                                            global_step,
                                            train_loader,
                                            best_val,
                                            scaler
                                            )
    checkpoint = {'epoch': args.epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir+"final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")
    save_ckp(checkpoint, logdir+'/model_final_epoch.pt')


if __name__ == '__main__':
    main()
