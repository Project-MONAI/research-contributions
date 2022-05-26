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
from apex import amp
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils.data_utils import get_loader
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
from models.ssl_head import SSLHead
from losses.loss import Loss
from utils.ops import rot_rand, aug_rand
import pdb


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args,
              global_step,
              train_loader,
              val_best,
              torchScaler):

        model.train()
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        loss_train = []
        loss_train_recon = []
        for step, batch in enumerate(epoch_iterator):
            x = (batch["image"].to(device))
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment

            if args.torch_amp:
                with autocast(enabled=args.torch_amp):
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                    rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
            else:
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
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            elif args.torch_amp:
                torchScaler.scale(loss).backward()
                torchScaler.step(optimizer)
                torchScaler.update()           
            else:
                loss.backward()
            if args.amp:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))
            global_step += 1
            try:
                local_rank = dist.get_rank()
            except:
                local_rank = -1
            if local_rank == 0 or local_rank == -1:
                if global_step % args.eval_num == 0:
                    epoch_iterator_val = tqdm(test_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                    val_loss, val_loss_recon = validation(args, epoch_iterator_val)
                    writer.add_scalar("Validation/loss_recon",
                                      scalar_value=val_loss_recon,
                                      global_step=global_step)
                    writer.add_scalar("train/loss_total",
                                      scalar_value=np.mean(loss_train),
                                      global_step=global_step)
                    writer.add_scalar("train/loss_recon",
                                      scalar_value=np.mean(loss_train_recon),
                                      global_step=global_step)
                    if val_loss_recon < val_best:
                        val_best = val_loss_recon
                        checkpoint = {'global_step': global_step,
                                      'state_dict': model.state_dict(),
                                      'optimizer': optimizer.state_dict()}
                        save_ckp(checkpoint, logdir + '/model_bestValRMSE.pt')
                        print('Model was saved ! Best Recon. Val Loss: {}, Recon. Val Loss: {}'.format(val_best,
                                                                                                       val_loss_recon))
                    else:
                        print('Model was not saved ! Best Recon. Val Loss: {} Recon. Val Loss: {}'.format(val_best,
                                                                                                          val_loss_recon
                                                                                                          ))

        return global_step, loss, val_best

    def validation(args, epoch_iterator_val):
        model.eval()
        device = torch.device(f"cuda:{args.local_rank}")
        torch.cuda.set_device(device)
        loss_val = []
        loss_val_recon = []
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs = (batch["image"].to(device))
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
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
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (loss=%2.5f)" % (global_step, 9.0, loss))
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (loss_recon=%2.5f)" % (global_step, 10.0, loss_recon))

        return np.mean(loss_val), np.mean(loss_val_recon)

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
    parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
    parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=1e-5, type=float, help='decay rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lrdecay', action='store_true', help='enable learning rate decay')
    parser.add_argument('--normal_dataset', action='store_true', help='use monai Dataset class')
    parser.add_argument('--smartcache_dataset', default=True, action='store_true', help='use monai smartcache Dataset class')
    parser.add_argument('--amp', action='store_true', help='enable using amp')
    parser.add_argument('--amp_scale', action='store_true', help='amp scale')
    parser.add_argument('--amp_scaler', default=2 ** 20, type=float, help='amp scaler')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='maximum gradient norm')
    parser.add_argument('--opt_level', default='O2', type=str)
    parser.add_argument('--loss_type', default='SSL', type=str)
    parser.add_argument('--opt', default='adamw', type=str, help='optimization algorithm')
    parser.add_argument('--lr_schedule', default='warmup_cosine', type=str)
    parser.add_argument('--resume', default=None, type=str, help='resume training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--torch_amp', action='store_true', help='enable torch amp')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true', help='enable distributed training')
    args = parser.parse_args()

    logdir = './runs/' + args.logdir
    torch.backends.cudnn.benchmark = True
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url)

    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    try:
        local_rank = dist.get_rank()
    except:
        local_rank = -1
    if local_rank == 0 or local_rank == -1:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(logdir)

    model = SSLHead(args)
    model.to(device)

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

    if args.amp:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.opt_level)
        if args.amp_scale:
            amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20
            
    torchScaler = None
    if args.torch_amp:
        torchScaler = GradScaler()
            
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

    loss_function = Loss(device,
                         args.batch_size * args.sw_batch_size,
                         args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[device])
    train_loader, test_loader = get_loader(args)

    global_step = 0
    best_val = 1e8
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args,
                                            global_step,
                                            train_loader,
                                            best_val,
                                            torchScaler)
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

