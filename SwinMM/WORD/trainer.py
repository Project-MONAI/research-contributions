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
import time

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import view_ops
from utils.misc import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(model, loader, optimizer, scaler, epoch, self_crit, mutual_crit, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_self_loss = AverageMeter()
    run_mutual_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        for param in model.parameters():
            param.grad = None
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data = data.cuda(args.rank)
        if not args.unsupervised:
            target = target.cuda(args.rank)
        data_list, view_list = view_ops.permute_rand(data)

        loss = 0
        self_loss_list, mutual_loss_list = [], []
        with autocast(enabled=args.amp):
            output1, output2 = model(data_list[0], data_list[1], view_list)
            out_list = [output1, output2]
            out_list = view_ops.permute_inverse(out_list, view_list)
            if args.unsupervised:
                target = torch.argmax(
                    (torch.softmax(out_list[0], dim=1) + torch.softmax(out_list[1], dim=1)) / 2, dim=1, keepdim=True
                ).cuda(args.rank)
            for i in range(len(out_list)):
                self_loss = self_crit(out_list[i], target)
                mutual_loss = 0
                for j in range(len(out_list)):  # KL divergence
                    if i != j:
                        mutual_end = mutual_crit(F.log_softmax(out_list[i], dim=1), F.softmax(out_list[j], dim=1))
                        mutual_loss += mutual_end
                loss = loss + (self_loss + mutual_loss / (len(out_list) - 1)) / len(out_list)
                self_loss_list.append(self_loss.item())
                mutual_loss_list.append(mutual_loss.item())
            self_loss = torch.mean(torch.tensor(self_loss_list)).cuda(args.rank)
            mutual_loss = torch.mean(torch.tensor(mutual_loss_list)).cuda(args.rank)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            is_valid = True
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=is_valid)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
            self_loss_list = distributed_all_gather([self_loss], out_numpy=True, is_valid=is_valid)
            run_self_loss.update(
                np.mean(np.mean(np.stack(self_loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
            mutual_loss_list = distributed_all_gather([mutual_loss], out_numpy=True, is_valid=is_valid)
            run_mutual_loss.update(
                np.mean(np.mean(np.stack(mutual_loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size,
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_self_loss.update(self_loss.item(), n=args.batch_size)
            run_mutual_loss.update(mutual_loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "self_loss: {:.4f}".format(run_self_loss.avg),
                "mutual_loss: {:.4f}".format(run_mutual_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, run_self_loss.avg, run_mutual_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data = data.cuda(args.rank)
            torch.cuda.empty_cache()
            with autocast(enabled=args.amp):
                i = np.random.randint(0, 3)
                if model_inferer is not None:
                    output1, output2 = model_inferer(data, i)
                else:
                    output1, output2 = model(data, i)
            output1, output2, target = output1.cpu(), output2.cpu(), target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            out = (output1 + output2) / 2
            val_outputs_list = decollate_batch(out)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc, not_nans = acc.cuda(args.rank), not_nans.cuda(args.rank)

            if args.distributed:
                is_valid = True
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True, is_valid=is_valid)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    unsupervised_loader,
    optimizer,
    self_crit,
    mutual_crit,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, self_loss, mutual_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            self_crit=self_crit,
            mutual_crit=mutual_crit,
            args=args,
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "self loss: {:.4f}".format(self_loss),
                "mutual loss: {:.4f}".format(mutual_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("self_loss", self_loss, epoch)
            writer.add_scalar("mutual_loss", mutual_loss, epoch)

        if args.unsupervised and (epoch + 1) % args.unsuper_every == 0:
            if args.distributed:
                unsupervised_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            print(args.rank, time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss, mutual_loss = train_epoch(
                model,
                unsupervised_loader,
                optimizer,
                scaler=scaler,
                epoch=epoch,
                self_crit=self_crit,
                mutual_crit=mutual_crit,
                args=args,
            )
            if args.rank == 0:
                print(
                    "Final unsupervised training  {}/{}".format(epoch, args.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "mutual loss: {:.4f}".format(mutual_loss),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
            if args.rank == 0 and writer is not None:
                writer.add_scalar("train_unsupervised_loss", train_loss, epoch)
                writer.add_scalar("unsupervised_self_loss", self_loss, epoch)
                writer.add_scalar("unsupervised_mutual_loss", mutual_loss, epoch)

        if epoch >= args.val_start and (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
