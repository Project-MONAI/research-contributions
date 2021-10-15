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

import numpy as np
import torch
import argparse
from tensorboardX import SummaryWriter
from apex import amp
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete,Activations,Compose
from optimizers.lr_scheduler import WarmupCosineSchedule
from tqdm import tqdm
from utils.data_utils import get_loader
from networks.unetr import UNETR

def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(global_step,train_loader,dice_val_best):
        model.train()
        epoch_iterator = tqdm(train_loader,desc="Training (X / X Steps) (loss=X.X)",dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))
            global_step += 1
            if global_step % args.eval_num == 0 and global_step!=0:
                epoch_iterator_val = tqdm(test_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val)
                writer.add_scalar("Validation/Mean Dice", scalar_value=dice_val, global_step=global_step)
                writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)
                if dice_val > dice_val_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, logdir + '/model.pt')
                    dice_val_best = dice_val
                    print('Model Was Saved ! Current Best Dice: {} Current Dice: {}'.format(dice_val_best, dice_val))
                else:
                    print('Model Was NOT Saved ! Current Best Dice: {} Current Dice: {}'.format(dice_val_best,dice_val))

        return global_step, dice_val_best

    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = list()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice))
            dice_metric.reset()
        mean_dice_val = np.mean(dice_vals)
        return mean_dice_val

    parser = argparse.ArgumentParser(description='UNETR Training')
    parser.add_argument('--logdir', default=None,type=str)
    parser.add_argument('--pos_embedd', default='perceptron', type=str)
    parser.add_argument('--norm_name', default='instance', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_steps', default=10000, type=int)
    parser.add_argument('--eval_num', default=500, type=int)
    parser.add_argument('--warmup_steps', default=500, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--mlp_dim', default=3072, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--feature_size', default=16, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=14, type=int)
    parser.add_argument('--num_classes', default=14, type=int)
    parser.add_argument('--res_block', action='store_true')
    parser.add_argument('--conv_block', action='store_true')
    parser.add_argument('--roi_x', default=96, type=int)
    parser.add_argument('--roi_y', default=96, type=int)
    parser.add_argument('--roi_z', default=96, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--sw_batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lrdecay', action='store_true')
    parser.add_argument('--clara_split', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--amp_scale', action='store_true')
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--ngc', action='store_true')
    parser.add_argument('--model_type', default='unetr', type=str)
    parser.add_argument('--opt_level', default='O2', type=str)
    parser.add_argument('--loss_type', default='dice_ce', type=str)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--name', default='test', type=str)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        feature_size=args.feature_size,
        hidden_size=args.hidden_size,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        pos_embed=args.pos_embedd,
        norm_name=args.norm_name,
        conv_block=args.conv_block,
        res_block=True,
        dropout_rate=0.0).to(device)

    logdir = './runs/' + args.name
    writer = SummaryWriter(logdir=logdir)

    if args.opt == "adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr,weight_decay= args.decay)

    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(params = model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.amp:
        model, optimizer = amp.initialize(models=model,optimizers=optimizer,opt_level=args.opt_level)
        if args.amp_scale:
            amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    if args.lrdecay:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

    if args.loss_type == 'dice':
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    elif args.loss_type == 'dice_ce':
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    train_loader, test_loader = get_loader(args)
    global_step = 0
    dice_val_best = 0.0

    while global_step < args.num_steps:
        global_step, dice_val_best = train(global_step,train_loader,dice_val_best)
    checkpoint = {'global_step': global_step,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    save_ckp(checkpoint, logdir+'/model_final_epoch.pt')

if __name__ == '__main__':
    main()
