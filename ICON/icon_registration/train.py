from datetime import datetime

import torch
import tqdm

from .losses import ICONLoss, to_floats
import icon_registration.config

def write_stats(writer, stats: ICONLoss, ite):
    for k, v in to_floats(stats)._asdict().items():
        writer.add_scalar(k, v, ite)


def train_batchfunction(
    net,
    optimizer,
    make_batch,
    steps=100000,
    step_callback=(lambda net: None),
    unwrapped_net=None,
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    import footsteps
    from torch.utils.tensorboard import SummaryWriter

    if unwrapped_net is None:
        unwrapped_net = net

    loss_curve = []
    writer = SummaryWriter(
        footsteps.output_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )

    visualization_moving, visualization_fixed = [m[:4] for m in make_batch()]
    for iteration in range(0, steps):
        optimizer.zero_grad()
        moving_image, fixed_image = make_batch()
        loss_object = net(moving_image, fixed_image)
        loss = torch.mean(loss_object.all_loss)
        loss.backward()

        step_callback(unwrapped_net)

        print(to_floats(loss_object))
        write_stats(writer, loss_object, iteration)
        optimizer.step()

        if iteration % 300 == 0:
            torch.save(
                optimizer.state_dict(),
                footsteps.output_dir + "optimizer_weights_" + str(iteration),
            )
            torch.save(
                unwrapped_net.regis_net.state_dict(),
                footsteps.output_dir + "network_weights_" + str(iteration),
            )
            unwrapped_net.eval()
            print("val (from train set)")
            warped = []
            with torch.no_grad():
                for i in range(4):
                    print( unwrapped_net(visualization_moving[i:i + 1], visualization_fixed[i:i + 1]))
                    warped.append(unwrapped_net.warped_image_A.cpu())
                warped = torch.cat(warped)
            unwrapped_net.train()

            def render(im):
                if len(im.shape) == 5:
                    im = im[:, :, :, :, im.shape[4] // 2]
                if torch.min(im) < 0:
                    im = im - torch.min(im)
                if torch.max(im) > 1:
                    im = im / torch.max(im)
                return im[:4, [0, 0, 0]].detach().cpu()

            writer.add_images(
                "moving_image", render(visualization_moving[:4]), iteration, dataformats="NCHW"
            )
            writer.add_images(
                "fixed_image", render(visualization_fixed[:4]), iteration, dataformats="NCHW"
            )
            writer.add_images(
                "warped_moving_image",
                render(warped),
                iteration,
                dataformats="NCHW",
            )
            writer.add_images(
                "difference",
                render(torch.clip((warped[:4, :1] - visualization_fixed[:4, :1].cpu()) + 0.5, 0, 1)),
                iteration,
                dataformats="NCHW",
            )

            


def train_datasets(net, optimizer, d1, d2, epochs=400):
    """A training function for quick experiments"""
    batch_size = net.identity_map.shape[0]
    loss_history = []
    for epoch in tqdm.tqdm(range(epochs)):
        for A, B in list(zip(d1, d2)):
            if True:  # A[0].size()[0] == batch_size:
                image_A = A[0].to(icon_registration.config.device)
                image_B = B[0].to(icon_registration.config.device)
                optimizer.zero_grad()

                loss_object = net(image_A, image_B)

                loss_object.all_loss.backward()
                optimizer.step()

            loss_history.append(to_floats(loss_object))
    return loss_history


train2d = train_datasets
